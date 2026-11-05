"""Run the CSF pipeline on a twin of a checkpoint: the same classifier in new
feature coordinates.

Every arm is an affine map x' = A x + c with the head W' = W A^-1, b' = b - W' c,
so the logits stay the same:

    none       A = I, c = 0
    center     A = I, c = -(mean training feature)
    bulk_vgg   bulk spectrum outside the class-mean and head span gets the
    bulk_vit   median VGG-13 or ViT shape of the source (NC metrics unchanged)
    bulk_away  the shape moves away from VGG-13 by the same distance
    unitscale  A = diag(mean(sigma) / sigma_j) (NC metrics move)

The unchanged `csf_fit.py` and `csf_eval.py` run through patches on `src.utils`,
so every CSF sees the twin. Results go to <out-root>/<arm>/<experiment>/, and the
checkpoint's own params/ and analysis/ stay untouched. Features are cached in
<out-root>/features/<experiment>/ and shared by the arms, so the arms differ only
by the map. The paper code is a read-only submodule at external/ood_systematic.
Protocols: ../documentation/reparam_invariance_protocol_2026-11-02.md and
../documentation/spectrum_unit_scale_protocol_2026-11-04.md

Run from code/ inside the paper container, with .env sourced:

    python reparam/run_twin.py --experiments-file reparam/experiments_gate.txt \
        --arms none center --out-root "$TWIN_ROOT"
    python reparam/run_twin.py --experiments-file reparam/experiments_stats.txt \
        --stats-only --out-root "$TWIN_ROOT"
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch

PROJECT_CODE = Path(__file__).resolve().parents[1]
PAPER_CODE = PROJECT_CODE / "external" / "ood_systematic"
for _path in (PROJECT_CODE, PAPER_CODE):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from pilot_from_configs import filter_modes_to_paper, read_manifest

from reparam.analyze_s0 import family_and_source

logger = logging.getLogger(__name__)

FAMILIES = {
    "plain": ("CTM,GEN,REN,NNGuide,fDBD,MahalanobisDistance,MahalanobisPP,NCI,pNML,"
              "ViM,Residual,NeCo,NeuralCollapse,MSR,MLS,PE,PCE,GE,Energy,GradNorm,"
              "Confidence"),
    "global": "PCA_RecError,KPCA_RecError",
}
COMMON_FLAGS = ("--no-rank_weight", "--no-rank_feature", "--ash", "None",
                "--use_cuda", "--temperature_scale")
MANIFEST_SOURCE = {"tiny-imagenet-200": "tinyimagenet", "super_cifar100": "supercifar"}
PROFILE_POINTS = 100
BULK_FLOOR = 1e-8
SCALE_FLOOR = 1e-6
#: Arms and, for the transplant arms, the family whose median profile is the target.
ARMS = ("none", "center", "bulk_vgg", "bulk_vit", "bulk_away", "unitscale")
BULK_TARGETS = {"bulk_vgg": "VGG13", "bulk_vit": "ViT", "bulk_away": "VGG13"}
PROFILES = Path(__file__).resolve().parent / "outputs" / "s0_profiles.json"


def apply_shift(evaluation: dict, shift: torch.Tensor) -> dict:
    """Return a copy of one set's evaluations with the features translated."""
    return {**evaluation, "encoded": evaluation["encoded"] + shift}


def compensate_bias(weights: torch.Tensor, bias: torch.Tensor,
                    shift: torch.Tensor) -> torch.Tensor:
    """Return the bias that keeps the logits unchanged under the shift."""
    return bias - weights @ shift.to(weights.device, weights.dtype)


def apply_map(evaluation: dict, matrix: torch.Tensor | None,
              shift: torch.Tensor) -> dict:
    """Apply x -> x @ matrix.T + shift to one set's features.

    With no matrix this is `apply_shift`, so the translation arms stay unchanged.
    """
    if matrix is None:
        return apply_shift(evaluation, shift)
    encoded = evaluation["encoded"]
    moved = encoded.double() @ matrix.T + shift
    return {**evaluation, "encoded": moved.to(encoded.dtype)}


def compensate_head(weights: torch.Tensor, bias: torch.Tensor,
                    inverse: torch.Tensor | None,
                    shift: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Head that keeps the logits unchanged: W' = W A^-1 and b' = b - W' c.

    Args:
        weights: Head weights, shape (rows, d).
        bias: Head bias, shape (rows,).
        inverse: A^-1 in float64, or None for a translation.
        shift: The translation c.
    """
    if inverse is None:
        return weights, compensate_bias(weights, bias, shift)
    new = (weights.double().cpu() @ inverse).to(weights.dtype).to(weights.device)
    return new, bias - new @ shift.to(new.device, new.dtype)


def origin_stats(encoded: torch.Tensor, labels: torch.Tensor,
                 weights: torch.Tensor) -> dict[str, float]:
    """Describe the feature origin and marginals of one checkpoint.

    Args:
        encoded: Training features, shape (n, d).
        labels: Training labels, shape (n,).
        weights: Head weights, shape (classes, d).

    Returns:
        Dimension, sign statistics, median unit skew, and the origin ratios.
    """
    encoded = encoded.double()
    global_mean = encoded.mean(0)
    class_means = torch.stack([encoded[labels == c].mean(0)
                               for c in labels.unique()])
    radius = (class_means - global_mean).norm(dim=1).mean()
    within = (encoded - class_means[torch.searchsorted(labels.unique(), labels)])
    unit_std = encoded.std(0)
    skew = ((encoded - global_mean) ** 3).mean(0) / unit_std.clamp_min(1e-12) ** 3
    weights = weights.double().cpu()
    row_space = torch.linalg.pinv(weights) @ weights
    outside = global_mean - row_space @ global_mean
    return {
        "dim": float(encoded.shape[1]),
        "zero_fraction": float((encoded == 0).double().mean()),
        "negative_fraction": float((encoded < 0).double().mean()),
        "median_unit_skew": float(skew.median()),
        "global_mean_norm": float(global_mean.norm()),
        "class_radius": float(radius),
        "within_radius": float(within.pow(2).sum(1).mean().sqrt()),
        "origin_ratio": float(global_mean.norm() / radius),
        "mean_share_outside_head": float(outside.norm() / global_mean.norm()),
    }


def bulk_basis(class_means: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Orthonormal basis of the complement of span(centered class means, head rows).

    Args:
        class_means: Centered class means, shape (classes, d).
        weights: Head weights, shape (rows, d).

    Returns:
        Basis matrix, shape (d, d_perp).
    """
    spanning = torch.cat([class_means, weights]).T
    left, singular, _ = torch.linalg.svd(spanning, full_matrices=True)
    rank = int((singular > 1e-10 * singular[0]).sum())
    return left[:, rank:]


def spectrum_stats(encoded: torch.Tensor, labels: torch.Tensor,
                   weights: torch.Tensor) -> dict[str, float | list[float]]:
    """Within-class spectrum outside the class-mean and head directions (stage S0).

    Directions with variance below BULK_FLOOR of the largest are counted but left out
    of the profile, so dead units do not dominate it.

    Args:
        encoded: Training features, shape (n, d).
        labels: Training labels, shape (n,).
        weights: Head weights, shape (rows, d).

    Returns:
        Bulk dimension and share, floor fraction, effective rank per dimension, unit
        scale dispersion, and the bulk profile at PROFILE_POINTS quantiles.
    """
    encoded = encoded.double()
    classes, index = labels.unique(return_inverse=True)
    sums = torch.zeros(len(classes), encoded.shape[1], dtype=encoded.dtype)
    sums.index_add_(0, index, encoded)
    class_means = sums / torch.bincount(index).double()[:, None]
    within = encoded - class_means[index]
    sigma_w = within.T @ within / len(encoded)
    basis = bulk_basis(class_means - encoded.mean(0), weights.double().cpu())
    bulk = torch.linalg.eigvalsh(basis.T @ sigma_w @ basis).flip(0).clamp_min(0)
    kept = bulk[bulk > BULK_FLOOR * bulk[0]]
    quantiles = (torch.arange(PROFILE_POINTS) + 0.5) / PROFILE_POINTS
    positions = (quantiles * len(kept)).long()
    unit_std = encoded.std(0)
    return {
        "bulk_dim": float(len(bulk)),
        "bulk_floor_fraction": float(1 - len(kept) / len(bulk)),
        "bulk_share": float(bulk.sum() / sigma_w.trace()),
        "bulk_effective_rank": float(kept.sum() ** 2 / kept.pow(2).sum() / len(bulk)),
        "unit_scale_cv": float(unit_std.std() / unit_std.mean()),
        "bulk_profile": (kept[positions] / kept.mean()).tolist(),
    }


def interpolate_profile(profile: list[float], count: int) -> torch.Tensor:
    """A profile at `count` evenly spaced relative positions, log-linear in between."""
    grid = (np.arange(len(profile)) + 0.5) / len(profile)
    points = (np.arange(count) + 0.5) / count
    return torch.from_numpy(np.exp(np.interp(points, grid, np.log(profile))))


def transplant_map(encoded: torch.Tensor, labels: torch.Tensor,
                   weights: torch.Tensor, target: list[float],
                   away: bool = False) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Bulk map that gives the within-class spectrum outside V the target shape.

    x' = mu_G + (P_V + B P_perp)(x - mu_G), written as x' = A x + c. B keeps the
    eigenvectors and the trace of the bulk spectrum. Directions below BULK_FLOOR
    stay unchanged. With `away`, the shape moves from the target instead:
    log p' = 2 log p - log p_target.

    Args:
        encoded: Training features, shape (n, d).
        labels: Training labels, shape (n,).
        weights: Head weights, shape (rows, d).
        target: Target profile at evenly spaced quantiles.
        away: Move away from the target instead of toward it.

    Returns:
        The matrix A, the shift c, and the range of the scale factors.
    """
    encoded = encoded.double()
    classes, index = labels.unique(return_inverse=True)
    sums = torch.zeros(len(classes), encoded.shape[1], dtype=encoded.dtype)
    sums.index_add_(0, index, encoded)
    class_means = sums / torch.bincount(index).double()[:, None]
    global_mean = encoded.mean(0)
    within = encoded - class_means[index]
    basis = bulk_basis(class_means - global_mean, weights.double().cpu())
    evals, evecs = torch.linalg.eigh(basis.T @ (within.T @ within / len(encoded))
                                     @ basis)
    evals, evecs = evals.flip(0).clamp_min(0), evecs.flip(1)
    kept = evals > BULK_FLOOR * evals[0]
    current = evals[kept]
    shape = interpolate_profile(target, len(current))
    if away:
        shape = (current / current.mean()) ** 2 / shape
    new = shape * current.sum() / shape.sum()
    factors = torch.ones_like(evals)
    factors[kept] = (new / current).sqrt()
    directions = basis @ evecs
    matrix = (torch.eye(encoded.shape[1], dtype=encoded.dtype)
              + directions @ torch.diag(factors - 1) @ directions.T)
    info = {"min_factor": float(factors.min()), "max_factor": float(factors.max()),
            "kept_directions": int(kept.sum())}
    return matrix, global_mean - matrix @ global_mean, info


def unit_scale_map(encoded: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Diagonal map D_jj = mean(sigma) / sigma_j, leaving near-dead units unchanged."""
    std = encoded.double().std(0)
    scale = torch.where(std >= SCALE_FLOOR * std.mean(), std.mean() / std,
                        torch.ones_like(std))
    info = {"min_factor": float(scale.min()), "max_factor": float(scale.max()),
            "kept_directions": int((std >= SCALE_FLOOR * std.mean()).sum())}
    return torch.diag(scale), torch.zeros_like(std), info


class Twin:
    """Feature cache, feature map, and compensated head for one checkpoint and arm."""

    def __init__(self, out_root: Path, experiment: str, arm: str,
                 stats_only: bool = False) -> None:
        self.arm = arm
        self.experiment = experiment
        self.stats_only = stats_only
        self.out_dir = out_root / arm / experiment
        self.feature_dir = out_root / "features" / experiment
        self.stats_path = out_root / "origin_stats" / f"{experiment}.json"
        self.source_dir = None
        self.shift = None
        self.matrix = None
        self.inverse = None
        for folder in (self.out_dir, self.feature_dir, self.stats_path.parent):
            folder.mkdir(parents=True, exist_ok=True)
        if stats_only:
            return
        if arm in ("none", "center") and self._mean_path.exists():
            self.shift = self._shift_from(torch.load(self._mean_path,
                                                     map_location="cpu"))
        elif self._map_path.exists():
            self._set_map(**torch.load(self._map_path, map_location="cpu"))

    @property
    def _mean_path(self) -> Path:
        return self.feature_dir / "train_mean.pt"

    @property
    def _map_path(self) -> Path:
        return self.feature_dir / f"map_{self.arm}.pt"

    def _shift_from(self, mean: torch.Tensor) -> torch.Tensor:
        return -mean if self.arm == "center" else torch.zeros_like(mean)

    def _set_map(self, matrix: torch.Tensor, shift: torch.Tensor) -> None:
        self.matrix, self.shift = matrix, shift
        self.inverse = torch.linalg.inv(matrix)

    def _build_map(self, train: dict, weights: torch.Tensor) -> None:
        """Compute, save, and set the map of a bulk or unit-scale arm."""
        if self.arm == "unitscale":
            matrix, shift, info = unit_scale_map(train["encoded"])
        else:
            family = BULK_TARGETS[self.arm]
            _, source = family_and_source(self.experiment)
            target = json.loads(PROFILES.read_text())[family][source]
            matrix, shift, info = transplant_map(
                train["encoded"], train["labels"], weights, target,
                away=self.arm == "bulk_away")
        torch.save({"matrix": matrix, "shift": shift}, self._map_path)
        (self.out_dir / "map_info.json").write_text(json.dumps(info, indent=2))
        self._set_map(matrix, shift)

    def _load(self, compute: Callable, model, datamodule, set_name: str) -> dict:
        path = self.feature_dir / f"{set_name}.pt"
        if path.exists():
            return torch.load(path, map_location="cpu")
        evaluation = compute(model, datamodule, set_name=set_name)
        if not self.stats_only:
            torch.save(evaluation, path)
        return evaluation

    def _prepare(self, compute: Callable, last_layer: Callable, model,
                 datamodule) -> None:
        train = self._load(compute, model, datamodule, "train")
        weights, _ = last_layer(model.module, model.study_name, return_model=False)
        stats = {**origin_stats(train["encoded"], train["labels"], weights),
                 **spectrum_stats(train["encoded"], train["labels"], weights)}
        self.stats_path.write_text(json.dumps(stats, indent=2))
        if self.stats_only:
            raise SystemExit(0)
        if self.arm not in ("none", "center"):
            self._build_map(train, weights)
            return
        mean = train["encoded"].mean(0)
        torch.save(mean, self._mean_path)
        self.shift = self._shift_from(mean)

    def evaluate(self, compute: Callable, last_layer: Callable, model, datamodule,
                 set_name: str) -> dict:
        """Patched `compute_model_evaluations`: cached, then mapped."""
        if self.shift is None:
            self._prepare(compute, last_layer, model, datamodule)
        return apply_map(self._load(compute, model, datamodule, set_name),
                         self.matrix, self.shift)

    def last_layer(self, layer: tuple) -> tuple:
        """Patched head: compensate the head once the map is known.

        `TrainedModule` reads the head before the first evaluation but does not
        use it, so an uncompensated head can only reach that constructor.
        """
        if self.shift is None:
            return layer
        *model, weights, bias = layer
        return (*model, *compensate_head(weights, bias, self.inverse, self.shift))


def install_patches(twin: Twin) -> None:
    """Point the pipeline at the twin: outputs, features, and head."""
    from fd_shifts.utils import exp_utils
    from src import utils

    get_conf = utils.get_conf
    best_checkpoint = exp_utils._get_path_to_best_ckpt
    compute = utils.compute_model_evaluations
    last_layer = utils.get_model_and_last_layer

    def twin_conf(path: str, study_name: str):
        config = get_conf(path, study_name)
        twin.source_dir = config.exp.dir
        config.exp.dir = type(config.exp.dir)(twin.out_dir)
        return config

    def twin_checkpoint(_exp_dir, *args, **kwargs):
        return best_checkpoint(twin.source_dir, *args, **kwargs)

    def twin_compute(model, datamodule, set_name: str) -> dict:
        return twin.evaluate(compute, last_layer, model, datamodule, set_name)

    def twin_layer(module, study_name: str, return_model: bool = True) -> tuple:
        return twin.last_layer(last_layer(module, study_name,
                                          return_model=return_model))

    utils.get_conf = twin_conf
    exp_utils._get_path_to_best_ckpt = twin_checkpoint
    utils.compute_model_evaluations = twin_compute
    utils.get_model_and_last_layer = twin_layer


def ood_modes(experiment: str) -> list[str]:
    """OOD modes of the paper's suite for this checkpoint's source.

    Raises:
        SystemExit: If a suite dataset has no manifest mode.
    """
    if experiment.startswith("vit/"):
        source = experiment.removeprefix("vit/").split("_modelvit")[0]
    else:
        source = experiment.split("_paper_sweep")[0]
    source = MANIFEST_SOURCE.get(source, source)
    configs = PAPER_CODE / "configs_exp"
    modes: list[str] = []
    for manifest in sorted(configs.glob(f"configs_{source}_ood_test_*.txt")):
        rows = read_manifest(manifest)
        if rows and rows[0][-1] not in modes:
            modes.append(rows[0][-1])
    kept, _, gaps = filter_modes_to_paper(modes, source, PAPER_CODE / "clip_scores")
    if gaps:
        raise SystemExit(f"{experiment}: no manifest mode for {gaps}")
    return kept


def run_step(args: argparse.Namespace) -> None:
    """Run one csf_fit or csf_eval call in this process, with patches installed."""
    twin = Twin(Path(args.out_root), args.model_path, args.arm, args.stats_only)
    install_patches(twin)
    flags = [f"--model_path={args.model_path}", *COMMON_FLAGS,
             "--csfs", FAMILIES[args.families], "--projections", args.families]
    if args.step == "fit":
        import csf_fit as entry
    else:
        import csf_eval as entry
        flags += ["--test_mode", args.test_mode]
    sys.argv = [entry.__file__, *flags]
    entry.main()


def planned_steps(experiment: str, stats_only: bool) -> list[tuple[str, str, str]]:
    """Ordered (step, families, test mode) calls for one checkpoint and arm."""
    if stats_only:
        return [("fit", "plain", "")]
    steps = [("fit", "plain", ""), ("fit", "global", "")]
    for mode in ood_modes(experiment):
        steps += [("eval", "plain", mode), ("eval", "global", mode)]
    return steps


def run_checkpoint(experiment: str, arm: str, args: argparse.Namespace) -> bool:
    """Run all steps of one checkpoint in subprocesses; finished steps are skipped.

    Returns:
        True if every step finished.
    """
    if ("do1" in experiment and not experiment.startswith("vit/")
            and not args.stats_only):
        raise SystemExit(f"{experiment}: CNN CSF runs cover dropout-off models only")
    done_dir = Path(args.out_root) / "done" / arm / experiment
    for step, families, mode in planned_steps(experiment, args.stats_only):
        marker = done_dir / f"{step}_{families}_{mode or 'fit'}"
        if marker.exists() and not args.stats_only:
            continue
        command = [sys.executable, str(Path(__file__).resolve()), "--step", step,
                   "--families", families,
                   "--model_path", experiment, "--arm", arm,
                   "--out-root", args.out_root]
        if mode:
            command += ["--test-mode", mode]
        if args.stats_only:
            command.append("--stats-only")
        print(f"[twin] {arm} {experiment} {step} {families} {mode}", flush=True)
        if subprocess.run(command, cwd=PAPER_CODE, check=False).returncode:
            logger.error("failed: %s %s %s %s %s", arm, experiment, step, families,
                         mode)
            return False
        if not args.stats_only:
            done_dir.mkdir(parents=True, exist_ok=True)
            marker.touch()
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--experiments-file")
    parser.add_argument("--arms", nargs="+", default=["none", "center"],
                        choices=ARMS)
    parser.add_argument("--stats-only", action="store_true")
    parser.add_argument("--step", choices=["fit", "eval"], help=argparse.SUPPRESS)
    parser.add_argument("--families", choices=sorted(FAMILIES),
                        help=argparse.SUPPRESS)
    parser.add_argument("--model_path", help=argparse.SUPPRESS)
    parser.add_argument("--arm", help=argparse.SUPPRESS)
    parser.add_argument("--test-mode", help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.out_root = str(Path(args.out_root).resolve())
    if args.step:
        run_step(args)
        return
    if not args.experiments_file:
        parser.error("--experiments-file is required")
    experiments = [line.strip() for line in Path(args.experiments_file).read_text()
                   .splitlines() if line.strip() and not line.startswith("#")]
    arms = args.arms[:1] if args.stats_only else args.arms
    failures = [f"{arm} {experiment}" for experiment in experiments for arm in arms
                if not run_checkpoint(experiment, arm, args)]
    print("[twin] failures: " + ("; ".join(failures) if failures else "none"))


if __name__ == "__main__":
    main()
