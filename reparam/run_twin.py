"""Run the CSF pipeline on the re-origined twin of a checkpoint.

The twin is the same classifier in shifted feature coordinates:

    x' = x + c,  W' = W,  b' = b - W c,  c = -(mean training feature)

The unchanged `csf_fit.py` and `csf_eval.py` run through patches on `src.utils`,
so every CSF sees the twin. Results go to <out-root>/<arm>/<experiment>/, and the
checkpoint's own params/ and analysis/ stay untouched. Features are cached in
<out-root>/features/<experiment>/ and shared by the arms, so the arms differ only
by the shift. The paper code is a read-only submodule at external/ood_systematic.
Protocol: ../documentation/reparam_invariance_protocol_2026-11-02.md

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

import torch

PAPER_CODE = Path(__file__).resolve().parents[1] / "external" / "ood_systematic"
if str(PAPER_CODE) not in sys.path:
    sys.path.insert(0, str(PAPER_CODE))

from pilot_from_configs import filter_modes_to_paper, read_manifest

logger = logging.getLogger(__name__)

FAMILIES = {
    "plain": ("CTM,GEN,REN,NNGuide,fDBD,MahalanobisDistance,MahalanobisPP,NCI,pNML,"
              "ViM,Residual,NeCo,NeuralCollapse,MSR,MLS,PE,PCE,GE,Energy,GradNorm,"
              "Confidence"),
    "global": "PCA_RecError,KPCA_RecError",
}
COMMON_FLAGS = ("--no-rank_weight", "--no-rank_feature", "--ash", "None",
                "--use_cuda", "--temperature_scale")
MANIFEST_SOURCE = {"tiny-imagenet-200": "tinyimagenet"}


def apply_shift(evaluation: dict, shift: torch.Tensor) -> dict:
    """Return a copy of one set's evaluations with the features translated."""
    return {**evaluation, "encoded": evaluation["encoded"] + shift}


def compensate_bias(weights: torch.Tensor, bias: torch.Tensor,
                    shift: torch.Tensor) -> torch.Tensor:
    """Return the bias that keeps the logits unchanged under the shift."""
    return bias - weights @ shift.to(weights.device, weights.dtype)


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


class Twin:
    """Feature cache, shift, and compensated head for one checkpoint and arm."""

    def __init__(self, out_root: Path, experiment: str, arm: str,
                 stats_only: bool = False) -> None:
        self.arm = arm
        self.stats_only = stats_only
        self.out_dir = out_root / arm / experiment
        self.feature_dir = out_root / "features" / experiment
        self.stats_path = out_root / "origin_stats" / f"{experiment}.json"
        self.source_dir = None
        self.shift = None
        for folder in (self.out_dir, self.feature_dir, self.stats_path.parent):
            folder.mkdir(parents=True, exist_ok=True)
        if self._mean_path.exists() and not stats_only:
            self.shift = self._shift_from(torch.load(self._mean_path,
                                                     map_location="cpu"))

    @property
    def _mean_path(self) -> Path:
        return self.feature_dir / "train_mean.pt"

    def _shift_from(self, mean: torch.Tensor) -> torch.Tensor:
        return -mean if self.arm == "center" else torch.zeros_like(mean)

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
        stats = origin_stats(train["encoded"], train["labels"], weights)
        self.stats_path.write_text(json.dumps(stats, indent=2))
        if self.stats_only:
            raise SystemExit(0)
        mean = train["encoded"].mean(0)
        torch.save(mean, self._mean_path)
        self.shift = self._shift_from(mean)

    def evaluate(self, compute: Callable, last_layer: Callable, model, datamodule,
                 set_name: str) -> dict:
        """Patched `compute_model_evaluations`: cached, then translated."""
        if self.shift is None:
            self._prepare(compute, last_layer, model, datamodule)
        return apply_shift(self._load(compute, model, datamodule, set_name),
                           self.shift)

    def last_layer(self, layer: tuple) -> tuple:
        """Patched head: compensate the bias once the shift is known.

        `TrainedModule` reads the head before the first evaluation but does not
        use it, so an uncompensated head can only reach that constructor.
        """
        if self.shift is None:
            return layer
        *model, weights, bias = layer
        return (*model, weights, compensate_bias(weights, bias, self.shift))


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
    if "do1" in experiment and not args.stats_only:
        raise SystemExit(f"{experiment}: CSF runs cover dropout-off models only")
    done_dir = Path(args.out_root) / "done" / arm / experiment
    done_dir.mkdir(parents=True, exist_ok=True)
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
        marker.touch()
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--experiments-file")
    parser.add_argument("--arms", nargs="+", default=["none", "center"],
                        choices=["none", "center"])
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
