"""Where each OOD set lies relative to the ID classes, from the cached features.

Reads <out-root>/features/<experiment>/{train,test_1,test_k}.pt, which the twin
runner wrote, and writes <out-root>/placement_stats/<experiment>.json with five
descriptors per OOD mode, each relative to the model's ID test set. CPU only.
Protocol: ../documentation/placement_explanation_protocol_2026-11-09.md

Run from code/ on the HPC:

    python reparam/placement_stats.py --out-root "$TWIN_ROOT" \
        --experiments-file reparam/experiments_vit.txt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reparam.run_twin import ood_modes

#: The test file of each OOD mode in csf_eval.py at the pinned commit. Both CIFAR
#: modes read test_3, so on TinyImageNet they are the same set.
MODE_FILES = {
    "ood_sncs_c10": "test_3", "ood_sncs_c100": "test_3", "ood_nsncs_svhn": "test_4",
    "ood_nsncs_ti": "test_5", "ood_nsncs_lsun_cropped": "test_6",
    "ood_nsncs_lsun_resize": "test_7", "ood_nsncs_isun": "test_8",
    "ood_nsncs_textures": "test_9", "ood_nsncs_places365": "test_10",
}


def class_geometry(features: torch.Tensor, labels: torch.Tensor
                   ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Global mean, centered class means, and an orthonormal basis of their span."""
    features = features.double()
    classes, index = labels.unique(return_inverse=True)
    sums = torch.zeros(len(classes), features.shape[1], dtype=features.dtype)
    sums.index_add_(0, index, features)
    means = sums / torch.bincount(index).double()[:, None]
    global_mean = features.mean(0)
    centered = means - global_mean
    left, singular, _ = torch.linalg.svd(centered.T, full_matrices=False)
    rank = int((singular > 1e-10 * singular[0]).sum())
    return global_mean, centered, left[:, :rank]


def set_summary(features: torch.Tensor, logits: torch.Tensor,
                geometry: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
                ) -> dict[str, float]:
    """Summary statistics of one evaluation set under the model's class geometry."""
    global_mean, centered_means, basis = geometry
    x = features.double() - global_mean
    norms = x.norm(dim=1).clamp_min(1e-12)
    inside = (x @ basis).pow(2).sum(dim=1)
    unit_means = centered_means / centered_means.norm(dim=1, keepdim=True)
    cosines = (x / norms[:, None]) @ unit_means.T
    top = logits.double().topk(2, dim=1).values
    return {
        "norm": float(norms.median()),
        "outside_share": float((1 - inside / norms.pow(2)).mean()),
        "class_cosine": float(cosines.max(dim=1).values.median()),
        "margin": float((top[:, 0] - top[:, 1]).median()),
        "centroid": float(x.mean(0).norm()),
    }


def placement_descriptors(reference: dict[str, float], ood: dict[str, float],
                          radius: float) -> dict[str, float]:
    """The five descriptors of the protocol, each relative to the ID test set."""
    return {
        "norm_ratio": ood["norm"] / reference["norm"],
        "outside_share_gap": ood["outside_share"] - reference["outside_share"],
        "class_cosine_gap": ood["class_cosine"] - reference["class_cosine"],
        "margin_ratio": ood["margin"] / max(reference["margin"], 1e-12),
        "centroid_offset": (ood["centroid"] - reference["centroid"]) / radius,
    }


def model_placement(feature_dir: Path, modes: list[str]) -> dict[str, dict]:
    """Descriptors of every OOD mode of one cached checkpoint."""
    train = torch.load(feature_dir / "train.pt", map_location="cpu")
    geometry = class_geometry(train["encoded"], train["labels"])
    radius = float(geometry[1].norm(dim=1).mean())
    test = torch.load(feature_dir / "test_1.pt", map_location="cpu")
    reference = set_summary(test["encoded"], test["logits"], geometry)
    summaries: dict[str, dict[str, float]] = {}
    out = {}
    for mode in modes:
        name = MODE_FILES[mode]
        if name not in summaries:
            ood = torch.load(feature_dir / f"{name}.pt", map_location="cpu")
            summaries[name] = set_summary(ood["encoded"], ood["logits"], geometry)
        out[mode] = {"file": name,
                     **placement_descriptors(reference, summaries[name], radius)}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--experiments-file", required=True)
    args = parser.parse_args()
    root = Path(args.out_root)
    experiments = [line.strip() for line in Path(args.experiments_file).read_text()
                   .splitlines() if line.strip() and not line.startswith("#")]
    failures = []
    for experiment in experiments:
        feature_dir = root / "features" / experiment
        target = root / "placement_stats" / f"{experiment}.json"
        try:
            stats = model_placement(feature_dir, ood_modes(experiment))
        except FileNotFoundError as error:
            failures.append(f"{experiment}: {error}")
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(stats, indent=2))
        print(f"[placement] {experiment}", flush=True)
    print("[placement] failures: " + ("; ".join(failures) if failures else "none"))


if __name__ == "__main__":
    main()
