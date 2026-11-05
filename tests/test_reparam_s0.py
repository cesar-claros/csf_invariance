"""Stage S0 descriptors, pre-checks, and the stats-only marker fix."""
from __future__ import annotations

import json
import subprocess
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from reparam.analyze_s0 import (
    decisions,
    family_and_source,
    load_stats,
    median_profiles,
    precheck,
    shape_distance,
)
from reparam.run_twin import (
    PROFILE_POINTS,
    bulk_basis,
    run_checkpoint,
    spectrum_stats,
)


def labelled_features(variances: torch.Tensor, n: int = 30000,
                      classes: int = 3) -> tuple[torch.Tensor, torch.Tensor,
                                                 torch.Tensor]:
    """Class means on the first axes, noise with the given per-axis variances."""
    generator = torch.Generator().manual_seed(0)
    dim = len(variances)
    labels = torch.arange(n) % classes
    means = torch.zeros(classes, dim)
    means[torch.arange(classes), torch.arange(classes)] = 5.0
    noise = torch.randn(n, dim, generator=generator) * variances.sqrt()
    weights = torch.zeros(classes, dim)
    weights[:, :classes] = torch.randn(classes, classes, generator=generator)
    return means[labels] + noise, labels, weights


def test_bulk_basis_is_orthonormal_and_orthogonal_to_the_span() -> None:
    means = torch.tensor([[1.0, 0, 0, 0], [0, 1.0, 0, 0]])
    weights = torch.tensor([[1.0, 1.0, 0, 0]])
    basis = bulk_basis(means.double(), weights.double())
    assert basis.shape == (4, 2)
    assert torch.allclose(basis.T @ basis, torch.eye(2, dtype=basis.dtype))
    assert torch.allclose(means.double() @ basis, torch.zeros(2, 2, dtype=basis.dtype))


def test_isotropic_bulk_gives_a_flat_profile() -> None:
    features, labels, weights = labelled_features(torch.ones(20))
    stats = spectrum_stats(features, labels, weights)
    # The empirical class means carry noise, so with the 3 head rows they span
    # (classes - 1) + classes = 5 directions.
    assert stats["bulk_dim"] == 15
    assert stats["bulk_share"] == pytest.approx(15 / 20, abs=0.01)
    assert stats["bulk_effective_rank"] == pytest.approx(1.0, abs=0.01)
    assert len(stats["bulk_profile"]) == PROFILE_POINTS
    assert max(stats["bulk_profile"]) / min(stats["bulk_profile"]) < 1.1


def test_decaying_bulk_and_dead_units() -> None:
    variances = torch.cat([torch.ones(3), 1.0 / torch.arange(1, 14), torch.zeros(4)])
    features, labels, weights = labelled_features(variances)
    stats = spectrum_stats(features, labels, weights)
    assert stats["bulk_dim"] == 15
    assert stats["bulk_floor_fraction"] == pytest.approx(4 / 15)
    profile = stats["bulk_profile"]
    assert profile[0] > 5 * profile[-1]
    assert np.mean(profile) == pytest.approx(1.0, abs=0.1)


@pytest.mark.parametrize(("experiment", "expected"), [
    ("cifar10_paper_sweep/devries_bbresnet18_do0_run1_rew2.2", ("ResNet18", "cifar10")),
    ("supercifar_paper_sweep/dg_bbvgg13_do0_run1_rew10", ("VGG13", "supercifar100")),
    ("vit/super_cifar100_modelvit_bbvit_lr0.003_bs128_run0_do0_rew0",
     ("ViT", "supercifar100")),
    ("vit/tiny-imagenet-200_modelvit_bbvit_lr0.01_bs128_run0_do0_rew0",
     ("ViT", "tinyimagenet")),
])
def test_family_and_source(experiment: str, expected: tuple[str, str]) -> None:
    assert family_and_source(experiment) == expected


def test_shape_distance_is_zero_for_equal_and_symmetric() -> None:
    first, second = np.linspace(2, 0.5, 100), np.linspace(1.5, 0.7, 100)
    assert shape_distance(first, first) == 0.0
    assert shape_distance(first, second) == shape_distance(second, first) > 0


def profiles_for(vit_offset: float) -> dict[tuple[str, str], np.ndarray]:
    base = np.linspace(2.0, 0.5, 100)
    profiles = {}
    for source in ("cifar10", "cifar100", "supercifar100", "tinyimagenet"):
        profiles["VGG13", source] = base
        profiles["ResNet18", source] = base * 1.05
        profiles["ViT", source] = base * np.exp(vit_offset)
    return profiles


def test_prechecks_follow_the_separation() -> None:
    assert decisions(precheck(profiles_for(0.5))) == {"S1": True, "S2": True}
    assert decisions(precheck(profiles_for(0.01))) == {"S1": False, "S2": False}


def test_prechecks_need_all_four_sources() -> None:
    profiles = profiles_for(0.5)
    del profiles["ViT", "tinyimagenet"]
    assert decisions(precheck(profiles)) == {"S1": False, "S2": False}


def test_load_stats_skips_files_without_descriptors(tmp_path: Path) -> None:
    folder = tmp_path / "cifar10_paper_sweep"
    folder.mkdir()
    old = {"zero_fraction": 0.3}
    new = {**old, "bulk_profile": [1.0] * PROFILE_POINTS, "bulk_share": 0.5}
    (folder / "devries_bbresnet18_do0_run1_rew2.2.json").write_text(json.dumps(new))
    (folder / "confidnet_bbvgg13_do0_run1_rew2.2.json").write_text(json.dumps(old))
    frame = load_stats(tmp_path)
    assert frame[["family", "source"]].values.tolist() == [["ResNet18", "cifar10"]]
    assert set(median_profiles(frame)) == {("ResNet18", "cifar10")}


def test_stats_only_writes_no_step_markers(tmp_path: Path,
                                           monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(subprocess, "run",
                        lambda *args, **kwargs: Namespace(returncode=0))
    args = Namespace(out_root=str(tmp_path), stats_only=True)
    assert run_checkpoint("vit/cifar10_modelvit_bbvit_lr0.0003_bs128_run0_do1_rew0",
                          "none", args)
    assert not (tmp_path / "done").exists()


def test_full_runs_still_write_step_markers(tmp_path: Path,
                                            monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(subprocess, "run",
                        lambda *args, **kwargs: Namespace(returncode=0))
    args = Namespace(out_root=str(tmp_path), stats_only=False)
    experiment = "cifar10_paper_sweep/devries_bbresnet18_do0_run1_rew2.2"
    assert run_checkpoint(experiment, "none", args)
    markers = list((tmp_path / "done" / "none" / experiment).iterdir())
    assert len(markers) == 2 + 2 * 8


def test_profile_frame_round_trip() -> None:
    frame = pd.DataFrame({"family": ["ViT", "ViT"], "source": ["cifar10"] * 2,
                          "bulk_profile": [[1.0, 2.0], [3.0, 4.0]]})
    assert median_profiles(frame)["ViT", "cifar10"].tolist() == [2.0, 3.0]
