"""Placement descriptors, nearest-neighbor shortlists, and the Step 1 rule."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from reparam.analyze_placement import (
    base_rate_shortlist,
    decide,
    difference,
    duplicate_rows,
    knn_shortlists,
    picked_regret,
)
from reparam.placement_stats import (
    class_geometry,
    model_placement,
    placement_descriptors,
    set_summary,
)


def features(seed: int, n: int = 600, shift: float = 0.0) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    labels = torch.arange(n) % 3
    means = torch.zeros(3, 10)
    means[torch.arange(3), torch.arange(3)] = 4.0
    noise = torch.randn(n, 10, generator=generator)
    encoded = means[labels] + noise
    encoded[:, 5:] += shift
    logits = encoded[:, :3] * 2
    return {"encoded": encoded, "logits": logits, "labels": labels}


def test_id_against_itself_gives_neutral_descriptors() -> None:
    train, test = features(0), features(1)
    geometry = class_geometry(train["encoded"], train["labels"])
    reference = set_summary(test["encoded"], test["logits"], geometry)
    values = placement_descriptors(reference, reference, 1.0)
    assert values["norm_ratio"] == pytest.approx(1.0)
    assert values["margin_ratio"] == pytest.approx(1.0)
    for name in ("outside_share_gap", "class_cosine_gap", "centroid_offset"):
        assert values[name] == pytest.approx(0.0)


def test_ood_outside_the_class_span_raises_the_outside_share() -> None:
    train, test, ood = features(0), features(1), features(2, shift=3.0)
    geometry = class_geometry(train["encoded"], train["labels"])
    reference = set_summary(test["encoded"], test["logits"], geometry)
    values = placement_descriptors(
        reference, set_summary(ood["encoded"], ood["logits"], geometry), 4.0)
    assert values["outside_share_gap"] > 0.2
    assert values["norm_ratio"] > 1.5
    assert values["centroid_offset"] > 0.5


def test_both_cifar_modes_read_the_same_file(tmp_path: Path) -> None:
    torch.save(features(0), tmp_path / "train.pt")
    torch.save(features(1), tmp_path / "test_1.pt")
    torch.save(features(2, shift=2.0), tmp_path / "test_3.pt")
    out = model_placement(tmp_path, ["ood_sncs_c10", "ood_sncs_c100"])
    assert out["ood_sncs_c10"] == out["ood_sncs_c100"]
    assert out["ood_sncs_c10"]["file"] == "test_3"


def test_knn_shortlist_follows_the_nearest_training_rows() -> None:
    train_x = np.array([[0.0], [0.1], [10.0], [10.1]])
    train_regret = np.array([[0.0, 5.0, 5.0], [0.0, 5.0, 5.0],
                             [5.0, 0.0, 5.0], [5.0, 0.0, 5.0]])
    lists = knn_shortlists(train_x, train_regret, np.array([[0.05], [9.9]]), 2, 1)
    assert lists[:, 0].tolist() == [0, 1]
    regret = np.array([[0.0, 1.0, 2.0], [3.0, 0.0, 1.0]])
    assert picked_regret(regret, lists).tolist() == [0.0, 0.0]


def test_own_scale_matches_rows_by_their_position_in_their_family() -> None:
    train_x = np.array([[0.0], [1.0], [2.0], [3.0]])
    train_regret = np.eye(4)[:, :3] * 5
    test_x = np.array([[100.0], [103.0]])
    far = knn_shortlists(train_x, train_regret, test_x, 1, 1)
    own = knn_shortlists(train_x, train_regret, test_x, 1, 1, own_scale=True)
    assert far[:, 0].tolist() == [far[0, 0], far[0, 0]]
    assert own[0, 0] != own[1, 0]


def test_base_rate_shortlist_is_the_same_for_every_row() -> None:
    train_regret = np.array([[3.0, 0.0, 1.0], [3.0, 1.0, 0.0]])
    lists = base_rate_shortlist(train_regret, rows=3, size=2)
    assert lists.tolist() == [[1, 2]] * 3


def test_duplicate_rows_are_the_tinyimagenet_cifar10_rows() -> None:
    frame = pd.DataFrame({"source": ["tinyimagenet", "tinyimagenet", "cifar100"],
                          "eval_dataset": ["cifar10", "cifar100", "cifar10"]})
    assert duplicate_rows(frame).tolist() == [True, False, False]


def comparisons(highs: dict[str, float], lows: dict[str, float]) -> pd.DataFrame:
    names = ["P - random", "P - B", "P - N"]
    return pd.DataFrame({"comparison": names, "mean": 0.0,
                         "low": [lows[n] for n in names],
                         "high": [highs[n] for n in names]})


@pytest.mark.parametrize(("highs", "lows", "expected"), [
    ({"P - random": -1, "P - B": -0.5, "P - N": -0.2},
     {"P - random": -2, "P - B": -1, "P - N": -0.5}, "P1"),
    ({"P - random": -1, "P - B": 1, "P - N": -0.2},
     {"P - random": -2, "P - B": 0.1, "P - N": -0.5}, "P0"),
    ({"P - random": 1, "P - B": 1, "P - N": 1},
     {"P - random": 0.2, "P - B": -1, "P - N": -1}, "P0"),
    ({"P - random": -1, "P - B": 0.5, "P - N": 0.5},
     {"P - random": -2, "P - B": -0.5, "P - N": -0.5}, "PZ"),
])
def test_decide(highs: dict[str, float], lows: dict[str, float],
                expected: str) -> None:
    assert decide(comparisons(highs, lows)).startswith(expected)


def test_difference_is_paired_by_row() -> None:
    scored = pd.DataFrame({"model_id": [f"m{i // 2}" for i in range(8)],
                           "P": np.arange(8.0), "B": np.arange(8.0) + 1})
    result = difference(scored, "P", "B")
    assert result["mean"] == pytest.approx(-1.0)
    assert result["low"] == pytest.approx(-1.0)
    assert result["high"] == pytest.approx(-1.0)
