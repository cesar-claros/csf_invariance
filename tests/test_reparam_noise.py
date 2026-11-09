"""Exact subset search, random-set regret, and the headroom reading."""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
import pytest

from reparam.analyze_noise import (
    best_subset,
    headroom_table,
    model_rows,
    random_regret,
    reading,
    set_regret,
)


def test_best_subset_matches_brute_force() -> None:
    regret = np.random.default_rng(0).random((6, 9))
    found = best_subset(regret, 3)
    scores = {subset: regret[:, list(subset)].min(axis=1).mean()
              for subset in combinations(range(9), 3)}
    assert scores[found] == pytest.approx(min(scores.values()))


def test_random_regret_matches_enumeration() -> None:
    regret = np.random.default_rng(1).random((4, 7))
    expected = np.mean([regret[:, list(s)].min(axis=1)
                        for s in combinations(range(7), 3)], axis=0)
    assert random_regret(regret, 3) == pytest.approx(expected)


def test_set_and_random_regret_of_empty_sets_use_the_worst_regret() -> None:
    regret = np.array([[0.0, 2.0, 5.0]])
    assert set_regret(regret, []).tolist() == [5.0]
    assert random_regret(regret, 0).tolist() == [5.0]


def wide(values: list[list[float]]) -> pd.DataFrame:
    index = pd.MultiIndex.from_tuples(
        [("m", f"d{i}", "near") for i in range(len(values))],
        names=["model_id", "eval_dataset", "regime"])
    return pd.DataFrame(values, index=index, columns=["a", "b", "c", "d"])


def test_identical_extractions_have_no_cross_regret() -> None:
    p = wide([[1.0, 2.0, 3.0, 4.0], [2.0, 1.0, 3.0, 4.0]])
    rows = model_rows(p, p, ["c", "d"])
    assert rows["o"].tolist() == [0.0, 0.0]
    assert rows["same_best"].all()
    assert (rows["r"] > 0).all()


def test_swapped_extraction_gives_cross_regret() -> None:
    p = wide([[1.0, 5.0, 5.0, 5.0], [1.0, 5.0, 5.0, 5.0]])
    r = wide([[5.0, 1.0, 5.0, 5.0], [5.0, 1.0, 5.0, 5.0]])
    rows = model_rows(p, r, ["c"])
    assert rows["o"].tolist() == [4.0, 4.0]
    assert not rows["same_best"].any()


@pytest.mark.parametrize(("low", "high", "expected"), [
    (0.0, 0.2, "noise explains most of the failure"),
    (0.8, 0.95, "noise explains little of the failure"),
    (0.2, 0.9, "partial"),
])
def test_reading(low: float, high: float, expected: str) -> None:
    assert reading(low, high) == expected


def test_headroom_table_reports_each_regime() -> None:
    rows = pd.DataFrame([{"model_id": f"m{m}", "regime": regime, "o": 1.0, "q": 4.0,
                          "r": 3.0, "spearman": 0.9, "same_best": True}
                         for m in range(5) for regime in ("near", "mid", "far")])
    table = headroom_table(rows)
    assert table["H"].tolist() == pytest.approx([0.75] * 3)
    assert table["regime"].tolist() == ["near", "mid", "far"]
