"""The pipeline's CSF and NC classes follow the derived invariance table."""
from __future__ import annotations

import importlib.util

import pytest

from reparam.invariance_table import (
    ACTIONS,
    EXPECTED,
    NC_METRICS,
    build_table,
    install_stubs_if_missing,
)

NEEDED = ("torch_pca", "torchmetrics")


@pytest.fixture(scope="module")
def table() -> list[dict]:
    missing = [name for name in NEEDED if importlib.util.find_spec(name) is None]
    if missing:
        pytest.skip(f"not installed: {missing}")
    install_stubs_if_missing()
    return build_table(seed=0)


def test_every_map_keeps_the_logits(table: list[dict]) -> None:
    assert max(row["logits_gap"] for row in table) < 1e-10


def test_every_cell_matches_the_derivation(table: list[dict]) -> None:
    wrong = [(row["row"], row["action"], row["observed"], row["expected"])
             for row in table if row["observed"] != row["expected"]]
    assert not wrong


def test_table_is_complete(table: list[dict]) -> None:
    assert len(table) == len(EXPECTED) * len(ACTIONS)


def test_exact_and_changed_cells_are_well_separated(table: list[dict]) -> None:
    exact = max(row["change"] for row in table if row["observed"] == "=")
    changed = min(row["change"] for row in table if row["observed"] == "x")
    assert exact * 100 < changed


def test_nc_metrics_do_not_see_translation(table: list[dict]) -> None:
    labels = {row["row"]: row["observed"] for row in table
              if row["action"] == "translate"}
    assert all(labels[name] == "=" for name in NC_METRICS)
    assert labels["CTM"] == "x"
