"""Bulk and unit-scale arms, and the S1 and D1 decision rules."""
from __future__ import annotations

import json
import subprocess
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from reparam import run_twin
from reparam.analyze_d1 import (
    check_definition_gate,
    decide_d1,
    definition_check,
    extrapolation_share,
    twin_inputs,
)
from reparam.analyze_s0 import shape_distance
from reparam.analyze_s1 import decide_s1, final_s1, gap_table
from reparam.analyze_twin import NC_METRICS, nc_change
from reparam.invariance_table import synthetic_features
from reparam.run_twin import (
    Twin,
    apply_map,
    compensate_head,
    interpolate_profile,
    ood_modes,
    run_checkpoint,
    spectrum_stats,
    transplant_map,
    unit_scale_map,
)

VIT = "vit/cifar10_modelvit_bbvit_lr0.0003_bs128_run0_do1_rew0"
RESNET = "cifar10_paper_sweep/devries_bbresnet18_do0_run1_rew2.2"


@pytest.fixture(scope="module")
def data() -> dict:
    torch.set_default_dtype(torch.float64)
    try:
        yield synthetic_features(seed=0)
    finally:
        torch.set_default_dtype(torch.float32)


def logits_gap(data: dict, matrix: torch.Tensor, shift: torch.Tensor) -> float:
    moved = apply_map({"encoded": data["eval"]}, matrix, shift)["encoded"]
    w, b = compensate_head(data["w"], data["b"], torch.linalg.inv(matrix), shift)
    return float((moved @ w.T + b - (data["eval"] @ data["w"].T + data["b"]))
                 .abs().max())


def decaying(points: int = 100, power: float = 2.0) -> list[float]:
    profile = (np.arange(points) + 1.0) ** -power
    return list(profile / profile.mean())


def test_transplant_keeps_the_head_and_logits(data: dict) -> None:
    matrix, shift, _ = transplant_map(data["train"], data["labels"], data["w"],
                                      decaying())
    inverse = torch.linalg.inv(matrix)
    assert (data["w"] @ inverse - data["w"]).abs().max() < 1e-10
    assert (data["w"] @ shift).abs().max() < 1e-10
    assert logits_gap(data, matrix, shift) < 1e-10


def test_transplant_gives_the_target_shape_and_keeps_the_trace(data: dict) -> None:
    target = decaying(power=1.0)
    before = spectrum_stats(data["train"], data["labels"], data["w"])
    matrix, shift, _ = transplant_map(data["train"], data["labels"], data["w"], target)
    moved = data["train"] @ matrix.T + shift
    after = spectrum_stats(moved, data["labels"], data["w"])
    assert shape_distance(np.array(after["bulk_profile"]), np.array(target)) < 0.1
    assert shape_distance(np.array(before["bulk_profile"]), np.array(target)) > 0.3
    # The class-mean block is unchanged and the bulk trace is kept, so the bulk
    # share of the within-class variance is unchanged too.
    assert after["bulk_share"] == pytest.approx(before["bulk_share"], rel=1e-8)
    assert after["bulk_dim"] == before["bulk_dim"]


def test_away_moves_the_shape_from_the_target(data: dict) -> None:
    target = decaying(power=1.0)
    before = np.array(spectrum_stats(data["train"], data["labels"],
                                     data["w"])["bulk_profile"])
    matrix, shift, _ = transplant_map(data["train"], data["labels"], data["w"],
                                      target, away=True)
    moved = data["train"] @ matrix.T + shift
    after = np.array(spectrum_stats(moved, data["labels"], data["w"])["bulk_profile"])
    target_array = np.array(target)
    assert shape_distance(after, target_array) > shape_distance(before, target_array)


def test_unit_scale_equalizes_units_and_keeps_logits(data: dict) -> None:
    matrix, shift, _ = unit_scale_map(data["train"])
    moved = data["train"] @ matrix.T
    std = moved.std(0)
    assert (std - std.mean()).abs().max() < 1e-8 * std.mean()
    assert logits_gap(data, matrix, shift) < 1e-9


def test_interpolate_profile_hits_the_grid() -> None:
    profile = [4.0, 2.0, 1.0, 0.5]
    assert interpolate_profile(profile, 4).tolist() == pytest.approx(profile)
    assert interpolate_profile(profile, 7).shape == (7,)


def bulk_compute(data: dict, calls: list[str]):
    def compute(model, datamodule, set_name: str) -> dict:
        calls.append(set_name)
        return {"encoded": data["train"].float(), "labels": data["labels"],
                "logits": (data["train"] @ data["w"].T + data["b"]).float()}
    return compute


def test_bulk_arm_saves_and_reuses_its_map(tmp_path: Path, data: dict,
                                           monkeypatch: pytest.MonkeyPatch) -> None:
    profiles = tmp_path / "profiles.json"
    profiles.write_text(json.dumps({"VGG13": {"cifar10": decaying(power=1.0)}}))
    monkeypatch.setattr(run_twin, "PROFILES", profiles)
    w, b = data["w"].float(), data["b"].float()
    model = SimpleNamespace(module=object(), study_name="devries")
    calls: list[str] = []
    twin = Twin(tmp_path, RESNET, "bulk_vgg")
    moved = twin.evaluate(bulk_compute(data, calls), lambda *a, **k: (w, b), model,
                          None, "val")
    assert (twin.feature_dir / "map_bulk_vgg.pt").exists()
    assert json.loads((twin.out_dir / "map_info.json").read_text())["max_factor"] > 1
    again = Twin(tmp_path, RESNET, "bulk_vgg")
    assert again.matrix is not None and torch.equal(again.matrix, twin.matrix)
    new_w, new_b = again.last_layer((w, b))
    gap = (moved["encoded"] @ new_w.T + new_b) - (data["train"].float() @ w.T + b)
    assert gap.abs().max() < 1e-3


def test_vit_modes_and_dropout_guard(tmp_path: Path,
                                     monkeypatch: pytest.MonkeyPatch) -> None:
    assert len(ood_modes(VIT)) == 8 and "ood_sncs_c100" in ood_modes(VIT)
    tiny = "vit/tiny-imagenet-200_modelvit_bbvit_lr0.01_bs128_run0_do0_rew0"
    assert "ood_nsncs_ti" not in ood_modes(tiny)
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: Namespace(returncode=0))
    args = Namespace(out_root=str(tmp_path), stats_only=False)
    assert run_checkpoint(VIT, "bulk_vgg", args)
    with pytest.raises(SystemExit):
        run_checkpoint(RESNET.replace("do0", "do1"), "none", args)


def nc_file(root: Path, arm: str, scale: float) -> None:
    folder = root / arm / RESNET / "params"
    folder.mkdir(parents=True, exist_ok=True)
    torch.save({"nc_metrics": {m: torch.tensor(scale) for m in NC_METRICS}},
               folder / "NeuralCollapse_params_RW0_RF0_ASHNone.pt")


def test_nc_change_reads_any_arm(tmp_path: Path) -> None:
    nc_file(tmp_path, "none", 1.0)
    nc_file(tmp_path, "unitscale", 1.5)
    assert nc_change(tmp_path, [RESNET], "unitscale") == pytest.approx(0.5)


def s1_rows(gap: float, gap_twin: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = [{"model_id": f"m{m}", "eval_dataset": f"d{d}", "regime": regime,
             "regret": gap + 1.0 + 0.01 * m, "random": 1.0}
            for m in range(6) for d in range(2) for regime in ("near", "mid", "far")]
    none = pd.DataFrame(rows)
    none["eval_dataset"] = none["eval_dataset"] + none["regime"]
    twin = none.assign(regret=none["regret"] - gap + gap_twin)
    return none, twin


def test_gap_table_measures_the_removed_disadvantage() -> None:
    none, twin = s1_rows(2.0, 0.5)
    table = gap_table(none, twin)
    assert table["regime"].tolist() == ["near", "mid", "far"]
    assert table["e"].tolist() == pytest.approx([1.5] * 3)
    assert (table["e_low"] <= table["e"]).all()


def s1_table(gap: list[float], e: list[float], low: list[float],
             high: list[float]) -> pd.DataFrame:
    return pd.DataFrame({"regime": ["near", "mid", "far"], "gap": gap, "e": e,
                         "e_low": low, "e_high": high})


@pytest.mark.parametrize(("table", "expected"), [
    (s1_table([0.1, 0.2, 0.3], [0, 0, 0], [0, 0, 0], [0, 0, 0]), "stop"),
    (s1_table([0.1, 1.0, 2.0], [0, 0.6, 1.2], [0, 0.1, 0.4], [0, 1, 2]), "X"),
    (s1_table([0.1, 1.0, 2.0], [0, 0.6, 0.2], [0, 0.1, 0.1], [0, 1, 0.5]), "Z"),
    (s1_table([0.1, 1.0, 2.0], [5, 0.1, 0.2], [5, -0.2, 0.0], [5, 0.4, 0.9]), "N"),
])
def test_decide_s1(table: pd.DataFrame, expected: str) -> None:
    assert decide_s1(table).startswith(expected)


def test_final_s1_needs_a_failed_control() -> None:
    table = s1_table([0.1, 1.0, 2.0], [0, 0.6, 1.2], [0, 0.1, 0.4], [0, 1, 2])
    primary = decide_s1(table)
    assert final_s1(primary, table, None).startswith("X pending")
    weak = s1_table([0.1, 1.0, 2.0], [0, 0.1, 0.2], [0, -1, -1], [0, 1, 1])
    assert final_s1(primary, table, weak).startswith("X")
    strong = s1_table([0.1, 1.0, 2.0], [0, 0.1, 1.5], [0, -1, 1], [0, 1, 2])
    assert final_s1(primary, table, strong).startswith("Z")
    assert final_s1("N: x", table, strong) == "N: x"


def test_twin_inputs_use_a_ratio_for_var_collapse() -> None:
    stored = pd.DataFrame({m: [1.0] for m in NC_METRICS})
    none = pd.DataFrame({m: [10.0] for m in NC_METRICS})
    twin = none * 1.2
    inputs = twin_inputs(stored, none, twin)
    assert inputs["var_collapse"].item() == pytest.approx(1.2)
    assert inputs["self_duality"].item() == pytest.approx(3.0)
    assert twin_inputs(stored, none, none).equals(stored)


def test_definition_gate() -> None:
    stored = pd.DataFrame({m: [1.0, 2.0] for m in NC_METRICS})
    none = stored.copy()
    none["var_collapse"] = stored["var_collapse"] * 10
    values = definition_check(stored, none, pd.Series([10, 10]))
    check_definition_gate(values)
    none["equiangular_uc"] *= 1.2
    with pytest.raises(RuntimeError):
        check_definition_gate(definition_check(stored, none, pd.Series([10, 10])))


def test_extrapolation_share() -> None:
    z = pd.DataFrame({"a": [0.0, 3.0], "b": [0.0, 0.0]})
    share = extrapolation_share(z, pd.Series({"a": -1.0, "b": -1.0}),
                                pd.Series({"a": 1.0, "b": 1.0}))
    assert share == 0.25


def d1_table(low: list[float], high: list[float]) -> pd.DataFrame:
    return pd.DataFrame({"threshold": [0.93, 0.88, 0.50], "ci_low": low,
                         "ci_high": high})


@pytest.mark.parametrize(("material", "tracking", "extrapolated", "expected"), [
    (d1_table([-0.1] * 3, [0.1] * 3), d1_table([-1] * 3, [-0.5] * 3), 0.0, "D0"),
    (d1_table([0.5, 0, 0], [0.9, 0.1, 0.1]), d1_table([-1] * 3, [-0.5] * 3), 0.3,
     "DZ: the twin"),
    (d1_table([0.5, 0, 0], [0.9, 0.1, 0.1]), d1_table([-1] * 3, [-0.5] * 3), 0.0,
     "DT"),
    (d1_table([0.5, 0, 0], [0.9, 0.1, 0.1]), d1_table([0, 0, -1], [1, 1, 1]), 0.0,
     "DN"),
    (d1_table([0.5, 0, 0], [0.9, 0.1, 0.1]), d1_table([-1, -1, 0], [1, -0.1, 1]),
     0.0, "DZ: inconclusive"),
])
def test_decide_d1(material: pd.DataFrame, tracking: pd.DataFrame,
                   extrapolated: float, expected: str) -> None:
    assert decide_d1(material, tracking, extrapolated).startswith(expected)
