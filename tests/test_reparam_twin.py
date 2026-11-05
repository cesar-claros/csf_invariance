"""Runner and analysis of the re-origined twin test, without fd_shifts."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from reparam.analyze_twin import (
    EXACT_REPRO,
    NC_METRICS,
    ORIGIN_FREE,
    UNGATED,
    check_arm_gate,
    check_nc_gate,
    cluster_interval,
    decide,
    exact_twin,
    final_outcome,
    model_id,
    read_arm,
    regime_table,
    with_regimes,
)
from reparam.run_twin import (
    Twin,
    apply_shift,
    compensate_bias,
    ood_modes,
    origin_stats,
    planned_steps,
)

EXPERIMENT = "cifar10_paper_sweep/devries_bbresnet18_do0_run1_rew2.2"
MODEL_ID = "ResNet18|devries|cifar10|1|0|2.2"


def fake_model() -> SimpleNamespace:
    return SimpleNamespace(module=object(), study_name="devries")


def fake_compute(calls: list[str]):
    def compute(model, datamodule, set_name: str) -> dict:
        calls.append(set_name)
        torch.manual_seed(len(set_name))
        encoded = torch.relu(torch.randn(90, 5)) + 1.0
        labels = torch.arange(90) % 3
        return {"encoded": encoded, "labels": labels, "logits": encoded[:, :3]}
    return compute


def fake_layer(module, study_name: str, return_model: bool = True) -> tuple:
    return torch.ones(3, 5), torch.zeros(3)


def test_shift_with_compensated_bias_keeps_the_logits() -> None:
    weights, bias = torch.randn(3, 5), torch.randn(3)
    encoded, shift = torch.randn(7, 5), torch.randn(5)
    moved = apply_shift({"encoded": encoded}, shift)["encoded"]
    new_bias = compensate_bias(weights, bias, shift)
    gap = (moved @ weights.T + new_bias) - (encoded @ weights.T + bias)
    assert gap.abs().max() < 1e-5


def test_origin_stats_on_rectified_features() -> None:
    encoded = torch.relu(torch.randn(600, 8)) + torch.tensor(2.0)
    encoded[:, 0] = 0.0
    labels = torch.arange(600) % 3
    stats = origin_stats(encoded, labels, torch.randn(3, 8))
    assert stats["zero_fraction"] == pytest.approx(1 / 8)
    assert stats["negative_fraction"] == 0.0
    assert stats["origin_ratio"] > 1.0
    assert 0.0 < stats["mean_share_outside_head"] <= 1.0


def test_arms_share_the_cache_and_differ_by_the_shift(tmp_path: Path) -> None:
    calls: list[str] = []
    compute = fake_compute(calls)
    center = Twin(tmp_path, EXPERIMENT, "center")
    centered = center.evaluate(compute, fake_layer, fake_model(), None, "val")
    assert calls == ["train", "val"]
    none = Twin(tmp_path, EXPERIMENT, "none")
    raw = none.evaluate(compute, fake_layer, fake_model(), None, "val")
    assert calls == ["train", "val"]
    train = center.evaluate(compute, fake_layer, fake_model(), None, "train")
    assert train["encoded"].mean(0).abs().max() < 1e-5
    assert torch.equal(raw["logits"], centered["logits"])
    assert (raw["encoded"] - centered["encoded"]).std(0).max() < 1e-5


def test_head_is_compensated_only_after_the_shift_exists(tmp_path: Path) -> None:
    twin = Twin(tmp_path, EXPERIMENT, "center")
    layer = (object(), torch.ones(2, 3), torch.zeros(2))
    assert twin.last_layer(layer) is layer
    twin.shift = torch.tensor([1.0, 2.0, 3.0])
    _, _, bias = twin.last_layer(layer)
    assert bias.tolist() == [-6.0, -6.0]
    bare = twin.last_layer((torch.ones(2, 3), torch.zeros(2)))
    assert bare[1].tolist() == [-6.0, -6.0]


def test_stats_only_stops_without_writing_features(tmp_path: Path) -> None:
    twin = Twin(tmp_path, EXPERIMENT, "none", stats_only=True)
    with pytest.raises(SystemExit):
        twin.evaluate(fake_compute([]), fake_layer, fake_model(), None, "val")
    assert twin.stats_path.exists()
    assert not list(twin.feature_dir.glob("*.pt"))


def test_stats_only_stops_even_when_features_exist(tmp_path: Path) -> None:
    Twin(tmp_path, EXPERIMENT, "none").evaluate(
        fake_compute([]), fake_layer, fake_model(), None, "val")
    twin = Twin(tmp_path, EXPERIMENT, "none", stats_only=True)
    with pytest.raises(SystemExit):
        twin.evaluate(fake_compute([]), fake_layer, fake_model(), None, "val")


def test_modes_cover_each_source_without_iid() -> None:
    modes = ood_modes(EXPERIMENT)
    assert "ood_sncs_c100" in modes and "iid_test" not in modes and len(modes) == 8
    tiny = ood_modes("tiny-imagenet-200_paper_sweep/dg_bbresnet18_do0_run1_rew2.2")
    assert "ood_nsncs_ti" not in tiny and len(tiny) == 8


def test_planned_steps_fit_before_eval_and_stats_only_is_one_step() -> None:
    steps = planned_steps(EXPERIMENT, stats_only=False)
    assert steps[:2] == [("fit", "plain", ""), ("fit", "global", "")]
    assert len(steps) == 2 + 2 * 8
    assert planned_steps("vit/x_do0", stats_only=True) == [("fit", "plain", "")]


def test_model_id_matches_the_pool_format() -> None:
    assert model_id(EXPERIMENT) == MODEL_ID
    other = "tiny-imagenet-200_paper_sweep/dg_bbresnet18_do0_run2_rew10.0"
    assert model_id(other) == "ResNet18|dg|tinyimagenet|2|0|10"
    with pytest.raises(ValueError):
        model_id("vit/cifar10_modelvit_bbvit_lr0.01_bs128_run0_do0_rew0")


def write_stats(root: Path, arm: str, mode: str, values: dict[str, float]) -> None:
    folder = root / arm / EXPERIMENT / "analysis"
    folder.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame({"AUGRC": values})
    frame.to_csv(folder / f"stats_RW0_RF0_ASHNone_{mode}.csv")


def test_read_arm_renames_and_drops_unknown_modes(tmp_path: Path) -> None:
    values = {"CTM": 1.0, "KPCA_RecError_global": 2.0}
    write_stats(tmp_path, "none", "ood_nsncs_svhn", values)
    write_stats(tmp_path, "none", "iid_test", values)
    long = read_arm(tmp_path, "none", [EXPERIMENT])
    assert set(long["eval_dataset"]) == {"svhn"}
    assert set(long["csf"]) == {"CTM", "KPCA RecError global"}
    assert set(long["model_id"]) == {MODEL_ID}


def test_with_regimes_rejects_missing_rows() -> None:
    published = pd.DataFrame({"model_id": ["m", "m"], "eval_dataset": ["a", "a"],
                              "regime": ["near", "near"], "csf": ["x", "y"],
                              "raw_augrc": [1.0, 2.0]})
    arm = pd.DataFrame({"model_id": ["m"], "eval_dataset": ["a"], "csf": ["x"],
                        "raw_augrc": [1.5]})
    with pytest.raises(RuntimeError):
        with_regimes(arm, published)


def test_gate_stops_on_a_deviation() -> None:
    check_arm_gate("G", pd.Series({"MSR": 0.01}), 0.05)
    with pytest.raises(RuntimeError):
        check_arm_gate("G", pd.Series({"MSR": 0.5}), 0.05)


def write_nc(root: Path, arm: str, scale: float) -> None:
    folder = root / arm / EXPERIMENT / "params"
    folder.mkdir(parents=True, exist_ok=True)
    metrics = {name: torch.tensor(scale) for name in NC_METRICS}
    torch.save({"nc_metrics": metrics},
               folder / "NeuralCollapse_params_RW0_RF0_ASHNone.pt")


def test_nc_gate(tmp_path: Path) -> None:
    write_nc(tmp_path, "none", 1.0)
    write_nc(tmp_path, "center", 1.0)
    assert check_nc_gate(tmp_path, [EXPERIMENT]) == 0.0
    write_nc(tmp_path, "center", 1.1)
    with pytest.raises(RuntimeError):
        check_nc_gate(tmp_path, [EXPERIMENT])


def rule_table(delta: list[float], low: list[float], high: list[float],
               threshold: float = 1.0) -> pd.DataFrame:
    return pd.DataFrame({"threshold": [threshold] * 3, "delta": delta, "ci_low": low,
                         "ci_high": high})


@pytest.mark.parametrize(("delta", "low", "high", "expected"), [
    ([1.2, 1.1, 0.1], [0.3, 0.2, -0.1], [2, 2, 0.3], "A: origin matters"),
    ([1.2, 0.1, 0.1], [0.3, -0.1, -0.1], [2, 0.3, 0.3], "C: inconclusive"),
    ([1.2, 1.1, 0.1], [-0.1, -0.1, -0.1], [2, 2, 0.3], "C: inconclusive"),
    ([0.1, 0.0, -0.1], [-0.2, -0.3, -0.4], [0.4, 0.3, 0.2], "B: origin immaterial"),
    ([0.1, 0.0, 0.3], [-0.2, -0.3, 0.1], [0.4, 0.3, 0.6], "C: inconclusive"),
    ([-2.0, -2.0, -2.0], [-3, -3, -3], [-1, -1, -1], "C: inconclusive"),
])
def test_decision_rule(delta: list[float], low: list[float], high: list[float],
                       expected: str) -> None:
    assert decide(rule_table(delta, low, high)) == expected


def test_regime_table_clusters_by_model() -> None:
    rows = []
    for model, shift in (("a", 1.0), ("b", 3.0)):
        for regime in ("near", "mid", "far"):
            rows += [{"model_id": model, "eval_dataset": d, "regime": regime,
                      "regret": shift} for d in ("x", "y", "z")]
    center = pd.DataFrame(rows)
    none = center.assign(regret=0.0)
    table = regime_table(none, center)
    assert table["delta"].tolist() == [2.0, 2.0, 2.0]
    assert table["models"].tolist() == [2, 2, 2]
    assert table["threshold"].tolist() == [0.93, 0.88, 0.50]


def test_cluster_interval_is_deterministic_and_ordered() -> None:
    sums, counts = np.array([0.0, 2.0, 4.0, 6.0]), np.array([1, 1, 2, 2])
    assert cluster_interval(sums, counts) == cluster_interval(sums, counts)
    low, high = cluster_interval(sums, counts)
    assert low < sums.sum() / counts.sum() < high


def test_regime_table_means_are_over_rows() -> None:
    rows = []
    for model, datasets, shift in (("a", ("x",), 3.0), ("b", ("x", "y", "z"), 1.0)):
        for regime in ("near", "mid", "far"):
            rows += [{"model_id": model, "eval_dataset": d, "regime": regime,
                      "regret": shift} for d in datasets]
    center = pd.DataFrame(rows)
    table = regime_table(center.assign(regret=0.0), center)
    assert table["delta"].tolist() == [1.5, 1.5, 1.5]


def test_amended_gate_sets() -> None:
    assert set(EXACT_REPRO) == {"MLS", "CTM", "Confidence"}
    assert set(UNGATED) <= set(ORIGIN_FREE)
    assert "CTM" not in ORIGIN_FREE


def test_exact_twin_takes_origin_free_csfs_from_arm_none() -> None:
    keys = {"model_id": "m", "eval_dataset": "x", "regime": "near"}
    none = pd.DataFrame([{**keys, "csf": c, "raw_augrc": 1.0} for c in ("MSR", "CTM")])
    center = none.assign(raw_augrc=[1.1, 5.0])
    merged = exact_twin(none, center).set_index("csf")["raw_augrc"]
    assert merged.to_dict() == {"CTM": 5.0, "MSR": 1.0}


@pytest.mark.parametrize(("primary", "sensitivity", "expected"), [
    ("A: origin matters", "A: origin matters", "A: origin matters"),
    ("B: origin immaterial", "B: origin immaterial", "B: origin immaterial"),
    ("A: origin matters", "C: inconclusive", "C: inconclusive"),
    ("B: origin immaterial", "A: origin matters", "C: inconclusive"),
])
def test_final_outcome_needs_agreement(primary: str, sensitivity: str,
                                       expected: str) -> None:
    assert final_outcome(primary, sensitivity) == expected
