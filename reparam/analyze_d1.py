"""Stage D1: does the NC-only predictor follow a change of unit scales?

The unitscale twin keeps the classifier but moves the NC metrics. The predictor gets
the twin's NC inputs: the stored NC values plus the change that the pipeline measures
from arm none to the twin on one shared feature cache (a ratio for var_collapse,
whose stored values use the legacy normalization). Then the stored heads give a new
shortlist. The rule is in ../documentation/spectrum_unit_scale_protocol_2026-11-04.md.

Run from code/ after copying analysis/ and params/NeuralCollapse_* of the arms none
and unitscale:

    python reparam/analyze_d1.py --twin-root <dir> \
        --experiments-file reparam/experiments_rn18_do0.txt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reparam.analyze_twin import (
    ABLATION_ROOT,
    DEFAULT_OUT_ROOT,
    EXACT_REPRO,
    INVARIANT_TOL,
    LOGIT_ONLY,
    NC_METRICS,
    PAPER_CODE,
    PREDICTOR,
    PUBLISHED_TOL,
    REGIMES,
    SPLIT,
    check_arm_gate,
    max_deviation,
    model_id,
    nc_values,
    prepare_pool,
    read_arm,
    regime_table,
    scored_rows,
    with_regimes,
)

for _path in (PAPER_CODE / "x9_imagenet",
              PAPER_CODE / "nc_csf_predictivity" / "ablations"):
    sys.path.insert(0, str(_path))
from phase3_transfer import fitted_heads
from regimefree_wilcoxon import load_parquet

HERE = Path(__file__).resolve().parent
ARM = "unitscale"
D1_FREE = (*LOGIT_ONLY, "Confidence")
RATIO_METRICS = ("var_collapse",)
DEFINITION_TOL = 0.05
HEAD_TOL = 1e-12
EXTRAPOLATION_SHARE = 0.25
CLASSES = {"cifar10": 10, "cifar100": 100, "supercifar100": 20, "tinyimagenet": 200}


def stored_nc(long_df: pd.DataFrame, ids: list[str]) -> pd.DataFrame:
    """The paper's NC inputs (legacy normalization), one row per model."""
    rows = long_df[long_df["model_id"].isin(ids)].drop_duplicates("model_id")
    return rows.set_index("model_id")[list(NC_METRICS)]


def measured_nc(twin_root: Path, arm: str, experiments: list[str]) -> pd.DataFrame:
    """NC metrics that the pipeline saved for one arm, one row per model."""
    return pd.DataFrame({model_id(e): nc_values(twin_root, arm, e) for e in experiments}
                        ).T[list(NC_METRICS)]


def definition_check(stored: pd.DataFrame, none: pd.DataFrame,
                     classes: pd.Series) -> dict[str, float | str]:
    """G5 values: arm none against the stored NC metrics.

    The 7 metrics other than var_collapse use the same code, so they must agree up to
    rerun noise. var_collapse uses 1/N instead of 1/(N C), so its ratio must be C.
    """
    same = [m for m in NC_METRICS if m not in RATIO_METRICS]
    gaps = ((none[same] - stored[same]).abs() / stored[same].abs()).max()
    ratio = (none["var_collapse"] / stored["var_collapse"]) / classes
    return {"max_relative_gap": float(gaps.max()),
            "worst_metric": str(gaps.idxmax()),
            "var_collapse_ratio_over_C_min": float(ratio.min()),
            "var_collapse_ratio_over_C_max": float(ratio.max())}


def check_definition_gate(values: dict[str, float | str]) -> None:
    """G5: arm none and the stored NC metrics share one definition.

    Raises:
        RuntimeError: If a gap or the var_collapse ratio is off by DEFINITION_TOL.
    """
    ratio_off = max(abs(float(values["var_collapse_ratio_over_C_min"]) - 1),
                    abs(float(values["var_collapse_ratio_over_C_max"]) - 1))
    gap = float(values["max_relative_gap"])
    if gap > DEFINITION_TOL or ratio_off > DEFINITION_TOL:
        raise RuntimeError(f"G5 failed: {values}")


def twin_inputs(stored: pd.DataFrame, none: pd.DataFrame,
                twin: pd.DataFrame) -> pd.DataFrame:
    """Stored NC plus the measured change.

    The change is a ratio for var_collapse and a difference for the other metrics.
    """
    out = stored + (twin - none)
    for metric in RATIO_METRICS:
        out[metric] = stored[metric] * twin[metric] / none[metric]
    return out


def resnet_scaling(long_df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """The per-architecture z-scoring of the paper for the ResNet-18 pool."""
    rows = long_df[long_df["architecture"] == "ResNet18"]
    return rows[list(NC_METRICS)].mean(), rows[list(NC_METRICS)].std()


def vgg_range(long_df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Range of the z-scored NC inputs over the VGG-13 training pool."""
    rows = long_df[long_df["architecture"] == "VGG13"][list(NC_METRICS)]
    z = (rows - rows.mean()) / rows.std()
    return z.min(), z.max()


def extrapolation_share(z: pd.DataFrame, low: pd.Series, high: pd.Series) -> float:
    """Share of (model, metric) inputs outside the training range."""
    outside = (z < low) | (z > high)
    return float(outside.to_numpy().mean())


def shortlist(heads: dict, z: pd.DataFrame) -> pd.DataFrame:
    """Regime-free shortlist rows in the layout of the stored predictions."""
    frames = []
    for csf, pipe in heads.items():
        p = pipe.predict_proba(z[list(NC_METRICS)])[:, 1]
        for regime in REGIMES:
            frames.append(pd.DataFrame({
                "model_id": z.index, "regime": regime, "csf": csf,
                "p_competitive": p, "predicted_competitive": p >= 0.5}))
    return pd.concat(frames, ignore_index=True)


def check_head_gate(rebuilt: pd.DataFrame, stored: pd.DataFrame) -> float:
    """G6: the heads on the z-scored stored NC give the stored probabilities.

    Raises:
        RuntimeError: On unmatched rows or a deviation above HEAD_TOL.
    """
    keys = ["model_id", "regime", "csf"]
    both = stored[keys + ["p_competitive"]].merge(
        rebuilt[keys + ["p_competitive"]], on=keys, how="outer",
        suffixes=("_stored", "_rebuilt"), indicator=True)
    worst = float((both["p_competitive_stored"] - both["p_competitive_rebuilt"])
                  .abs().max())
    if (both["_merge"] != "both").any() or worst > HEAD_TOL:
        unmatched = int((both["_merge"] != "both").sum())
        raise RuntimeError(f"G6 failed: unmatched {unmatched}, max |dp| {worst:.3g}")
    return worst


def decide_d1(material: pd.DataFrame, tracking: pd.DataFrame,
              extrapolated: float) -> str:
    """The D1 rule: material change first, then extrapolation, then tracking."""
    half = 0.5 * material["threshold"]
    if ((material["ci_low"] > -half) & (material["ci_high"] < half)).all():
        return "D0: the unit scales do not move the rankings enough to test tracking"
    if extrapolated > EXTRAPOLATION_SHARE:
        return "DZ: the twin NC inputs extrapolate beyond the training pool"
    if int((tracking["ci_high"] < 0).sum()) >= 2:
        return "DT: the NC-only predictor tracks the change"
    if int((tracking["ci_low"] >= 0).sum()) >= 2:
        return "DN: the NC-only predictor does not track the change"
    return "DZ: inconclusive"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--twin-root", required=True)
    parser.add_argument("--experiments-file", required=True)
    parser.add_argument("--out", default=str(HERE / "outputs" / "d1_report.md"))
    parser.add_argument("--gates-only", action="store_true",
                        help="stop after the gates; read no outcome")
    args = parser.parse_args()
    twin_root = Path(args.twin_root)
    experiments = [line.strip() for line in Path(args.experiments_file).read_text()
                   .splitlines() if line.strip() and not line.startswith("#")]
    ids = [model_id(e) for e in experiments]

    long_df, *_, published, _, _ = prepare_pool(DEFAULT_OUT_ROOT)
    none_arm = with_regimes(read_arm(twin_root, "none", experiments), published)
    twin_arm = with_regimes(read_arm(twin_root, ARM, experiments), published)
    target = published[published["model_id"].isin(ids)]
    floor = max_deviation(none_arm, target, tuple(target["csf"].unique()))
    exact = floor[floor.index.isin(EXACT_REPRO)]
    check_arm_gate("G1", exact, PUBLISHED_TOL)
    between = max_deviation(none_arm, twin_arm, (*D1_FREE, "Maha"))
    gated = between.drop("Maha", errors="ignore")
    check_arm_gate("G2", gated, INVARIANT_TOL)

    stored = stored_nc(long_df, ids)
    none_nc = measured_nc(twin_root, "none", experiments).loc[stored.index]
    twin_nc = measured_nc(twin_root, ARM, experiments).loc[stored.index]
    sources = long_df.drop_duplicates("model_id").set_index("model_id")["source"]
    classes = sources.map(CLASSES).loc[stored.index]
    definition = definition_check(stored, none_nc, classes)
    check_definition_gate(definition)

    heads = fitted_heads(DEFAULT_OUT_ROOT)
    mean, std = resnet_scaling(long_df)
    stored_preds = load_parquet(DEFAULT_OUT_ROOT / ABLATION_ROOT / "track1" / SPLIT
                                / PREDICTOR / "preds.parquet")
    stored_preds = stored_preds[stored_preds["model_id"].isin(ids)]
    head_gap = check_head_gate(shortlist(heads, (stored - mean) / std), stored_preds)
    gates = (f"G1 passed ({exact.max():.2e}). G2 passed ({gated.max():.2e}; ungated "
             f"Maha {between.get('Maha', float('nan')):.2e}). G5 passed "
             f"({definition}). G6 passed (max |dp| {head_gap:.1e}). GH passed.")
    if args.gates_only:
        print(gates)
        return

    z_twin = (twin_inputs(stored, none_nc, twin_nc) - mean) / std
    low, high = vgg_range(long_df)
    extrapolated = extrapolation_share(z_twin, low, high)
    twin_preds = shortlist(heads, z_twin)
    base = scored_rows(stored_preds, none_arm)
    orig_on_twin = scored_rows(stored_preds, twin_arm)
    twin_on_twin = scored_rows(twin_preds, twin_arm)
    material = regime_table(base, orig_on_twin)
    tracking = regime_table(orig_on_twin, twin_on_twin)
    outcome = decide_d1(material, tracking, extrapolated)

    kept = stored_preds[stored_preds["regime"] == "mid"]
    old = kept[kept["predicted_competitive"]].groupby("model_id")["csf"].apply(set)
    new_mid = twin_preds[(twin_preds["regime"] == "mid")
                         & twin_preds["predicted_competitive"]]
    new = new_mid.groupby("model_id")["csf"].apply(set)
    overlap = np.mean([len(old.get(i, set()) & new.get(i, set()))
                       / max(len(old.get(i, set()) | new.get(i, set())), 1)
                       for i in stored.index])
    shift = (z_twin - (stored - mean) / std).abs().median()
    sections = [
        ("Outcome", f"**{outcome}**"),
        ("Material change M = regret of the stored shortlist, twin minus original",
         f"```\n{material.round(3).to_string(index=False)}\n```"),
        ("Tracking K = new shortlist minus stored shortlist, both on the twin",
         f"```\n{tracking.round(3).to_string(index=False)}\n```"),
        ("Inputs", (f"Extrapolation share {extrapolated:.3f} (limit "
                   f"{EXTRAPOLATION_SHARE}). Mean Jaccard overlap of the shortlists "
                   f"{overlap:.3f}.\n\nMedian change of the z-scored NC inputs:\n\n"
                   f"```\n{shift.round(3).to_string()}\n```")),
        ("Gates", gates),
    ]
    text = "# Stage D1: unit-scale tracking test\n" + "".join(
        f"\n## {title}\n\n{body}\n" for title, body in sections)
    Path(args.out).write_text(text)
    print(outcome)
    print(material.round(3).to_string(index=False))
    print(tracking.round(3).to_string(index=False))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
