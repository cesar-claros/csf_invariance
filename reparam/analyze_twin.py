"""Gates and decision rule of the re-origined twin test.

Reads the stats CSVs that `run_twin.py` writes, checks gates G1-G3, then scores
the stored NC-only predictor shortlists on both arms and applies the rule fixed
in ../documentation/reparam_invariance_protocol_2026-11-02.md.

Run from code/ after copying the twin root's analysis/, params/NeuralCollapse_*,
and origin_stats/ to this machine:

    python reparam/analyze_twin.py --twin-root <dir> --gates-only \
        --experiments-file reparam/experiments_gate.txt
    python reparam/analyze_twin.py --twin-root <dir> \
        --experiments-file reparam/experiments_rn18_do0.txt
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import orjson
import pandas as pd
import torch
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
PAPER_CODE = HERE.parent / "external" / "ood_systematic"
sys.path.insert(0, str(PAPER_CODE / "nc_csf_predictivity" / "evaluation"))
import newcsf_heads
from newcsf_heads import (
    ABLATION_ROOT,
    ARCH,
    DEFAULT_OUT_ROOT,
    EXPERIMENT,
    KEYS,
    MODE_TO_EVAL,
    SOURCE_NAMES,
    SPLIT,
    oracle_table,
    prepare_pool,
)
from regimefree_wilcoxon import arm_rows, load_parquet

# The paper repository tracks only the compressed table, and pandas reads it as is.
newcsf_heads.AGGREGATE = PAPER_CODE / "new_csfs_pilot_summary_long.csv.gz"

LOGIT_ONLY = ("MSR", "MLS", "Energy", "GEN", "REN", "PCE", "GE", "PE")
ORIGIN_FREE = (*LOGIT_ONLY, "Confidence", "fDBD", "Maha", "NeCo")
#: Amendment A1: G1 checks only the CSFs with no refitted step (no temperature,
#: no training features), which a rerun reproduces exactly.
EXACT_REPRO = ("MLS", "CTM", "Confidence")
#: Amendment A1: Maha is invariant in exact arithmetic but not in the pipeline's
#: float32 pseudo-inverse, so G2 reports its gap and does not gate it.
UNGATED = ("Maha",)
NC_METRICS = ("var_collapse", "equiangular_uc", "equiangular_wc", "equinorm_uc",
              "equinorm_wc", "max_equiangular_uc", "max_equiangular_wc",
              "self_duality")
REGIMES = ("near", "mid", "far")
#: Headline regret of the NC-only predictor on all 56 ResNet-18 models.
THRESHOLD = {"near": 0.93, "mid": 0.88, "far": 0.50}
PUBLISHED_TOL = 1e-3
INVARIANT_TOL = 0.05
NC_TOL = 1e-3
N_BOOT = 2000
SHORTLIST_SIZE = 8
PREDICTOR = "none_nr_marginal"
STATS_PREFIX = "stats_RW0_RF0_ASHNone_"


def model_id(experiment: str) -> str:
    """Map a ResNet-18 or VGG-13 experiment path to the pool's model id.

    Raises:
        ValueError: If the path does not match the sweep naming.
    """
    match = re.match(EXPERIMENT, experiment)
    if match is None:
        raise ValueError(f"unexpected experiment name: {experiment}")
    source = SOURCE_NAMES.get(match["source"], match["source"])
    return (f"{ARCH[match['arch']]}|{match['paradigm']}|{source}|{match['run']}"
            f"|{match['do']}|{float(match['rew']):g}")


def read_arm(twin_root: Path, arm: str, experiments: list[str]) -> pd.DataFrame:
    """Long AUGRC table of one arm: model_id, eval_dataset, csf, raw_augrc."""
    frames = []
    for experiment in experiments:
        analysis = twin_root / arm / experiment / "analysis"
        for path in sorted(analysis.glob(f"{STATS_PREFIX}*.csv")):
            mode = path.stem.removeprefix(STATS_PREFIX)
            if mode not in MODE_TO_EVAL:
                continue
            table = pd.read_csv(path, index_col=0)["AUGRC"].rename("raw_augrc")
            frames.append(table.rename_axis("csf").reset_index().assign(
                model_id=model_id(experiment), eval_dataset=MODE_TO_EVAL[mode]))
    if not frames:
        raise SystemExit(f"no stats files under {twin_root / arm}")
    long = pd.concat(frames, ignore_index=True)
    long["csf"] = long["csf"].str.replace("_RecError_global", " RecError global")
    return long


def with_regimes(arm: pd.DataFrame, published: pd.DataFrame) -> pd.DataFrame:
    """Restrict an arm to the published roster and rows, and add the regime.

    Raises:
        RuntimeError: If a published (row, CSF) pair is missing from the arm.
    """
    target = published[published["model_id"].isin(arm["model_id"])]
    merged = target[KEYS + ["csf"]].merge(arm, on=["model_id", "eval_dataset", "csf"],
                                          how="left")
    missing = int(merged["raw_augrc"].isna().sum())
    if missing:
        raise RuntimeError(f"{missing} of {len(merged)} arm rows are missing")
    return merged


def max_deviation(left: pd.DataFrame, right: pd.DataFrame, csfs: tuple) -> pd.Series:
    """Largest absolute AUGRC gap per CSF between two arms, over the given CSFs."""
    both = left.merge(right, on=KEYS + ["csf"], suffixes=("_left", "_right"))
    both = both[both["csf"].isin(csfs)]
    gaps = (both["raw_augrc_left"] - both["raw_augrc_right"]).abs()
    return gaps.groupby(both["csf"]).max().sort_values(ascending=False)


def check_arm_gate(name: str, deviations: pd.Series, tolerance: float) -> None:
    """Stop before any outcome if a gate fails.

    Raises:
        RuntimeError: If any deviation is above the tolerance.
    """
    failed = deviations[deviations > tolerance]
    if len(failed):
        raise RuntimeError(f"{name} failed (tolerance {tolerance}):\n"
                           f"{failed.to_string()}")


def check_nc_gate(twin_root: Path, experiments: list[str]) -> float:
    """G3: largest relative change of the 8 NC metrics between arms.

    Raises:
        RuntimeError: If the change is above NC_TOL.
    """
    worst = 0.0
    name = "NeuralCollapse_params_RW0_RF0_ASHNone.pt"
    for experiment in experiments:
        none, center = (torch.load(twin_root / arm / experiment / "params" / name,
                                   map_location="cpu")["nc_metrics"]
                        for arm in ("none", "center"))
        for metric in NC_METRICS:
            gap = abs(float(center[metric]) - float(none[metric]))
            worst = max(worst, gap / (abs(float(none[metric])) + 1e-12))
    if worst > NC_TOL:
        raise RuntimeError(f"G3 failed: relative NC change {worst:.3g}")
    return worst


def scored_rows(preds: pd.DataFrame, arm: pd.DataFrame) -> pd.DataFrame:
    """Joint-side regret per row of the predictor's shortlist, on one arm."""
    rows = arm_rows(preds, arm, oracle_table(arm))
    return rows.loc[rows["side"] == "all", KEYS + ["regret"]]


def cluster_interval(sums: np.ndarray, counts: np.ndarray,
                     seed: int = 0) -> tuple[float, float]:
    """95% percentile interval of a row mean, resampling whole models.

    Args:
        sums: Sum of the row values per model.
        counts: Number of rows per model.
        seed: Random seed.
    """
    draws = np.random.default_rng(seed).integers(0, len(sums),
                                                 size=(N_BOOT, len(sums)))
    means = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def regime_table(none_rows: pd.DataFrame, center_rows: pd.DataFrame) -> pd.DataFrame:
    """r, r', Delta, and the model-clustered interval per regime.

    Means are over rows, as in the paper, so a model with more datasets in a regime
    keeps its weight.
    """
    both = none_rows.merge(center_rows, on=KEYS, suffixes=("_none", "_center"))
    both["delta"] = both["regret_center"] - both["regret_none"]
    records = []
    for regime in REGIMES:
        part = both[both["regime"] == regime]
        models = part.groupby("model_id")["delta"].agg(["sum", "count"])
        low, high = cluster_interval(models["sum"].to_numpy(),
                                     models["count"].to_numpy())
        records.append({
            "regime": regime, "models": len(models),
            "threshold": THRESHOLD[regime],
            "r": float(part["regret_none"].mean()),
            "r_twin": float(part["regret_center"].mean()),
            "delta": float(part["delta"].mean()), "ci_low": low, "ci_high": high,
        })
    return pd.DataFrame(records)


def decide(table: pd.DataFrame) -> str:
    """Apply the protocol's rule to the output of `regime_table`."""
    matters = (table["delta"] >= table["threshold"]) & (table["ci_low"] > 0)
    if matters.sum() >= 2:
        return "A: origin matters"
    half = 0.5 * table["threshold"]
    if ((table["ci_low"] > -half) & (table["ci_high"] < half)).all():
        return "B: origin immaterial"
    return "C: inconclusive"


def exact_twin(none_arm: pd.DataFrame, center_arm: pd.DataFrame) -> pd.DataFrame:
    """Arm `center` with the origin-free CSFs taken from arm `none` (amendment A1).

    By the invariance table these CSFs do not change under translation in exact
    arithmetic on the CNN path, so this removes float32 noise from them.
    """
    moved = center_arm[~center_arm["csf"].isin(ORIGIN_FREE)]
    fixed = none_arm[none_arm["csf"].isin(ORIGIN_FREE)]
    return pd.concat([moved, fixed], ignore_index=True)


def final_outcome(primary: str, sensitivity: str) -> str:
    """Amendment A1: the outcome stands only if the sensitivity analysis agrees."""
    return primary if primary == sensitivity else "C: inconclusive"


def rank_summary(none_arm: pd.DataFrame, center_arm: pd.DataFrame) -> pd.DataFrame:
    """Spearman between arms, top-1 change, and oracle change per regime."""
    both = none_arm.merge(center_arm, on=KEYS + ["csf"], suffixes=("_none", "_center"))
    records = []
    for keys, part in both.groupby(KEYS):
        records.append({
            "regime": keys[2],
            "spearman": spearmanr(part["raw_augrc_none"],
                                  part["raw_augrc_center"]).statistic,
            "top1_changed": (part.loc[part["raw_augrc_none"].idxmin(), "csf"]
                             != part.loc[part["raw_augrc_center"].idxmin(), "csf"]),
            "oracle_change": (part["raw_augrc_center"].min()
                              - part["raw_augrc_none"].min()),
        })
    return (pd.DataFrame(records).groupby("regime")
            .mean(numeric_only=False).reindex(list(REGIMES)))


def fixed_set_regret(none_arm: pd.DataFrame, center_arm: pd.DataFrame) -> pd.Series:
    """Twin regret of the 8 best CSFs per (model, regime) known from arm `none`."""
    keys = ["model_id", "regime"]
    means = none_arm.groupby(keys + ["csf"])["raw_augrc"].mean().reset_index()
    means["position"] = means.groupby(keys)["raw_augrc"].rank(method="first")
    chosen = means.loc[means["position"] <= SHORTLIST_SIZE, keys + ["csf"]]
    picked = center_arm.merge(chosen, on=keys + ["csf"])
    best = picked.groupby(KEYS)["raw_augrc"].min()
    oracle = center_arm.groupby(KEYS)["raw_augrc"].min()
    return (best - oracle).groupby("regime").mean().reindex(list(REGIMES))


def family_stats(twin_root: Path) -> pd.DataFrame:
    """Median origin statistics per architecture family."""
    records = []
    folder = twin_root / "origin_stats"
    for path in sorted(folder.rglob("*.json")):
        name = path.relative_to(folder).as_posix()
        family = ("ViT" if name.startswith("vit/") else
                  "ResNet18" if "resnet18" in name else "VGG13")
        records.append({"family": family, **orjson.loads(path.read_bytes())})
    if not records:
        return pd.DataFrame()
    frame = pd.DataFrame(records)
    counts = frame.groupby("family").size().rename("models")
    return pd.concat([counts, frame.groupby("family").median()], axis=1)


def report(sections: list[tuple[str, str]], path: Path) -> None:
    """Write the sections as Markdown."""
    text = "# Re-origined twin test\n"
    for title, body in sections:
        text += f"\n## {title}\n\n{body}\n"
    path.write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--twin-root", required=True)
    parser.add_argument("--experiments-file", required=True)
    parser.add_argument("--out", default=str(HERE / "outputs" / "twin_report.md"))
    parser.add_argument("--gates-only", action="store_true",
                        help="stop after the gates; read no outcome")
    args = parser.parse_args()
    twin_root = Path(args.twin_root)
    experiments = [line.strip() for line in Path(args.experiments_file).read_text()
                   .splitlines() if line.strip() and not line.startswith("#")]

    *_, published, _, _ = prepare_pool(DEFAULT_OUT_ROOT)
    none_arm = with_regimes(read_arm(twin_root, "none", experiments), published)
    center_arm = with_regimes(read_arm(twin_root, "center", experiments), published)
    target = published[published["model_id"].isin(none_arm["model_id"])]

    floor = max_deviation(none_arm, target, tuple(target["csf"].unique()))
    exact = floor[floor.index.isin(EXACT_REPRO)]
    check_arm_gate("G1", exact, PUBLISHED_TOL)
    invariant = max_deviation(none_arm, center_arm, ORIGIN_FREE)
    gated = invariant.drop(list(UNGATED), errors="ignore")
    check_arm_gate("G2", gated, INVARIANT_TOL)
    nc_change = check_nc_gate(twin_root, experiments)
    gates = (f"G1 passed; max gap of {', '.join(EXACT_REPRO)} to published is "
             f"{exact.max():.2e}.\n\nG2 passed; max gap between arms is "
             f"{gated.max():.2e}; ungated Maha gap is "
             f"{invariant.get('Maha', float('nan')):.2e}.\n\n"
             f"G3 passed; max relative NC change is {nc_change:.2e}.")
    if args.gates_only:
        print(gates.replace("\n\n", "\n"))
        print("rerun noise floor against published AUGRC:")
        print(floor.round(3).to_string())
        return

    preds = load_parquet(DEFAULT_OUT_ROOT / ABLATION_ROOT / "track1" / SPLIT / PREDICTOR
                         / "preds.parquet")
    preds = preds[preds["model_id"].isin(none_arm["model_id"])]
    baseline = scored_rows(preds, none_arm)
    table = regime_table(baseline, scored_rows(preds, center_arm))
    sensitivity = regime_table(baseline,
                               scored_rows(preds, exact_twin(none_arm, center_arm)))
    outcome = final_outcome(decide(table), decide(sensitivity))

    by_csf = (center_arm.merge(none_arm, on=KEYS + ["csf"], suffixes=("_c", "_n"))
              .assign(change=lambda d: d["raw_augrc_c"] - d["raw_augrc_n"])
              .pivot_table(index="csf", columns="regime", values="change")
              [list(REGIMES)].round(2).sort_values("mid"))
    report([
        ("Outcome", (f"**{outcome}**\n\nPrimary: {decide(table)}\n\n```\n"
                    f"{table.round(3).to_string(index=False)}\n```\n\n"
                    f"Sensitivity, origin-free CSFs from arm none: "
                    f"{decide(sensitivity)}\n\n```\n"
                    f"{sensitivity.round(3).to_string(index=False)}\n```")),
        ("Gates (amendment A1)", gates),
        ("Rerun noise floor against published AUGRC (descriptive)",
         f"```\n{floor.round(3).to_string()}\n```"),
        ("Rank agreement between arms (descriptive)",
         f"```\n{rank_summary(none_arm, center_arm).round(3).to_string()}\n```"),
        ("Mean AUGRC change, center minus none (descriptive)",
         f"```\n{by_csf.to_string()}\n```"),
        ((f"Twin regret of the best {SHORTLIST_SIZE} CSFs known from arm none "
         "(descriptive)"),
         f"```\n{fixed_set_regret(none_arm, center_arm).round(3).to_string()}\n```"),
        ("Origin statistics per family (descriptive)",
         f"```\n{family_stats(twin_root).round(3).to_string()}\n```"),
    ], Path(args.out))
    print(outcome)
    print(table.round(3).to_string(index=False))
    print(sensitivity.round(3).to_string(index=False))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
