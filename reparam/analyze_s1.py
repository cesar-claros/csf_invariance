"""Stage S1: does the VGG-13 bulk-spectrum shape explain the ViT failure?

Reads the ViT stats CSVs of arm none and a bulk arm, rebuilds the CNN-trained
shortlists of the 40 ViTs (gate G4), and applies the S1 rule of
../documentation/spectrum_unit_scale_protocol_2026-11-04.md. The bulk twins keep all
8 NC metrics, so the shortlists stay the stored ones.

Run from code/ after copying the twin root's analysis/, params/NeuralCollapse_*,
and origin_stats/ to this machine:

    python reparam/analyze_s1.py --twin-root <dir> --gates-only \
        --experiments-file reparam/experiments_vit_s1a.txt
    python reparam/analyze_s1.py --twin-root <dir> \
        --experiments-file reparam/experiments_vit.txt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reparam.analyze_twin import (
    BULK_FREE,
    DEFAULT_OUT_ROOT,
    INVARIANT_TOL,
    LOGIT_ONLY,
    PAPER_CODE,
    PUBLISHED_TOL,
    REGIMES,
    UNGATED,
    check_arm_gate,
    check_nc_gate,
    cluster_interval,
    read_arm,
)

for _path in (PAPER_CODE / "x9_imagenet",
              PAPER_CODE / "nc_csf_predictivity" / "ablations"):
    sys.path.insert(0, str(_path))
import vit_transfer_22
from calibration_features_clique import add_model_id
from phase3_transfer import fitted_heads, per_row
from regimefree_wilcoxon import load_parquet

# The paper repository tracks only the compressed table, and pandas reads it as is.
vit_transfer_22.AGGREGATE = PAPER_CODE / "new_csfs_pilot_summary_long.csv.gz"

HERE = Path(__file__).resolve().parent
EXACT_VIT = ("MLS", "CTM")
ROW_KEYS = ["model_id", "eval_dataset"]
FAILURE_GAP = 0.5
EXPLAINED_SHARE = 0.5
G4_TOL = 1e-9
FULL_POOL = 40


def published_vit() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """The pool table, the published ViT AUGRC rows (gate GV), and the regimes."""
    long_df = add_model_id(load_parquet(DEFAULT_OUT_ROOT / "track1" / "dataset"
                                        / "long_harmonized.parquet"))
    rows = vit_transfer_22.load_vit(long_df)
    regimes = (long_df.loc[long_df["architecture"] == "ViT",
                           ["source", "eval_dataset", "regime"]]
               .drop_duplicates().rename(columns={"source": "family"}))
    return long_df, rows, regimes


def vit_arm(twin_root: Path, arm: str, experiments: list[str],
            published: pd.DataFrame) -> pd.DataFrame:
    """One arm's rows in the layout of the published rows, on its own models.

    Raises:
        RuntimeError: If a published (row, CSF) pair is missing from the arm.
    """
    values = read_arm(twin_root, arm, experiments)
    target = published[published["model_id"].isin(values["model_id"])]
    merged = target.drop(columns="raw_augrc").merge(
        values, on=ROW_KEYS + ["csf"], how="left")
    missing = int(merged["raw_augrc"].isna().sum())
    if missing:
        raise RuntimeError(f"{missing} of {len(merged)} rows of arm {arm} are missing")
    return merged


def max_gap(left: pd.DataFrame, right: pd.DataFrame, csfs: tuple,
            statistic: str = "max") -> pd.Series:
    """Largest (or median) absolute AUGRC gap per CSF between two row tables."""
    both = left.merge(right, on=ROW_KEYS + ["csf"], suffixes=("_left", "_right"))
    both = both[both["csf"].isin(csfs)]
    gaps = (both["raw_augrc_left"] - both["raw_augrc_right"]).abs()
    return gaps.groupby(both["csf"]).agg(statistic).sort_values(ascending=False)


def scored(rows: pd.DataFrame, probs: pd.DataFrame, roster: list[str],
           regimes: pd.DataFrame) -> pd.DataFrame:
    """Per-row regret of the shortlist and of a same-size random set, with regimes."""
    out = per_row(rows, probs, roster)
    return out.merge(regimes, on=["family", "eval_dataset"], how="left")


def check_shortlist_gate(rebuilt: pd.DataFrame, stored_path: Path) -> float:
    """G4: the rebuilt shortlists reproduce the stored per-row regret of the ViTs.

    Raises:
        RuntimeError: On unmatched rows or a deviation above G4_TOL.
    """
    stored = pd.read_csv(stored_path, usecols=ROW_KEYS + ["regret", "random"])
    both = stored.merge(rebuilt[ROW_KEYS + ["regret", "random"]], on=ROW_KEYS,
                        how="outer", suffixes=("_stored", "_rebuilt"), indicator=True)
    worst = float(max((both[f"{c}_stored"] - both[f"{c}_rebuilt"]).abs().max()
                      for c in ("regret", "random")))
    if (both["_merge"] != "both").any() or worst > G4_TOL:
        unmatched = int((both["_merge"] != "both").sum())
        raise RuntimeError(f"G4 failed: unmatched {unmatched}, "
                           f"max deviation {worst:.3g}")
    return worst


def gap_table(none_rows: pd.DataFrame, twin_rows: pd.DataFrame,
              by: list[str] | None = None) -> pd.DataFrame:
    """G (predictor minus random) per arm, E = G - G', and E's interval.

    Means are over rows; the interval resamples whole models.
    """
    by = ["regime"] if by is None else by
    both = none_rows.merge(twin_rows[ROW_KEYS + ["regret", "random"]], on=ROW_KEYS,
                           suffixes=("", "_twin"))
    both["gap"] = both["regret"] - both["random"]
    both["gap_twin"] = both["regret_twin"] - both["random_twin"]
    both["e"] = both["gap"] - both["gap_twin"]
    records = []
    for keys, part in both.groupby(by, sort=False):
        models = part.groupby("model_id")["e"].agg(["sum", "count"])
        low, high = cluster_interval(models["sum"].to_numpy(),
                                     models["count"].to_numpy())
        keys = keys if isinstance(keys, tuple) else (keys,)
        records.append({**dict(zip(by, keys)), "rows": len(part),
                        "models": len(models), "gap": part["gap"].mean(),
                        "gap_twin": part["gap_twin"].mean(), "e": part["e"].mean(),
                        "e_low": low, "e_high": high})
    table = pd.DataFrame(records)
    if by == ["regime"]:
        table = table.set_index("regime").reindex(list(REGIMES)).reset_index()
    return table


def failing(table: pd.DataFrame) -> pd.DataFrame:
    """Regimes where the predictor trails the random set by at least FAILURE_GAP."""
    return table[table["gap"] >= FAILURE_GAP]


def decide_s1(table: pd.DataFrame) -> str:
    """The S1 rule on the output of `gap_table`."""
    fail = failing(table)
    if fail.empty:
        return "stop: no failure regime in arm none"
    explained = (fail["e"] >= EXPLAINED_SHARE * fail["gap"]) & (fail["e_low"] > 0)
    if explained.all():
        return "X: the bulk shape explains the failure"
    if (fail["e_high"] < EXPLAINED_SHARE * fail["gap"]).all():
        return "N: the bulk shape does not explain the failure"
    return "Z: inconclusive"


def final_s1(primary: str, table: pd.DataFrame,
             control: pd.DataFrame | None) -> str:
    """X stands only if the control twin does not also reach the X threshold."""
    if not primary.startswith("X"):
        return primary
    if control is None:
        return "X pending: run the control arm bulk_away"
    regimes = failing(table)["regime"]
    reached = control[control["regime"].isin(regimes)]
    gaps = table.set_index("regime").loc[reached["regime"], "gap"].to_numpy()
    if (reached["e"].to_numpy() >= EXPLAINED_SHARE * gaps).any():
        return "Z: the control twin also removes the disadvantage"
    return primary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--twin-root", required=True)
    parser.add_argument("--experiments-file", required=True)
    parser.add_argument("--arm", default="bulk_vgg")
    parser.add_argument("--control", default="bulk_away")
    parser.add_argument("--out", default=str(HERE / "outputs" / "s1_report.md"))
    parser.add_argument("--gates-only", action="store_true",
                        help="stop after the gates; read no outcome")
    args = parser.parse_args()
    twin_root = Path(args.twin_root)
    experiments = [line.strip() for line in Path(args.experiments_file).read_text()
                   .splitlines() if line.strip() and not line.startswith("#")]

    long_df, published, regimes = published_vit()
    roster = sorted(published["csf"].unique())
    probs = vit_transfer_22.vit_probabilities(fitted_heads(DEFAULT_OUT_ROOT), long_df)
    g4 = check_shortlist_gate(scored(published, probs, roster, regimes),
                              DEFAULT_OUT_ROOT / "vit_transfer_22" / "per_row.csv")

    none_arm = vit_arm(twin_root, "none", experiments, published)
    twin_arm = vit_arm(twin_root, args.arm, experiments, published)
    target = published[published["model_id"].isin(none_arm["model_id"])]
    floor = max_gap(none_arm, target, tuple(roster))
    median = max_gap(none_arm, target, EXACT_VIT, statistic="median")
    check_arm_gate("G1 (amendment A3)", median, PUBLISHED_TOL)
    between = max_gap(none_arm, twin_arm, tuple(c for c in BULK_FREE if c in roster))
    gated = between.drop(list(UNGATED), errors="ignore")
    check_arm_gate("G2", gated[gated.index.isin(LOGIT_ONLY)], INVARIANT_TOL)
    nc_gap = check_nc_gate(twin_root, experiments, args.arm)
    gates = (f"Arm {args.arm}, {none_arm['model_id'].nunique()} models.\n\n"
             f"G1 (amendment A3) passed; median gap of {', '.join(EXACT_VIT)} to "
             f"published is {median.max():.2e} (largest single gap "
             f"{floor[floor.index.isin(EXACT_VIT)].max():.2f}).\n\n"
             f"G2 passed; max gap between arms is "
             f"{gated.max():.2e}; ungated Maha gap is "
             f"{between.get('Maha', float('nan')):.2e}.\n\n"
             f"G3 passed; max relative NC change is {nc_gap:.2e}.\n\n"
             f"G4 passed; the rebuilt ViT shortlists reproduce the stored per-row "
             f"regret (max deviation {g4:.1e}).")
    if args.gates_only:
        print(gates.replace("\n\n", "\n"))
        print("rerun noise floor against published AUGRC:")
        print(floor.round(3).to_string())
        return

    none_rows = scored(none_arm, probs, roster, regimes)
    table = gap_table(none_rows, scored(twin_arm, probs, roster, regimes))
    control = None
    if (twin_root / args.control).exists():
        control_arm = vit_arm(twin_root, args.control, experiments, published)
        control = gap_table(none_rows, scored(control_arm, probs, roster, regimes))
    primary = decide_s1(table)
    outcome = final_s1(primary, table, control)
    if none_arm["model_id"].nunique() < FULL_POOL:
        outcome += (f" (read on {none_arm['model_id'].nunique()} models; the protocol "
                    f"reads S1 on all {FULL_POOL})")
    per_source = gap_table(none_rows, scored(twin_arm, probs, roster, regimes),
                           by=["family", "regime"])
    sections = [
        ("Outcome", (f"**{outcome}**\n\nPrimary: {primary}\n\n```\n"
                    f"{table.round(3).to_string(index=False)}\n```")),
        ("Control arm " + args.control,
         "not run" if control is None
         else f"```\n{control.round(3).to_string(index=False)}\n```"),
        ("Gates", gates),
        ("Per source (exploratory)",
         f"```\n{per_source.round(3).to_string(index=False)}\n```"),
        ("Rerun noise floor against published AUGRC (descriptive)",
         f"```\n{floor.round(3).to_string()}\n```"),
    ]
    text = "# Stage S1: ViT transport test\n" + "".join(
        f"\n## {title}\n\n{body}\n" for title, body in sections)
    Path(args.out).write_text(text)
    print(outcome)
    print(table.round(3).to_string(index=False))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
