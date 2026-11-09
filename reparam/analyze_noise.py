"""Exploratory: how much of the ViT failure can rerun variability explain?

Compares two extractions of the same checkpoints: the paper's AUGRC values (P) and
the rerun of arm none (R). For each model, the oracle shortlist of one extraction,
with the predictor's shortlist size, is scored on the other. The headroom
H = (q - o) / q compares that cross regret o with the regret q of a random shortlist.
Measures and reading: ../documentation/vit_rerun_noise_analysis_2026-11-09.md

Run from code/:

    python reparam/analyze_noise.py --vit-root twin_results/s1b \
        --resnet-root twin_results/main
"""
from __future__ import annotations

import argparse
import sys
from itertools import combinations, islice
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reparam.analyze_s1 import (
    fitted_heads,
    load_parquet,
    published_vit,
    scored,
    vit_arm,
    vit_transfer_22,
)
from reparam.analyze_twin import (
    ABLATION_ROOT,
    DEFAULT_OUT_ROOT,
    KEYS,
    N_BOOT,
    PREDICTOR,
    REGIMES,
    SPLIT,
    prepare_pool,
    read_arm,
    scored_rows,
    with_regimes,
)

HERE = Path(__file__).resolve().parent
CHUNK = 50_000
CHECK_TOL = 1e-9
LOW, HIGH = 0.25, 0.75


def regret_matrix(wide: pd.DataFrame) -> np.ndarray:
    """Per-row regret: each CSF's AUGRC minus the row's best AUGRC."""
    values = wide.to_numpy(dtype=float)
    return values - values.min(axis=1, keepdims=True)


def best_subset(regret: np.ndarray, size: int) -> tuple[int, ...]:
    """The subset of `size` columns with the lowest mean set regret, by exact search.

    The set regret of a row is the smallest regret among the subset's columns.
    """
    best: tuple[int, ...] = ()
    best_value = np.inf
    candidates = combinations(range(regret.shape[1]), size)
    while chunk := list(islice(candidates, CHUNK)):
        values = regret[:, np.array(chunk)].min(axis=2).mean(axis=0)
        position = int(values.argmin())
        if values[position] < best_value:
            best, best_value = chunk[position], float(values[position])
    return best


def set_regret(regret: np.ndarray, members: list[int]) -> np.ndarray:
    """Set regret per row; an empty set gets the row's worst regret."""
    if not members:
        return regret.max(axis=1)
    return regret[:, members].min(axis=1)


def random_regret(regret: np.ndarray, size: int) -> np.ndarray:
    """Expected set regret per row of a uniformly random subset of `size` columns."""
    n = regret.shape[1]
    if size == 0:
        return regret.max(axis=1)
    ordered = np.sort(regret, axis=1)
    counts = np.array([comb(n - i, size - 1) for i in range(1, n + 1)])
    weights = counts / comb(n, size)
    return ordered @ weights


def model_rows(p_wide: pd.DataFrame, r_wide: pd.DataFrame,
               members: list[str]) -> pd.DataFrame:
    """Cross regret, random regret, predictor regret, and rank agreement per row."""
    columns = list(p_wide.columns)
    regret_p, regret_r = regret_matrix(p_wide), regret_matrix(r_wide[columns])
    picked = [columns.index(c) for c in members]
    size = len(picked)
    oracle_p = list(best_subset(regret_p, size)) if size else []
    oracle_r = list(best_subset(regret_r, size)) if size else []
    out = pd.DataFrame({
        "o_pr": set_regret(regret_r, oracle_p), "o_rp": set_regret(regret_p, oracle_r),
        "q_p": random_regret(regret_p, size), "q_r": random_regret(regret_r, size),
        "r_p": set_regret(regret_p, picked), "r_r": set_regret(regret_r, picked),
        "spearman": [spearmanr(a, b).statistic for a, b in
                     zip(p_wide.to_numpy(), r_wide[columns].to_numpy())],
        "same_best": regret_p.argmin(axis=1) == regret_r.argmin(axis=1),
        "size": size,
    }, index=p_wide.index)
    out["o"] = (out["o_pr"] + out["o_rp"]) / 2
    out["q"] = (out["q_p"] + out["q_r"]) / 2
    out["r"] = (out["r_p"] + out["r_r"]) / 2
    return out


def noise_rows(paper: pd.DataFrame, rerun: pd.DataFrame,
               shortlists: dict[str, list[str]]) -> pd.DataFrame:
    """`model_rows` for every model.

    The inputs are long tables with the KEYS columns, csf, and raw_augrc.
    """
    frames = []
    for model, members in shortlists.items():
        p_wide = paper[paper["model_id"] == model].pivot_table(
            index=KEYS, columns="csf", values="raw_augrc")
        r_wide = rerun[rerun["model_id"] == model].pivot_table(
            index=KEYS, columns="csf", values="raw_augrc").loc[p_wide.index]
        frames.append(model_rows(p_wide, r_wide, members))
    return pd.concat(frames).reset_index()


def ratio_interval(numerator: np.ndarray, denominator: np.ndarray,
                   seed: int = 0) -> tuple[float, float]:
    """95% percentile interval of 1 - sum(numerator) / sum(denominator) over models."""
    draws = np.random.default_rng(seed).integers(
        0, len(numerator), size=(N_BOOT, len(numerator)))
    values = 1 - numerator[draws].sum(axis=1) / denominator[draws].sum(axis=1)
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def reading(low: float, high: float) -> str:
    """The reading fixed in the analysis note."""
    if high < LOW:
        return "noise explains most of the failure"
    if low > HIGH:
        return "noise explains little of the failure"
    return "partial"


def headroom_table(rows: pd.DataFrame) -> pd.DataFrame:
    """Per regime: mean regrets, headroom H with its interval, and rank agreement."""
    records = []
    for regime in REGIMES:
        part = rows[rows["regime"] == regime]
        sums = part.groupby("model_id")[["o", "q"]].sum()
        low, high = ratio_interval(sums["o"].to_numpy(), sums["q"].to_numpy())
        records.append({
            "regime": regime, "rows": len(part), "models": len(sums),
            "predictor_r": part["r"].mean(), "random_q": part["q"].mean(),
            "cross_oracle_o": part["o"].mean(),
            "H": 1 - part["o"].sum() / part["q"].sum(), "H_low": low, "H_high": high,
            "reading": reading(low, high),
            "spearman": part["spearman"].mean(), "same_best": part["same_best"].mean(),
        })
    return pd.DataFrame(records)


def check(name: str, mine: pd.Series, reference: pd.Series) -> float:
    """Stop if this script's regrets differ from the pipeline's.

    Raises:
        RuntimeError: If the largest deviation is above CHECK_TOL.
    """
    worst = float((mine - reference).abs().max())
    if not np.isfinite(worst) or worst > CHECK_TOL:
        raise RuntimeError(f"{name} failed: max deviation {worst:.3g}")
    return worst


def vit_inputs(vit_root: Path, experiments: list[str]
               ) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, list[str]],
                          list[pd.DataFrame]]:
    """Paper and rerun rows of the ViTs with regimes, shortlists, and checks."""
    long_df, published, regimes = published_vit()
    roster = sorted(published["csf"].unique())
    probs = vit_transfer_22.vit_probabilities(fitted_heads(DEFAULT_OUT_ROOT), long_df)
    rerun = vit_arm(vit_root, "none", experiments, published)
    paper = published.merge(regimes, on=["family", "eval_dataset"])
    rerun = rerun.merge(regimes, on=["family", "eval_dataset"])
    members = probs[(probs["p"] >= 0.5) & probs["csf"].isin(roster)]
    shortlists = {m: sorted(g["csf"]) for m, g in members.groupby("model_id")}
    shortlists = {m: shortlists.get(m, []) for m in rerun["model_id"].unique()}
    reference = {name: scored(rows, probs, roster, regimes)
                 .set_index(["model_id", "eval_dataset"])
                 for name, rows in (("paper", published), ("rerun", rerun))}
    return paper, rerun, shortlists, [reference["paper"], reference["rerun"]]


def resnet_inputs(resnet_root: Path, experiments: list[str]
                  ) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, list[str]],
                             list[pd.DataFrame]]:
    """Paper and rerun rows of the ResNet-18 models, stored shortlists, and checks.

    Raises:
        RuntimeError: If a stored shortlist differs between regimes.
    """
    *_, published, _, _ = prepare_pool(DEFAULT_OUT_ROOT)
    rerun = with_regimes(read_arm(resnet_root, "none", experiments), published)
    paper = published[published["model_id"].isin(rerun["model_id"])]
    preds = load_parquet(DEFAULT_OUT_ROOT / ABLATION_ROOT / "track1" / SPLIT / PREDICTOR
                         / "preds.parquet")
    preds = preds[preds["model_id"].isin(rerun["model_id"])]
    chosen = preds[preds["predicted_competitive"]]
    per_regime = chosen.groupby(["model_id", "regime"])["csf"].apply(
        lambda c: tuple(sorted(c))).unstack()
    if (per_regime.nunique(axis=1) > 1).any():
        raise RuntimeError("a stored ResNet-18 shortlist differs between regimes")
    shortlists = {m: list(per_regime.loc[m].iloc[0]) for m in per_regime.index}
    reference = [scored_rows(preds, rows).set_index(KEYS)
                 for rows in (paper, rerun)]
    return paper, rerun, shortlists, reference


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--vit-root", required=True)
    parser.add_argument("--resnet-root", required=True)
    parser.add_argument("--vit-experiments", default=str(HERE / "experiments_vit.txt"))
    parser.add_argument("--resnet-experiments",
                        default=str(HERE / "experiments_rn18_do0.txt"))
    parser.add_argument("--out", default=str(HERE / "outputs" / "noise_report.md"))
    args = parser.parse_args()

    tables, checks = {}, []
    for family, loader, root, listing in (
            ("ViT", vit_inputs, args.vit_root, args.vit_experiments),
            ("ResNet18", resnet_inputs, args.resnet_root, args.resnet_experiments)):
        experiments = Path(listing).read_text().split()
        paper, rerun, shortlists, reference = loader(Path(root), experiments)
        rows = noise_rows(paper, rerun, shortlists)
        keys = ["model_id", "eval_dataset"] if family == "ViT" else KEYS
        keyed = rows.set_index(keys)
        for name, suffix, table in (("paper", "p", reference[0]),
                                    ("rerun", "r", reference[1])):
            pairs = [(f"r_{suffix}", "regret", "predictor")]
            if family == "ViT":
                pairs.append((f"q_{suffix}", "random", "random"))
            for mine, theirs, label in pairs:
                joined = keyed[[mine]].join(table[[theirs]], how="inner")
                gap = check(f"{family} {name} {label}", joined[mine], joined[theirs])
                checks.append(f"{family} {name}: {label} regret matches the pipeline "
                              f"on {len(joined)} rows (max deviation {gap:.1e})")
        tables[family] = headroom_table(rows)

    sections = [("Checks", "\n\n".join(checks))]
    for family, table in tables.items():
        sections.append((f"{family}: headroom H = (q - o) / q",
                         f"```\n{table.round(3).to_string(index=False)}\n```"))
    text = ("# Rerun variability and the ViT failure (exploratory)\n" + "".join(
        f"\n## {title}\n\n{body}\n" for title, body in sections))
    Path(args.out).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
