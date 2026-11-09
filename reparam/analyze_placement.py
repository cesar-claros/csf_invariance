"""Step 1: does a five-number description of OOD placement find good ViT shortlists?

Reads the placement descriptors that `placement_stats.py` writes and the AUGRC of
arm none, builds nearest-neighbor shortlists from placement (P), from the NC metrics
(N), and from both (NP), and applies the rule of
../documentation/placement_explanation_protocol_2026-11-09.md.

Run from code/ after copying placement_stats/ from the HPC:

    python reparam/analyze_placement.py --placement-dir twin_results/placement \
        --vit-root twin_results/s1b --resnet-root twin_results/main
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reparam.analyze_noise import random_regret, regret_matrix
from reparam.analyze_s1 import (
    add_model_id,
    load_parquet,
    published_vit,
    vit_arm,
)
from reparam.analyze_twin import (
    DEFAULT_OUT_ROOT,
    MODE_TO_EVAL,
    NC_METRICS,
    REGIMES,
    cluster_interval,
    model_id,
    prepare_pool,
    read_arm,
    with_regimes,
)

HERE = Path(__file__).resolve().parent
DESCRIPTORS = ("norm_ratio", "outside_share_gap", "class_cosine_gap", "margin_ratio",
               "centroid_offset")
NEIGHBORS = 20
SHORTLIST = 8
SENSITIVITY = (10, 40)
ROW_KEYS = ["model_id", "eval_dataset"]


def duplicate_rows(frame: pd.DataFrame) -> pd.Series:
    """TinyImageNet CIFAR-10 rows, which repeat the CIFAR-100 set in the pinned code."""
    return (frame["source"] == "tinyimagenet") & (frame["eval_dataset"] == "cifar10")


def load_placement(folder: Path, experiments: list[str]) -> pd.DataFrame:
    """Descriptors per (model, OOD set).

    Raises:
        SystemExit: If an experiment has no descriptor file.
    """
    records = []
    for experiment in experiments:
        path = folder / f"{experiment}.json"
        if not path.exists():
            raise SystemExit(f"no placement file for {experiment}")
        for mode, values in json.loads(path.read_text()).items():
            records.append({"model_id": model_id(experiment),
                            "eval_dataset": MODE_TO_EVAL[mode],
                            **{name: values[name] for name in DESCRIPTORS}})
    return pd.DataFrame(records)


def family_rows(long: pd.DataFrame, placement: pd.DataFrame, nc: pd.DataFrame,
                roster: list[str]) -> tuple[pd.DataFrame, np.ndarray]:
    """One row per (model, OOD set) with inputs, and the aligned regret matrix.

    Raises:
        RuntimeError: If a row misses a descriptor or a CSF.
    """
    wide = long.pivot_table(index=ROW_KEYS + ["source", "regime"], columns="csf",
                            values="raw_augrc")[roster]
    rows = wide.index.to_frame(index=False)
    keep = ~duplicate_rows(rows)
    rows, wide = rows[keep].reset_index(drop=True), wide[keep.to_numpy()]
    rows = rows.merge(placement, on=ROW_KEYS, how="left").merge(
        nc, left_on="model_id", right_index=True, how="left")
    missing = int(rows[list(DESCRIPTORS) + list(NC_METRICS)].isna().any(axis=1).sum())
    if missing or wide.isna().any().any():
        raise RuntimeError(f"{missing} rows miss inputs, or a CSF value is missing")
    return rows, regret_matrix(wide)


def zscore(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Values z-scored with the mean and spread of the reference rows."""
    std = reference.std(axis=0)
    std[std == 0] = 1.0
    return (values - reference.mean(axis=0)) / std


def knn_shortlists(train_x: np.ndarray, train_regret: np.ndarray,
                   test_x: np.ndarray, neighbors: int, size: int,
                   own_scale: bool = False) -> np.ndarray:
    """Shortlist per test row: the CSFs with the lowest mean regret over its neighbors.

    Inputs are z-scored on the training rows. With `own_scale`, the test rows are
    z-scored on themselves instead, the paper's per-architecture convention for
    transfer between families. Distance is Euclidean.
    """
    test_z = zscore(test_x, test_x if own_scale else train_x)
    distance = cdist(test_z, zscore(train_x, train_x))
    nearest = np.argsort(distance, axis=1, kind="stable")[:, :neighbors]
    scores = train_regret[nearest].mean(axis=1)
    return np.argsort(scores, axis=1, kind="stable")[:, :size]


def base_rate_shortlist(train_regret: np.ndarray, rows: int, size: int) -> np.ndarray:
    """The same shortlist for every test row: the lowest mean training regret."""
    best = np.argsort(train_regret.mean(axis=0), kind="stable")[:size]
    return np.tile(best, (rows, 1))


def picked_regret(regret: np.ndarray, shortlists: np.ndarray) -> np.ndarray:
    """Set regret per row of each row's shortlist."""
    return np.take_along_axis(regret, shortlists, axis=1).min(axis=1)


def arm_regrets(train: tuple[pd.DataFrame, np.ndarray],
                test: tuple[pd.DataFrame, np.ndarray],
                neighbors: int = NEIGHBORS, size: int = SHORTLIST,
                own_scale: bool = False) -> pd.DataFrame:
    """Regret of every arm on the test rows (see `knn_shortlists` for `own_scale`)."""
    train_rows, train_regret = train
    test_rows, test_regret = test
    inputs = {"P": list(DESCRIPTORS), "N": list(NC_METRICS),
              "NP": list(DESCRIPTORS) + list(NC_METRICS)}
    out = test_rows[ROW_KEYS + ["source", "regime"]].copy()
    for arm, columns in inputs.items():
        lists = knn_shortlists(train_rows[columns].to_numpy(float), train_regret,
                               test_rows[columns].to_numpy(float), neighbors, size,
                               own_scale)
        out[arm] = picked_regret(test_regret, lists)
    out["B"] = picked_regret(test_regret,
                             base_rate_shortlist(train_regret, len(test_rows), size))
    out["random"] = random_regret(test_regret, size)
    return out


def leave_one_source_out(rows: pd.DataFrame, regret: np.ndarray,
                         neighbors: int = NEIGHBORS) -> pd.DataFrame:
    """Each source is the test fold once; the other sources are the training rows."""
    folds = []
    for source in sorted(rows["source"].unique()):
        held = (rows["source"] == source).to_numpy()
        folds.append(arm_regrets((rows[~held], regret[~held]),
                                 (rows[held].reset_index(drop=True), regret[held]),
                                 neighbors))
    return pd.concat(folds, ignore_index=True)


def difference(scored: pd.DataFrame, arm: str, other: str) -> dict[str, float]:
    """Mean of arm minus other over rows, with the model-clustered interval."""
    gap = scored[arm] - scored[other]
    models = gap.groupby(scored["model_id"]).agg(["sum", "count"])
    low, high = cluster_interval(models["sum"].to_numpy(), models["count"].to_numpy())
    return {"comparison": f"{arm} - {other}", "mean": float(gap.mean()),
            "low": low, "high": high}


def decide(comparisons: pd.DataFrame) -> str:
    """The Step 1 rule on P - random, P - B, and P - N."""
    table = comparisons.set_index("comparison")
    if (table.loc[["P - random", "P - B", "P - N"], "high"] < 0).all():
        return "P1: placement carries the shortlist information"
    if (table.loc[["P - B", "P - random"], "low"] >= 0).any():
        return "P0: placement does not carry the shortlist information"
    return "PZ: inconclusive"


def summary(scored: pd.DataFrame) -> pd.DataFrame:
    """Mean regret of every arm, overall and per regime."""
    arms = ["P", "N", "NP", "B", "random"]
    overall = scored[arms].mean().to_frame("all").T
    by_regime = scored.groupby("regime")[arms].mean().reindex(list(REGIMES))
    return pd.concat([overall, by_regime])


def vit_long(root: Path, experiments: list[str]) -> pd.DataFrame:
    """ViT AUGRC of arm none, with source and regime."""
    _, published, regimes = published_vit()
    rows = vit_arm(root, "none", experiments, published)
    rows = rows.merge(regimes, on=["family", "eval_dataset"])
    return rows.rename(columns={"family": "source"})


def resnet_long(root: Path, experiments: list[str]) -> pd.DataFrame:
    """ResNet-18 AUGRC of arm none, with source and regime."""
    *_, published, _, _ = prepare_pool(DEFAULT_OUT_ROOT)
    rows = with_regimes(read_arm(root, "none", experiments), published)
    return rows.assign(source=rows["model_id"].str.split("|").str[2])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--placement-dir", required=True)
    parser.add_argument("--vit-root", required=True)
    parser.add_argument("--resnet-root", required=True)
    parser.add_argument("--out", default=str(HERE / "outputs" / "placement_report.md"))
    args = parser.parse_args()
    vit_experiments = (HERE / "experiments_vit.txt").read_text().split()
    resnet_experiments = (HERE / "experiments_rn18_do0.txt").read_text().split()
    placement = load_placement(Path(args.placement_dir),
                               vit_experiments + resnet_experiments)
    long_df = load_parquet(DEFAULT_OUT_ROOT / "track1" / "dataset"
                           / "long_harmonized.parquet")
    nc = (add_model_id(long_df).drop_duplicates("model_id")
          .set_index("model_id")[list(NC_METRICS)])
    vit = vit_long(Path(args.vit_root), vit_experiments)
    roster = sorted(vit["csf"].unique())
    vit_rows, vit_regret = family_rows(vit, placement, nc, roster)
    resnet = resnet_long(Path(args.resnet_root), resnet_experiments)
    resnet_rows, resnet_regret = family_rows(resnet, placement, nc, roster)

    primary = leave_one_source_out(vit_rows, vit_regret)
    comparisons = pd.DataFrame([difference(primary, "P", other)
                                for other in ("random", "B", "N")]
                               + [difference(primary, "NP", "N"),
                                  difference(primary, "N", "random")])
    outcome = decide(comparisons)
    cross = arm_regrets((resnet_rows, resnet_regret), (vit_rows, vit_regret),
                        own_scale=True)
    sensitivity = pd.DataFrame([
        {"K": k, **{c["comparison"]: c["mean"] for c in
                    (difference(leave_one_source_out(vit_rows, vit_regret, k), "P", o)
                     for o in ("random", "B", "N"))}} for k in SENSITIVITY])
    per_source = primary.groupby("source")[["P", "N", "NP", "B", "random"]].mean()
    sections = [
        ("Outcome", (f"**{outcome}**\n\n```\n"
                     f"{comparisons.round(3).to_string(index=False)}\n```")),
        ("Mean regret per arm, ViT leave one source out",
         f"```\n{summary(primary).round(3).to_string()}\n```"),
        ("Mean regret per arm and held-out source (exploratory)",
         f"```\n{per_source.round(3).to_string()}\n```"),
        ("CNN to ViT: trained on ResNet-18, tested on ViT (descriptive)",
         f"```\n{summary(cross).round(3).to_string()}\n```"),
        ("Sensitivity to K (descriptive)",
         f"```\n{sensitivity.round(3).to_string(index=False)}\n```"),
        ("Rows", (f"ViT {len(vit_rows)} rows, ResNet-18 {len(resnet_rows)} rows, "
                  f"{len(roster)} CSFs. TinyImageNet CIFAR-10 rows excluded.")),
    ]
    text = "# Step 1: OOD placement and the ViT shortlists\n" + "".join(
        f"\n## {title}\n\n{body}\n" for title, body in sections)
    Path(args.out).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
