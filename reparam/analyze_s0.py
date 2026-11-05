"""Stage S0: spectrum and unit-scale descriptors per family, and the pre-checks.

Reads the origin_stats/*.json files that `run_twin.py --stats-only` writes, builds
the median bulk profile per (family, source), and applies the S0 pre-checks fixed in
../documentation/spectrum_unit_scale_protocol_2026-11-04.md. The median profiles are
saved as the target profiles of the transplant twins.

Run from code/ after copying origin_stats/ from the HPC:

    python reparam/analyze_s0.py --stats-dir twin_results/s0/origin_stats
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SOURCES = ("cifar10", "cifar100", "supercifar100", "tinyimagenet")
SOURCE_NAMES = {"supercifar": "supercifar100", "super_cifar100": "supercifar100",
                "tiny-imagenet-200": "tinyimagenet"}
CNN = re.compile(r"^(?P<source>[\w-]+?)_paper_sweep/\w+?_bb(?P<arch>vgg13|resnet18)_")
VIT = re.compile(r"^vit/(?P<source>.+?)_modelvit_")
ARCH = {"vgg13": "VGG13", "resnet18": "ResNet18"}
DESCRIPTORS = ("bulk_share", "bulk_effective_rank", "bulk_floor_fraction",
               "unit_scale_cv", "bulk_dim")
REQUIRED_SOURCES = 3


def family_and_source(experiment: str) -> tuple[str, str]:
    """Family and source dataset of an experiment path.

    Raises:
        ValueError: If the path matches neither the CNN nor the ViT naming.
    """
    match = CNN.match(experiment)
    if match:
        family = ARCH[match["arch"]]
    else:
        match = VIT.match(experiment)
        if match is None:
            raise ValueError(f"unexpected experiment name: {experiment}")
        family = "ViT"
    return family, SOURCE_NAMES.get(match["source"], match["source"])


def load_stats(folder: Path) -> pd.DataFrame:
    """One row per model with its family, source, descriptors, and profile.

    Raises:
        SystemExit: If no file has the S0 descriptors.
    """
    records = []
    for path in sorted(folder.rglob("*.json")):
        stats = json.loads(path.read_text())
        if "bulk_profile" not in stats:
            continue
        family, source = family_and_source(path.relative_to(folder).with_suffix("")
                                           .as_posix())
        records.append({"family": family, "source": source, **stats})
    if not records:
        raise SystemExit(f"no S0 descriptors under {folder}")
    return pd.DataFrame(records)


def median_profiles(frame: pd.DataFrame) -> dict[tuple[str, str], np.ndarray]:
    """Median bulk profile per (family, source), taken quantile by quantile."""
    return {key: np.median(np.stack(group["bulk_profile"].to_list()), axis=0)
            for key, group in frame.groupby(["family", "source"])}


def shape_distance(first: np.ndarray, second: np.ndarray) -> float:
    """Mean absolute difference of the log profiles."""
    return float(np.abs(np.log(first) - np.log(second)).mean())


def precheck(profiles: dict[tuple[str, str], np.ndarray]) -> pd.DataFrame:
    """Distances per source for the two pre-checks of the protocol."""
    rows = []
    for source in SOURCES:
        vit, vgg, resnet = (profiles.get((family, source))
                            for family in ("ViT", "VGG13", "ResNet18"))
        if vit is None or vgg is None or resnet is None:
            continue
        rows.append({
            "source": source,
            "d_vit_vgg": shape_distance(vit, vgg),
            "d_resnet_vgg": shape_distance(resnet, vgg),
            "d_vit_resnet": shape_distance(vit, resnet),
        })
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    table["s1_separates"] = table["d_vit_vgg"] > table["d_resnet_vgg"]
    table["s2_separates"] = table["d_vit_resnet"] > table["d_resnet_vgg"]
    return table


def decisions(table: pd.DataFrame) -> dict[str, bool]:
    """S1 and S2 run only if their condition holds on at least 3 of the 4 sources."""
    if len(table) < len(SOURCES):
        return {"S1": False, "S2": False}
    return {"S1": int(table["s1_separates"].sum()) >= REQUIRED_SOURCES,
            "S2": int(table["s2_separates"].sum()) >= REQUIRED_SOURCES}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--stats-dir", required=True)
    parser.add_argument("--out-dir", default=str(HERE / "outputs"))
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frame = load_stats(Path(args.stats_dir))
    profiles = median_profiles(frame)
    table = precheck(profiles)
    decision = decisions(table)
    counts = frame.groupby(["family", "source"]).size().unstack(fill_value=0)
    summary = frame.groupby("family")[list(DESCRIPTORS)].median()

    targets = {family: {source: profile.tolist()
                        for (fam, source), profile in profiles.items() if fam == family}
               for family in sorted({fam for fam, _ in profiles})}
    (out_dir / "s0_profiles.json").write_text(json.dumps(targets, indent=2))
    report = "\n".join([
        "# Stage S0: spectrum and unit-scale descriptors", "",
        "## Models", "", "```", counts.to_string(), "```", "",
        "## Median descriptors per family", "", "```",
        summary.round(4).to_string(), "```", "",
        "## Shape distances per source", "", "```",
        table.round(4).to_string(index=False), "```", "",
        "## Pre-checks", "",
        f"S1 (ViT transport) runs: {decision['S1']}", "",
        f"S2 (ResNet-18 sufficiency) runs: {decision['S2']}", "",
    ])
    (out_dir / "s0_report.md").write_text(report)
    print(report)
    print(f"wrote {out_dir / 's0_report.md'} and {out_dir / 's0_profiles.json'}")


if __name__ == "__main__":
    main()
