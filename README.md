# CSF invariance

This project tests whether Neural Collapse (NC) metrics can determine the ranking of confidence scoring functions (CSFs). It follows the NeurIPS 2026 paper "A Systematic Analysis of Out-of-Distribution Detection Under Representation and Training-Objective Shifts".

The paper code is a read-only submodule at `external/ood_systematic`, pinned to commit `563fb63`. This project imports it and does not copy or edit it. The protocol and the decision rule are in `../documentation/reparam_invariance_protocol_2026-11-02.md`.

## Contents

| Path | Purpose |
|---|---|
| `reparam/invariance_table.py` | Runs the paper's CSF and NC classes under last-layer reparameterizations and checks the derived table. |
| `reparam/run_twin.py` | Runs the unchanged paper pipeline on the centered twin of each checkpoint (HPC). |
| `reparam/analyze_twin.py` | Checks gates G1 to G3 and applies the decision rule (local). |
| `reparam/analyze_s0.py` | Stage S0 of the spectrum protocol: descriptors per family, pre-checks, and target profiles (local). |
| `reparam/analyze_s1.py` | Stage S1: the ViT transport test (local). |
| `reparam/analyze_d1.py` | Stage D1: the unit-scale tracking test (local). |
| `reparam/experiments_*.txt` | Checkpoint lists: gate stage, main stage, and feature statistics. |
| `reparam/outputs/` | Saved invariance table and reports. |
| `tests/` | Unit tests. They do not need `fd_shifts` or a GPU. |

## Local setup

Local work covers the invariance table, the tests, and the analysis.

```bash
git clone --recurse-submodules <repository-url>
```

```bash
uv sync
```

```bash
uv run pytest
```

```bash
uv run python -m reparam.invariance_table
```

## HPC runs

Run these from `code/` inside the paper container (`cesarclaros/systematic_ood`). Create a `.env` file that sets `EXPERIMENT_ROOT_DIR` and `DATASET_ROOT_DIR`, with the same values as the paper checkout. ViT models need a writable `TORCH_HOME` for the timm backbone. Set `TWIN_ROOT` to a folder with enough space, about 300 MB per checkpoint for the feature cache.

```bash
set -a; source .env; set +a; export TWIN_ROOT=/path/to/reparam_twin
```

Gate stage, 4 ResNet-18 models:

```bash
python reparam/run_twin.py --experiments-file reparam/experiments_gate.txt --arms none center --out-root "$TWIN_ROOT" 2>&1 | tee twin_gate.log
```

Feature statistics, 12 VGG-13 and 8 ViT models:

```bash
python reparam/run_twin.py --experiments-file reparam/experiments_stats.txt --stats-only --out-root "$TWIN_ROOT" 2>&1 | tee twin_stats.log
```

Main stage, 28 ResNet-18 models. Start this only after the gates pass:

```bash
python reparam/run_twin.py --experiments-file reparam/experiments_rn18_do0.txt --arms none center --out-root "$TWIN_ROOT" 2>&1 | tee twin_main.log
```

A run can restart. The runner skips the steps that finished. Results go under `$TWIN_ROOT`, and the checkpoints' own `params/` and `analysis/` folders stay untouched.

## Stage S0 of the spectrum protocol

This stage only reads training features. The ResNet-18 features are already cached, and the VGG-13 and ViT models extract them again.

```bash
python reparam/run_twin.py --experiments-file reparam/experiments_rn18_do0.txt --stats-only --out-root "$TWIN_ROOT" 2>&1 | tee twin_s0_rn18.log
```

```bash
CSF_BATCH_SIZE=32 python reparam/run_twin.py --experiments-file reparam/experiments_stats.txt --stats-only --out-root "$TWIN_ROOT" 2>&1 | tee twin_s0_stats.log
```

Copy `origin_stats/` to `twin_results/s0/`, then run the analysis locally:

```bash
uv run python reparam/analyze_s0.py --stats-dir twin_results/s0/origin_stats
```

## Stages S1, S2, and D1 of the spectrum protocol

The ResNet-18 arms reuse the feature cache and the `none` arm of the origin test, so S2 and D1 run one arm each. Remove the stale step markers of the old stats-only runs once before the first ViT run:

```bash
rm -rf "$TWIN_ROOT/done/none/vit" && find "$TWIN_ROOT/done/none" -type d -name '*bbvgg13*' -prune -exec rm -rf {} +
```

S2, 28 ResNet-18 models with the ViT bulk shape:

```bash
nohup python reparam/run_twin.py --experiments-file reparam/experiments_rn18_do0.txt --arms bulk_vit --out-root "$TWIN_ROOT" > twin_s2.log 2>&1 &
```

D1, 28 ResNet-18 models with standardized unit scales:

```bash
nohup python reparam/run_twin.py --experiments-file reparam/experiments_rn18_do0.txt --arms unitscale --out-root "$TWIN_ROOT" > twin_d1.log 2>&1 &
```

S1a, 8 ViT models, gates and run time only:

```bash
CSF_BATCH_SIZE=32 nohup python reparam/run_twin.py --experiments-file reparam/experiments_vit_s1a.txt --arms none bulk_vgg --out-root "$TWIN_ROOT" > twin_s1a.log 2>&1 &
```

S1b runs the same command on `reparam/experiments_vit.txt`, all 40 ViT models, after the S1a gates pass. The runner skips the 8 finished models. The control arm `bulk_away` runs only after outcome X.

Run one command at a time on a GPU. Local analyses:

```bash
uv run python reparam/analyze_twin.py --twin-root twin_results/s2 --arm bulk_vit --experiments-file reparam/experiments_rn18_do0.txt
```

```bash
uv run python reparam/analyze_d1.py --twin-root twin_results/d1 --experiments-file reparam/experiments_rn18_do0.txt
```

```bash
uv run python reparam/analyze_s1.py --twin-root twin_results/s1 --gates-only --experiments-file reparam/experiments_vit_s1a.txt
```

## Step 1 of the placement test

The script reads only the cached features, so it runs on CPU in a few minutes per model:

```bash
nohup python reparam/placement_stats.py --out-root "$TWIN_ROOT" --experiments-file reparam/experiments_vit.txt > twin_placement_vit.log 2>&1 &
```

```bash
nohup python reparam/placement_stats.py --out-root "$TWIN_ROOT" --experiments-file reparam/experiments_rn18_do0.txt > twin_placement_rn18.log 2>&1 &
```

Copy `$TWIN_ROOT/placement_stats/` to `twin_results/placement/`, then run locally:

```bash
uv run python reparam/analyze_placement.py --placement-dir twin_results/placement --vit-root twin_results/s1b --resnet-root twin_results/main
```

## Analysis

Copy `analysis/`, `params/NeuralCollapse_*`, and `origin_stats/` from `$TWIN_ROOT` to `twin_results/`. The feature cache stays on the HPC. Then run these locally:

```bash
uv run python reparam/analyze_twin.py --twin-root twin_results --gates-only --experiments-file reparam/experiments_gate.txt
```

```bash
uv run python reparam/analyze_twin.py --twin-root twin_results --experiments-file reparam/experiments_rn18_do0.txt
```
