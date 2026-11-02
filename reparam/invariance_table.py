"""Invariance of CSFs and NC metrics under last-layer reparameterizations.

Every map (x, W, b) -> (A x + c, W A^-1, b - W A^-1 c) keeps the logits of every
input, so it keeps the classifier. This script runs the pipeline's own CSF and
NC classes on synthetic rectified features before and after each map, labels
each cell, and compares the labels with the derived table EXPECTED.

Cell labels:
  "="  same score
  "~"  same ranking, different values (so the same AUGRC)
  "x"  different ranking

The table runs in float64, so a label reports the algebra and not float32 noise.
The features only exercise that algebra. The magnitudes say nothing about real
models; `run_twin.py` measures those.

Run from code/:  python -m reparam.invariance_table
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from scipy.stats import spearmanr

PAPER_CODE = Path(__file__).resolve().parents[1] / "external" / "ood_systematic"
OUT_DIR = Path(__file__).resolve().parent / "outputs"

ACTIONS = ("translate", "scale", "rotate", "diagonal", "linear")
NC_METRICS = (
    "var_collapse", "equiangular_uc", "equiangular_wc", "equinorm_uc",
    "equinorm_wc", "max_equiangular_uc", "max_equiangular_wc", "self_duality",
)
LOGIT_ONLY = ("MSR", "MLS", "Energy", "GEN", "REN", "PCE", "GE", "PE")

#: translate, scale, rotate, diagonal, linear. Derivations are in
#: ../documentation/reparam_invariance_protocol_2026-11-02.md. Maha assumes a
#: full-rank within-class covariance; NCI is at a fixed alpha > 0.
EXPECTED: dict[str, str] = {
    **{name: "=====" for name in LOGIT_ONLY},
    "GradNorm": "x~xxx",
    "pNML": "xx=xx",
    "CTM": "x==xx",
    "fDBD": "===xx",
    "NNGuide": "x==xx",
    "NeCo_CNN": "===xx",
    "NeCo_ViT": "x==xx",
    "Maha": "=====",
    "MahaPP": "x==xx",
    "NCI": "xxxxx",
    "ViM": "x==xx",
    "Residual": "x~=xx",
    "PCA_RecError_global": "x==xx",
    "KPCA_RecError_global": "x==xx",
    **{name: "===xx" for name in NC_METRICS},
}

EXACT_TOL = 1e-6
RANK_TOL = 1e-7


def install_stubs_if_missing() -> None:
    """Stand in for HPC-only packages so the table also runs on a laptop.

    Nothing is replaced when the real package imports. `src.utils` is replaced
    only when fd_shifts is absent, because it imports fd_shifts at module level.
    """
    if importlib.util.find_spec("fd_shifts") is None:
        def quiet(*_args: object, **_kwargs: object) -> None:
            return None

        fd_shifts = types.ModuleType("fd_shifts")
        vars(fd_shifts).update(logger=SimpleNamespace(
            info=quiet, warning=quiet, disable=quiet, enable=quiet))
        sys.modules["fd_shifts"] = fd_shifts

        def last_layer(module, study_name, return_model=True):
            if return_model:
                return module.model, module.w, module.b
            return module.w, module.b

        utils = types.ModuleType("src.utils")
        vars(utils).update(get_model_and_last_layer=last_layer)
        sys.modules["src.utils"] = utils
    for name in ("faiss", "bayes_opt"):
        if importlib.util.find_spec(name) is None:
            stub = types.ModuleType(name)
            vars(stub).update(BayesianOptimization=None)
            sys.modules[name] = stub
    if str(PAPER_CODE) not in sys.path:
        sys.path.insert(0, str(PAPER_CODE))


def synthetic_features(seed: int = 0, n_classes: int = 10, dim: int = 128,
                       n_train: int = 3000, n_eval: int = 500) -> dict:
    """Rectified Gaussian mixture, a nearby OOD cloud, and a linear head."""
    gen = torch.Generator().manual_seed(seed)
    decay = (1.0 + torch.arange(dim)) ** -0.5
    mixing = torch.randn(dim, dim, generator=gen) / dim ** 0.5
    means = 3 * decay * torch.randn(n_classes, dim, generator=gen)
    offset = 0.5 + torch.rand(dim, generator=gen)

    def sample(labels: torch.Tensor, spread: float, drift: float) -> torch.Tensor:
        latent = means[labels] * drift + spread * decay * torch.randn(
            len(labels), dim, generator=gen)
        return torch.relu(latent @ mixing.T + offset)

    labels = torch.randint(n_classes, (n_train,), generator=gen)
    train = sample(labels, 1.0, 1.0)
    eval_labels = torch.randint(n_classes, (n_eval,), generator=gen)
    evaluation = torch.vstack((sample(eval_labels, 1.0, 1.0),
                               sample(eval_labels, 1.6, 0.3)))
    unit = train.norm(dim=1).mean()
    train, evaluation = train / unit, evaluation / unit
    class_means = torch.stack([train[labels == c].mean(0)
                               for c in range(n_classes)])
    centered = class_means - class_means.mean(0)
    w = 4 * centered / centered.norm(dim=1, keepdim=True).mean() ** 2
    w = w + 0.05 * w.std() * torch.randn(n_classes, dim, generator=gen)
    b = 0.3 * torch.randn(n_classes, generator=gen)
    return {"train": train, "labels": labels, "eval": evaluation, "w": w, "b": b}


def make_action(action: str, data: dict, seed: int = 1
                ) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (A, c) for one reparameterization of this feature set."""
    dim = data["train"].shape[1]
    gen = torch.Generator().manual_seed(seed)
    eye = torch.eye(dim)
    zero = torch.zeros(dim)
    if action == "translate":
        return eye, -data["train"].mean(0)
    if action == "scale":
        return 7.0 * eye, zero
    if action == "rotate":
        q, _ = torch.linalg.qr(torch.randn(dim, dim, generator=gen))
        return q, zero
    if action == "diagonal":
        return torch.diag(torch.exp(0.7 * torch.randn(dim, generator=gen))), zero
    if action == "linear":
        u, _ = torch.linalg.qr(torch.randn(dim, dim, generator=gen))
        v, _ = torch.linalg.qr(torch.randn(dim, dim, generator=gen))
        singular = torch.exp(0.7 * torch.randn(dim, generator=gen))
        return u @ torch.diag(singular) @ v, zero
    raise ValueError(action)


def apply_action(data: dict, matrix: torch.Tensor, shift: torch.Tensor) -> dict:
    """Move the features and compensate the head so the logits do not change."""
    inverse = torch.linalg.inv(matrix)
    w_new = data["w"] @ inverse
    return {
        **data,
        "train": data["train"] @ matrix.T + shift,
        "eval": data["eval"] @ matrix.T + shift,
        "w": w_new,
        "b": data["b"] - w_new @ shift,
    }


def _config(n_classes: int, backbone: str) -> SimpleNamespace:
    return SimpleNamespace(
        data=SimpleNamespace(num_classes=n_classes),
        eval=SimpleNamespace(ext_confid_name="devries", confidence_measures=None),
        model=SimpleNamespace(test_mcd_samples=1,
                              network=SimpleNamespace(backbone=backbone, name=None)),
    )


def csf_scores(data: dict, n_classes: int = 10) -> dict[str, np.ndarray]:
    """All feature-and-head CSFs of the 22-CSF roster on data['eval'].

    Tuned hyperparameters are held at fixed values so that a change in a cell
    comes from the score, not from the tuner. Confidence needs the trained
    confidence head, so it is out of scope here.
    """
    from src import scores_funcs
    from src.csfs import (
        NCI,
        ClassTypicalMatching,
        GradNorm,
        KernelPCA,
        MahalanobisDistance,
        MahalanobisPP,
        NeCo,
        NNGuide,
        ProjectionFiltering,
        ResidualScore,
        ViMScore,
        fDBD,
        pNML,
    )
    from src.csfs.entropy_funcs import generalized_entropy, renyi_entropy

    train, labels, evaluation = data["train"], data["labels"], data["eval"]
    w, b = data["w"], data["b"]
    module = SimpleNamespace(model=SimpleNamespace(), w=w, b=b)
    study, cnn = "devries", _config(n_classes, "vgg13")
    logits_train = train @ w.T + b
    logits = evaluation @ w.T + b
    softmax = torch.softmax(logits, dim=1)

    scores = {
        "MSR": scores_funcs.maximum_softmax_response(softmax),
        "MLS": scores_funcs.maximum_logit_score(logits),
        "Energy": scores_funcs.energy(logits),
        "GEN": generalized_entropy(softmax, gamma=0.1, M=5),
        "REN": renyi_entropy(softmax, gamma=0.1, M=5),
        "PCE": scores_funcs.predictive_collision_entropy(softmax),
        "GE": scores_funcs.guessing_entropy(softmax),
        "PE": scores_funcs.predictive_entropy(softmax),
    }
    scores["GradNorm"] = GradNorm(module, study, cnn).get_scores(
        evaluation, use_cuda=False)

    pnml = pNML(module, study, cnn)
    pnml.compute_pNML_params(train)
    scores["pNML"] = pnml.get_scores(evaluation)

    ctm = ClassTypicalMatching(module, study, cnn, mode="global")
    ctm.compute_CTM_params(train, labels)
    scores["CTM"] = ctm.get_scores(evaluation, similarity="weight")

    fdbd = fDBD(module, study, cnn)
    fdbd.compute_fDBD_params(train)
    scores["fDBD"] = fdbd.get_scores(evaluation, logits_eval=logits)

    nnguide = NNGuide(module, study, cnn)
    nnguide.compute_NNGuide_params(train, labels_train=labels,
                                   logits_train=logits_train,
                                   proportion=0.2, k_clusters=10)
    scores["NNGuide"] = nnguide.get_scores(evaluation)

    for name, config in (("NeCo_CNN", cnn),
                         ("NeCo_ViT", _config(n_classes, "ViT"))):
        neco = NeCo(module, study, config)
        neco.compute_NeCo_params(train)
        scores[name] = neco.get_scores(evaluation)

    for name, cls in (("Maha", MahalanobisDistance), ("MahaPP", MahalanobisPP)):
        maha = cls(cnn)
        maha.compute_MahaDist_params(train, labels)
        scores[name] = maha.get_scores(evaluation)

    nci = NCI(module, study, cnn)
    nci.compute_NCI_params(train)
    scores["NCI"] = nci.get_scores(evaluation, logits_eval=logits)

    vim = ViMScore(module, study, cnn)
    vim.compute_ViM_params(train)
    scores["ViM"] = vim.get_scores(evaluation)

    residual = ResidualScore(module, study, cnn)
    residual.compute_Residual_params(train)
    scores["Residual"] = residual.get_scores(evaluation)

    projection = ProjectionFiltering(module, study, cnn, mode="global")
    projection.compute_ProjectionFiltering_params(
        train, labels=labels, only_correct=True, variance_explained=0.9)
    scores["PCA_RecError_global"] = projection.get_scores(evaluation)

    kernel = KernelPCA(module, study, cnn, mode="global")
    kernel.compute_KernelPCA_params(
        train, labels=labels, only_correct=False, temperature=1.0, n_landmarks=200,
        explained_variance=0.95, gamma=0.2, center_on="all", kernel="rbf")
    scores["KPCA_RecError_global"] = kernel.get_scores(evaluation)

    return {name: np.asarray(value, dtype=np.float64).ravel()
            for name, value in scores.items()}


def nc_values(data: dict, n_classes: int = 10) -> dict[str, float]:
    """The 8 NC predictor inputs from the pipeline's NeuralCollapseMetrics."""
    from src.neural_collapse import NeuralCollapseMetrics

    module = SimpleNamespace(model=SimpleNamespace(), w=data["w"], b=data["b"])
    metrics = NeuralCollapseMetrics(module, "devries", _config(n_classes, "vgg13"))
    metrics.compute_NeuralCollapse_params(data["train"], data["labels"])
    return {name: float(metrics.nc_metrics[name]) for name in NC_METRICS}


def label_score(before: np.ndarray, after: np.ndarray) -> tuple[str, float, float]:
    """Label one CSF cell; also return the scaled change and the rank correlation."""
    change = float(np.abs(after - before).max() / (before.std() + 1e-12))
    rank = float(spearmanr(before, after).statistic)
    if change < EXACT_TOL:
        return "=", change, rank
    return ("~" if 1 - rank < RANK_TOL else "x"), change, rank


def label_metric(before: float, after: float) -> tuple[str, float, float]:
    """Label one NC cell by its relative change."""
    change = abs(after - before) / (abs(before) + 1e-12)
    return ("=" if change < EXACT_TOL else "x"), change, float("nan")


def build_table(seed: int = 0) -> list[dict]:
    """Observed label and expected label for every (row, action) pair."""
    torch.set_default_dtype(torch.float64)
    data = synthetic_features(seed=seed)
    csf_before = csf_scores(data)
    nc_before = nc_values(data)
    rows = []
    for position, action in enumerate(ACTIONS):
        matrix, shift = make_action(action, data)
        moved = apply_action(data, matrix, shift)
        logits_gap = float(
            (moved["eval"] @ moved["w"].T + moved["b"]
             - (data["eval"] @ data["w"].T + data["b"])).abs().max())
        csf_after = csf_scores(moved)
        nc_after = nc_values(moved)
        cells = [(name, *label_score(value, csf_after[name]))
                 for name, value in csf_before.items()]
        cells += [(name, *label_metric(value, nc_after[name]))
                  for name, value in nc_before.items()]
        for name, label, change, rank in cells:
            rows.append({
                "row": name, "action": action, "observed": label,
                "expected": EXPECTED[name][position], "change": change,
                "spearman": rank, "logits_gap": logits_gap,
            })
    return rows


def render(rows: list[dict]) -> str:
    """Markdown table of observed labels; a mismatch prints as observed!expected."""
    names = list(dict.fromkeys(row["row"] for row in rows))
    lookup = {(row["row"], row["action"]): row for row in rows}
    lines = ["| row | " + " | ".join(ACTIONS) + " |",
             "|---|" + "---|" * len(ACTIONS)]
    for name in names:
        cells = []
        for action in ACTIONS:
            row = lookup[name, action]
            same = row["observed"] == row["expected"]
            cells.append(row["observed"] if same
                         else f"{row['observed']}!{row['expected']}")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    install_stubs_if_missing()
    rows = build_table(args.seed)
    OUT_DIR.mkdir(exist_ok=True)
    table = render(rows)
    (OUT_DIR / "invariance_table.md").write_text(table + "\n")
    columns = list(rows[0])
    (OUT_DIR / "invariance_table.csv").write_text(
        ",".join(columns) + "\n"
        + "\n".join(",".join(str(row[c]) for c in columns) for row in rows) + "\n")
    print(table)
    mismatches = sum(row["observed"] != row["expected"] for row in rows)
    print(f"\nmismatches against the derived table: {mismatches} of {len(rows)}")


if __name__ == "__main__":
    main()
