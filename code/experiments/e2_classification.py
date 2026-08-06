"""E2: hierarchical classification with precomputed quantum kernels.

Reported as parity evidence only. The paper's headline claim is exactness (E1) and
does not depend on this experiment. The success criterion was fixed in the design
spec before any number was seen: if kernel-SVM accuracy lands below v-PuNNs, that is
reported plainly, because v-PuNNs is a trained deep model and this is a fixed encoding
with a closed form. Nothing here is tuned to improve accuracy.

Two tasks are scored. *Leaf accuracy* predicts the depth-2 ancestor, the fine-grained
label. *Root accuracy* predicts the depth-1 ancestor, the coarse one. Spearman rho
measures how faithfully the kernel's induced distance ranks true tree distance, and it
is the quantity the paper actually cares about: geometric fidelity, not accuracy.
"""

from __future__ import annotations

import logging
import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC

from experiments.common import build_dataset, encoding_zoo, parse_args, save_run, savefig
from padic_kernel.encoding import EncodingFactory
from padic_kernel.kernels import fidelity_gram
from padic_kernel.profiles import geometric_profile
from padic_kernel.tree import ancestor_matrix, lca_depth_matrix
from padic_kernel.utils import make_output_dir

logger = logging.getLogger(__name__)

N_SPLITS = 5


def _viable(labels: np.ndarray) -> np.ndarray:
    """Mask of points whose class has at least ``N_SPLITS`` members.

    Stratified cross-validation needs that many; singleton classes are dropped and the
    count is reported rather than silently absorbed.
    """
    counts = np.bincount(labels)
    return counts[labels] >= N_SPLITS


def _safe_spearman(dist: np.ndarray, truth: np.ndarray) -> float | None:
    """Spearman rho, or ``None`` when the kernel is constant off the diagonal.

    A delta kernel -- basis encoding, one-hot RBF -- gives every distinct pair the same
    distance, so no rank correlation is defined. Reporting ``None`` says that; reporting
    NaN would read as a failed computation.
    """
    if float(np.ptp(dist)) < 1e-12:
        return None
    return float(spearmanr(dist, truth).statistic)


def _cross_validated_accuracy(
    kernel: np.ndarray, labels: np.ndarray, seed: int
) -> tuple[float, float]:
    if len(np.unique(labels)) < 2:
        return float("nan"), float("nan")
    folds = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    scores = []
    for train, test in folds.split(kernel, labels):
        clf = SVC(kernel="precomputed", C=1.0)
        clf.fit(kernel[np.ix_(train, train)], labels[train])
        scores.append(float(clf.score(kernel[np.ix_(test, train)], labels[test])))
    return float(np.mean(scores)), float(np.std(scores))


def main() -> None:
    cfg = parse_args(__doc__ or "E2")
    out = make_output_dir("e2_classification")
    ds = build_dataset(cfg)
    anc = ancestor_matrix(ds.tree, ds.leaves)
    f = geometric_profile(ds.height, cfg.profile_base, cfg.profile_s)

    fine = ds.labels
    coarse = np.unique(anc[:, 1], return_inverse=True)[1]
    keep = _viable(fine) & _viable(coarse)
    dropped = int((~keep).sum())
    if dropped:
        logger.warning(
            "dropping %d of %d leaves whose class has fewer than %d members",
            dropped,
            len(keep),
            N_SPLITS,
        )
    leaves = ds.leaves[keep]
    fine, coarse = fine[keep], coarse[keep]
    lca = lca_depth_matrix(ancestor_matrix(ds.tree, leaves))
    true_dist = (ds.height - lca).astype(float)
    iu = np.triu_indices(len(leaves), k=1)

    def _record(label: str, k: np.ndarray, dim: int, seconds: float) -> dict:
        leaf_mean, leaf_std = _cross_validated_accuracy(k, fine, cfg.seed)
        root_mean, root_std = _cross_validated_accuracy(k, coarse, cfg.seed)
        rho = _safe_spearman(1.0 - k[iu], true_dist[iu])
        logger.info(
            "%-15s leaf=%.3f+-%.3f root=%s rho=%s",
            label,
            leaf_mean,
            leaf_std,
            "n/a" if np.isnan(root_mean) else f"{root_mean:.3f}+-{root_std:.3f}",
            "undefined (constant kernel)" if rho is None else f"{rho:+.3f}",
        )
        return {
            "encoding": label,
            "dim": dim,
            "leaf_accuracy": leaf_mean,
            "leaf_accuracy_std": leaf_std,
            "root_accuracy": None if np.isnan(root_mean) else root_mean,
            "root_accuracy_std": None if np.isnan(root_std) else root_std,
            "spearman_rho": rho,
            "spearman_note": None if rho is not None else "undefined: kernel constant off-diagonal",
            "fit_seconds": seconds,
        }

    records = []
    for label, (name, kwargs) in encoding_zoo(ds, cfg, f).items():
        start = time.perf_counter()
        try:
            psi = EncodingFactory(name, **kwargs).states(ds.tree, leaves)
        except ValueError as exc:
            logger.warning("skipping %s: %s", label, exc)
            records.append({"encoding": label, "skipped": str(exc)})
            continue
        k = fidelity_gram(psi)
        records.append(_record(label, k, int(psi.shape[1]), time.perf_counter() - start))

    # Classical baselines on the same folds and the same sample.
    coords = np.arange(len(leaves), dtype=float)[:, None]
    sq = (coords - coords.T) ** 2
    records.append(
        _record("rbf_integer", np.exp(-sq / (2.0 * np.median(sq[sq > 0]))), len(leaves), 0.0)
    )
    records.append(
        _record("rbf_onehot", np.exp(-(1.0 - np.eye(len(leaves)))), len(leaves), 0.0)
    )

    save_run(
        out,
        cfg,
        {
            "records": records,
            "dataset": cfg.dataset,
            "leaves_used": int(len(leaves)),
            "leaves_dropped": dropped,
            "fine_classes": int(len(np.unique(fine))),
            "coarse_classes": int(len(np.unique(coarse))),
            "coarse_task_scored": bool(len(np.unique(coarse)) >= 2),
            "tree_height": int(ds.height),
            "n_splits": N_SPLITS,
        },
    )

    plotted = [r for r in records if "leaf_accuracy" in r]
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.3))
    names = [r["encoding"] for r in plotted]
    colours = ["#1b4965" if n == "path_state" else "#9db4c0" for n in names]
    axes[0].bar(
        names,
        [r["leaf_accuracy"] for r in plotted],
        yerr=[r["leaf_accuracy_std"] for r in plotted],
        capsize=3,
        color=colours,
    )
    axes[0].set_ylabel(f"leaf accuracy ({N_SPLITS}-fold)")
    axes[0].set_title("(a) classification", fontsize=9)
    rhos = [0.0 if r["spearman_rho"] is None else r["spearman_rho"] for r in plotted]
    axes[1].bar(names, rhos, color=colours)
    for i, r in enumerate(plotted):
        if r["spearman_rho"] is None:
            axes[1].text(i, 0.02, "undef.", ha="center", fontsize=6, rotation=90)
    axes[1].set_ylabel("Spearman rho vs true tree distance")
    axes[1].set_title("(b) geometric fidelity", fontsize=9)
    for ax in axes:
        ax.set_xlabel("encoding")
        ax.tick_params(axis="x", rotation=30, labelsize=7)
    fig.suptitle(
        f"{cfg.dataset}: {len(leaves)} leaves, {len(np.unique(fine))} classes", fontsize=9
    )
    savefig(fig, out, "e2_accuracy")


if __name__ == "__main__":
    main()
