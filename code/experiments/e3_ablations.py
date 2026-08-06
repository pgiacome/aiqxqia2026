"""E3: which design choices matter -- profile shape, tree depth, branching factor.

Two things this experiment must show. First, that exactness is not fragile: the
path-state residual stays at machine precision for every strictly monotone profile,
at every depth and radix. Second, the Theorem B crossing: for ``p >= 3`` the qubit
lower bound rises above ``n``, so no ``n``-qubit encoding can realise the kernel,
while at ``p = 2`` it stays below and Theorem A is what closes the product case.

Profiles are tied to the branching factor here, unlike E1-E2, because on the regular
p-ary tree ``base = p`` is the p-adic specialisation and that is the object of study.
"""

from __future__ import annotations

import itertools
import logging

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC

from experiments.common import parse_args, save_run, savefig
from padic_kernel.datasets import DatasetFactory, DatasetSpec
from padic_kernel.encoding import EncodingFactory
from padic_kernel.kernels import fidelity_gram
from padic_kernel.metrics import (
    dimension_lower_bound,
    level_constancy,
    profile_residual,
    regular_tree_qubit_bound,
    resolution_depth,
    strong_triangle_violations,
)
from padic_kernel.profiles import Profile, geometric_profile, linear_profile
from padic_kernel.tree import ancestor_matrix, lca_depth_matrix
from padic_kernel.utils import make_output_dir, set_seed

logger = logging.getLogger(__name__)

RADICES = (2, 3, 5)
DEPTHS = (2, 3, 4, 5, 6)
PROFILES = ("geometric_0.5", "geometric_1.0", "geometric_2.0", "linear")
MAX_LEAVES = 128


def _profile(name: str, n: int, p: int) -> Profile:
    if name == "linear":
        return linear_profile(n)
    return geometric_profile(n, p, float(name.split("_")[1]))


def _accuracy(kernel: np.ndarray, labels: np.ndarray, seed: int) -> float | None:
    counts = np.bincount(labels)
    if len(counts) < 2 or counts[counts > 0].min() < 5:
        return None
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    scores = [
        SVC(kernel="precomputed", C=1.0)
        .fit(kernel[np.ix_(tr, tr)], labels[tr])
        .score(kernel[np.ix_(te, tr)], labels[te])
        for tr, te in folds.split(kernel, labels)
    ]
    return float(np.mean(scores))


def main() -> None:
    cfg = parse_args(__doc__ or "E3")
    out = make_output_dir("e3_ablations")
    set_seed(cfg.seed)
    records = []
    for p, n, pname in itertools.product(RADICES, DEPTHS, PROFILES):
        if p**n > 4096:
            continue
        ds = DatasetFactory("synthetic", p=p, n=n)(
            DatasetSpec(
                name="synthetic", max_leaves=min(MAX_LEAVES, p**n), max_depth=n, seed=cfg.seed
            )
        )
        lca = lca_depth_matrix(ancestor_matrix(ds.tree, ds.leaves))
        f = _profile(pname, n, p)
        k = fidelity_gram(EncodingFactory("path_state", profile=f).states(ds.tree, ds.leaves))
        count, _ = strong_triangle_violations(1.0 - k)
        bound = dimension_lower_bound(f(lca))
        records.append(
            {
                "radix": p,
                "depth": n,
                "profile": pname,
                "leaves": int(len(ds.leaves)),
                "violations": count,
                "profile_residual": profile_residual(k, lca, f),
                "level_constancy": level_constancy(k, lca),
                "resolution_depth": resolution_depth(k, lca),
                "levels_present": int(np.unique(lca).size),
                "leaf_accuracy": _accuracy(k, ds.labels, cfg.seed),
                "dimension_lower_bound_sampled": bound,
                "qubit_lower_bound_sampled": float(np.log2(bound)),
                "qubit_lower_bound_full_tree": regular_tree_qubit_bound(f, p, n),
                "subsampled": bool(len(ds.leaves) < p**n),
                "qubits_used": float(np.log2(ds.tree.num_nodes)),
            }
        )
        logger.info(
            "p=%d n=%d %-14s violations=%d residual=%.2e qubit_bound=%.2f vs n=%d",
            p,
            n,
            pname,
            count,
            records[-1]["profile_residual"],
            records[-1]["qubit_lower_bound_full_tree"],
            n,
        )

    save_run(out, cfg, {"records": records, "max_leaves": MAX_LEAVES})
    _plot(records, out)


def _plot(records: list[dict], out) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.1))

    for pname in PROFILES:
        sel = sorted(
            (r for r in records if r["profile"] == pname and r["radix"] == 2),
            key=lambda r: r["depth"],
        )
        axes[0].semilogy(
            [r["depth"] for r in sel],
            [max(r["profile_residual"], 1e-18) for r in sel],
            marker="o",
            markersize=4,
            label=pname,
        )
    axes[0].axhline(1e-12, color="grey", linestyle="--", linewidth=0.8)
    axes[0].set_xlabel("tree depth $n$")
    axes[0].set_ylabel(r"$\max|K - f(\lambda)|$")
    axes[0].set_title("(a) exactness is not fragile", fontsize=9)
    axes[0].legend(fontsize=6)

    for p in RADICES:
        sel = sorted(
            (r for r in records if r["radix"] == p and r["profile"] == "geometric_2.0"),
            key=lambda r: r["depth"],
        )
        axes[1].plot(
            [r["depth"] for r in sel],
            [r["qubit_lower_bound_full_tree"] for r in sel],
            marker="s",
            markersize=4,
            label=f"Theorem B bound, $p={p}$",
        )
    depths = sorted({r["depth"] for r in records})
    axes[1].plot(depths, depths, color="black", linestyle=":", label="$n$ qubits")
    axes[1].set_xlabel("tree depth $n$")
    axes[1].set_ylabel("qubits")
    axes[1].set_title("(b) Theorem B crossing (full tree)", fontsize=9)
    axes[1].legend(fontsize=6)

    for pname in PROFILES:
        sel = sorted(
            (
                r
                for r in records
                if r["profile"] == pname and r["radix"] == 3 and r["leaf_accuracy"] is not None
            ),
            key=lambda r: r["depth"],
        )
        if sel:
            axes[2].plot(
                [r["depth"] for r in sel],
                [r["leaf_accuracy"] for r in sel],
                marker="^",
                markersize=4,
                label=pname,
            )
    axes[2].set_xlabel("tree depth $n$")
    axes[2].set_ylabel("leaf accuracy")
    axes[2].set_title("(c) profile choice vs accuracy, $p=3$", fontsize=9)
    axes[2].legend(fontsize=6)

    savefig(fig, out, "e3_ablation")


if __name__ == "__main__":
    main()
