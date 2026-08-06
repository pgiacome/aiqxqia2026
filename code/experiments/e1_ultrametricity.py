"""E1: how far is each encoding's induced distance from being an ultrametric?

Makes Theorems A and B visible, and reports two quantities rather than one, because
one is not enough. Strong-triangle violations detect whether the induced distance is
an ultrametric at all; *resolution depth* detects whether it is a useful one. A basis
encoding scores perfectly on the first and worst-possible on the second: its delta
kernel is the discrete metric, which is ultrametric while collapsing the hierarchy
entirely. That is precisely the ``v* = n`` corner of Theorem A, not a counterexample
to it.

Only the path-state encoding is both level-constant (``level_constancy = 0``) and
fully resolving (``resolution_depth = height + 1``).
"""

from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np

from experiments.common import build_dataset, encoding_zoo, parse_args, save_run, savefig
from padic_kernel.encoding import EncodingFactory
from padic_kernel.kernels import fidelity_gram
from padic_kernel.metrics import (
    dimension_lower_bound,
    gromov_delta,
    level_constancy,
    profile_residual,
    resolution_depth,
    strong_triangle_violations,
    violation_rate,
)
from padic_kernel.profiles import geometric_profile
from padic_kernel.tree import ancestor_matrix, lca_depth_matrix
from padic_kernel.utils import make_output_dir

logger = logging.getLogger(__name__)


def main() -> None:
    cfg = parse_args(__doc__ or "E1")
    out = make_output_dir("e1_ultrametricity")
    ds = build_dataset(cfg)
    lca = lca_depth_matrix(ancestor_matrix(ds.tree, ds.leaves))
    # Resolution depth cannot exceed the number of LCA levels the sample actually
    # populates: if no pair of sampled leaves meets at depth v, no kernel can
    # distinguish that level. Recorded so the ceiling is never read as a shortfall.
    levels_present = int(np.unique(lca).size)
    f = geometric_profile(ds.height, cfg.profile_base, cfg.profile_s)
    bound = dimension_lower_bound(f(lca))

    records = []
    for label, (name, kwargs) in encoding_zoo(ds, cfg, f).items():
        try:
            psi = EncodingFactory(name, **kwargs).states(ds.tree, ds.leaves)
        except ValueError as exc:
            logger.warning("skipping %s: %s", label, exc)
            records.append({"encoding": label, "skipped": str(exc)})
            continue
        k = fidelity_gram(psi)
        dist = 1.0 - k
        count, excess = strong_triangle_violations(dist)
        records.append(
            {
                "encoding": label,
                "dim": int(psi.shape[1]),
                "qubits": float(np.log2(psi.shape[1])),
                "meets_dimension_bound": bool(psi.shape[1] >= bound),
                "violations": count,
                "violation_rate": violation_rate(dist),
                "max_excess": excess,
                "gromov_delta": gromov_delta(dist),
                "profile_residual": profile_residual(k, lca, f),
                "level_constancy": level_constancy(k, lca),
                "resolution_depth": resolution_depth(k, lca),
            }
        )
        logger.info(
            "%-15s dim=%-6d violations=%-10d level_const=%.3e resolution=%d/%d",
            label,
            psi.shape[1],
            count,
            records[-1]["level_constancy"],
            records[-1]["resolution_depth"],
            levels_present,
        )

    # Classical reference points on the same sample.
    coords = np.arange(len(ds.leaves), dtype=float)[:, None]
    sq = (coords - coords.T) ** 2
    k_rbf = np.exp(-sq / (2.0 * np.median(sq[sq > 0])))
    k_onehot = np.exp(-(1.0 - np.eye(len(coords))))
    for label, k_cls in (("rbf_integer", k_rbf), ("rbf_onehot", k_onehot)):
        dist = 1.0 - k_cls
        count, excess = strong_triangle_violations(dist)
        records.append(
            {
                "encoding": label,
                "dim": int(len(ds.leaves)),
                "qubits": float(np.log2(len(ds.leaves))),
                "meets_dimension_bound": bool(len(ds.leaves) >= bound),
                "violations": count,
                "violation_rate": violation_rate(dist),
                "max_excess": excess,
                "gromov_delta": gromov_delta(dist),
                "profile_residual": profile_residual(k_cls, lca, f),
                "level_constancy": level_constancy(k_cls, lca),
                "resolution_depth": resolution_depth(k_cls, lca),
            }
        )

    results = {
        "records": records,
        "dataset": cfg.dataset,
        "leaves": int(len(ds.leaves)),
        "tree_height": int(ds.height),
        "tree_nodes": int(ds.tree.num_nodes),
        "radix": int(ds.radix),
        "profile_base": int(cfg.profile_base),
        "profile_s": float(cfg.profile_s),
        "source_nodes": int(ds.source_nodes),
        "source_height": int(ds.source_height),
        "path_state_qubits": int(np.ceil(np.log2(ds.tree.num_nodes))),
        "dimension_lower_bound": bound,
        "qubit_lower_bound": float(np.log2(bound)),
        "max_resolution_depth": levels_present,
        "tree_levels": int(ds.height) + 1,
    }
    save_run(out, cfg, results)

    _plot(records, ds, cfg, out, levels_present)


def _plot(records: list[dict], ds, cfg, out, levels_present: int) -> None:
    """Two panels, because violations alone do not separate the encodings."""
    plotted = [r for r in records if "violation_rate" in r]
    names = [r["encoding"] for r in plotted]
    colours = ["#1b4965" if n == "path_state" else "#9db4c0" for n in names]
    floor = 1e-7

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.3))

    rates = [max(r["violation_rate"], floor) for r in plotted]
    axes[0].bar(names, rates, color=colours)
    axes[0].set_yscale("log")
    axes[0].set_ylim(floor / 2, 1.0)
    axes[0].axhline(floor, color="#1b4965", linestyle=":", linewidth=0.8)
    axes[0].set_ylabel("strong-triangle violation rate")
    axes[0].set_title("(a) is the distance an ultrametric?", fontsize=9)
    axes[0].text(-0.4, floor * 1.3, "zero, drawn at the floor", fontsize=6, color="#1b4965")

    depths = [r["resolution_depth"] for r in plotted]
    axes[1].bar(names, depths, color=colours)
    axes[1].axhline(
        levels_present,
        color="#1b4965",
        linestyle="--",
        linewidth=0.9,
        label="LCA levels present in the sample",
    )
    axes[1].set_ylabel("resolution depth (distinct kernel values)")
    axes[1].set_title("(b) how much of the hierarchy survives?", fontsize=9)
    axes[1].legend(fontsize=7)

    for ax in axes:
        ax.set_xlabel("encoding")
        ax.tick_params(axis="x", rotation=30, labelsize=7)
    fig.suptitle(
        f"{cfg.dataset}: {len(ds.leaves)} leaves, height {ds.height}, radix {ds.radix}",
        fontsize=9,
    )
    savefig(fig, out, "e1_violations")


if __name__ == "__main__":
    main()
