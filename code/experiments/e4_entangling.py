"""E4: the expressivity / ultrametricity trade-off of the re-uploading family.

A single unitary applied after the path state leaves the kernel invariant, so any
non-trivial extension has to re-upload the data. The family here interleaves
data-dependent diagonal phases with fixed entanglers, and ``gamma = 0, entangle = 0``
recovers the exact construction, which anchors the front at distortion zero.

The interesting question is whether distortion grows continuously from zero -- making
the exact construction the endpoint of a tunable family rather than an isolated point
-- and what alignment is bought per unit of distortion.
"""

from __future__ import annotations

import itertools
import logging

import matplotlib.pyplot as plt

from experiments.common import build_dataset, parse_args, save_run, savefig
from padic_kernel.encoding import EncodingFactory
from padic_kernel.kernels import fidelity_gram
from padic_kernel.metrics import (
    kernel_target_alignment,
    level_constancy,
    resolution_depth,
    strong_triangle_violations,
)
from padic_kernel.profiles import geometric_profile
from padic_kernel.reupload import ReuploadEncoding
from padic_kernel.tree import ancestor_matrix, lca_depth_matrix
from padic_kernel.utils import make_output_dir

logger = logging.getLogger(__name__)

GAMMAS = (0.0, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)
ENTANGLES = (0.0, 0.25, 0.5, 1.0)
LAYERS = (1, 2, 4)


def _pareto_front(records: list[dict]) -> list[dict]:
    """Points not dominated on (low distortion, high alignment)."""
    front = []
    for r in sorted(records, key=lambda r: (r["distortion"], -r["alignment"])):
        if not front or r["alignment"] > front[-1]["alignment"] + 1e-12:
            front.append(r)
    return front


def main() -> None:
    cfg = parse_args(__doc__ or "E4")
    out = make_output_dir("e4_entangling")
    ds = build_dataset(cfg)
    lca = lca_depth_matrix(ancestor_matrix(ds.tree, ds.leaves))
    f = geometric_profile(ds.height, cfg.profile_base, cfg.profile_s)
    base = EncodingFactory("path_state", profile=f)

    records = []
    for gamma, entangle, layers in itertools.product(GAMMAS, ENTANGLES, LAYERS):
        enc = ReuploadEncoding(
            base=base, layers=layers, gamma=gamma, entangle=entangle, seed=cfg.seed
        )
        k = fidelity_gram(enc.states(ds.tree, ds.leaves))
        _, excess = strong_triangle_violations(1.0 - k)
        records.append(
            {
                "gamma": gamma,
                "entangle": entangle,
                "layers": layers,
                "distortion": max(float(excess), 0.0),
                "level_constancy": level_constancy(k, lca),
                "resolution_depth": resolution_depth(k, lca),
                "alignment": kernel_target_alignment(k, ds.labels),
            }
        )
    logger.info("Collected %d re-uploading configurations", len(records))

    exact = next(r for r in records if r["gamma"] == 0.0 and r["entangle"] == 0.0)
    front = _pareto_front(records)
    best = max(records, key=lambda r: r["alignment"])
    logger.info(
        "exact: distortion=%.3e alignment=%.4f | best alignment=%.4f at distortion=%.3f",
        exact["distortion"],
        exact["alignment"],
        best["alignment"],
        best["distortion"],
    )

    save_run(
        out,
        cfg,
        {
            "records": records,
            "pareto_front": front,
            "exact_point": exact,
            "best_alignment": best,
            "alignment_gain_over_exact": best["alignment"] - exact["alignment"],
            "dataset": cfg.dataset,
            "leaves": int(len(ds.leaves)),
        },
    )

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.3))
    for layers in LAYERS:
        sel = [r for r in records if r["layers"] == layers]
        axes[0].scatter(
            [r["distortion"] for r in sel],
            [r["alignment"] for r in sel],
            s=16,
            alpha=0.7,
            label=f"$L = {layers}$",
        )
    axes[0].plot(
        [r["distortion"] for r in front],
        [r["alignment"] for r in front],
        color="#1b4965",
        linewidth=1.0,
        label="Pareto front",
    )
    axes[0].scatter(
        [exact["distortion"]],
        [exact["alignment"]],
        marker="*",
        s=180,
        color="#1b4965",
        zorder=5,
        label="exact path state",
    )
    axes[0].set_xlabel("ultrametric distortion $D$")
    axes[0].set_ylabel("kernel-target alignment")
    axes[0].set_title("(a) the trade-off", fontsize=9)
    axes[0].legend(fontsize=6)

    for entangle in ENTANGLES:
        sel = sorted(
            (r for r in records if r["entangle"] == entangle and r["layers"] == 2),
            key=lambda r: r["gamma"],
        )
        axes[1].plot(
            [r["gamma"] for r in sel],
            [r["distortion"] for r in sel],
            marker="o",
            markersize=4,
            label=f"entangle $={entangle}$",
        )
    axes[1].set_xlabel(r"re-uploading phase $\gamma$")
    axes[1].set_ylabel("ultrametric distortion $D$")
    axes[1].set_title(r"(b) distortion grows continuously from $\gamma=0$", fontsize=9)
    axes[1].legend(fontsize=6)

    fig.suptitle(f"{cfg.dataset}: {len(ds.leaves)} leaves, height {ds.height}", fontsize=9)
    savefig(fig, out, "e4_pareto")


if __name__ == "__main__":
    main()
