"""E5: does exact ultrametricity survive depolarising noise and finite sampling?

Exactness is a statement about an ideal kernel; on hardware the kernel is estimated
from finitely many shots of a noisy state. The two error sources fail in genuinely
different ways, and the experiment separates them.

**Depolarising noise is a bias, and it cannot break the strong triangle inequality.**
It sends ``K -> (1 - r)K + r/D`` for the *register* dimension ``D``, hence
``d -> (1 - r)d + r(1 - 1/D)``, a non-decreasing affine map. Such maps commute with
``max`` and preserve order, so the inequality survives *exactly*, at any rate. What it
destroys is contrast: the profile flattens and resolution collapses as ``r -> 1``. This
is Proposition D in the paper, verified in
``test_proposition_d_depolarising_preserves_ultrametricity_exactly``. Note the profile
of the noisy kernel no longer satisfies ``f(n) = 1``, so strictly it is level constancy
plus the strong triangle inequality that survive, not ultrametricity as defined.

**Sampling noise is variance, and it is what actually breaks the inequality.** In an
ultrametric every triangle is isoceles with its two longest sides equal, so a constant
fraction of triples -- over a quarter, on the trees here -- satisfies the inequality
with equality. An arbitrarily small perturbation flips about half of those into
violations. The violation *count* therefore saturates near that tie fraction and does
not decay as the shot budget grows, which makes it useless as a robustness measure.
The magnitude does decay, as ``O(1/sqrt(shots))``, so the shot budget is defined on
the worst-case violation magnitude instead.
"""

from __future__ import annotations

import itertools
import logging

import matplotlib.pyplot as plt
import numpy as np

from experiments.common import build_dataset, parse_args, save_run, savefig
from padic_kernel.encoding import EncodingFactory
from padic_kernel.kernels import (
    depolarise,
    fidelity_gram,
    register_dimension,
    sample_kernel,
)
from padic_kernel.metrics import (
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

RATES = (0.0, 1e-2, 1e-1, 0.3, 0.5, 0.9, 0.99)
SHOTS: tuple[int | None, ...] = (100, 1_000, 10_000, 100_000, 1_000_000, None)
REPEATS = 10
#: Worst-case violation magnitude we are willing to tolerate, as a fraction of the
#: kernel's [0, 1] range. The shot budget is the smallest tested count that meets it.
TARGET_EXCESS = 0.01


def main() -> None:
    cfg = parse_args(__doc__ or "E5")
    out = make_output_dir("e5_noise")
    ds = build_dataset(cfg)
    lca = lca_depth_matrix(ancestor_matrix(ds.tree, ds.leaves))
    f = geometric_profile(ds.height, cfg.profile_base, cfg.profile_s)
    psi = EncodingFactory("path_state", profile=f).states(ds.tree, ds.leaves)
    exact = fidelity_gram(psi)
    # The register, not the support: Proposition D's floor is 1 / 2**ceil(log2 |V|).
    dim = register_dimension(int(psi.shape[1]))

    records = []
    for rate, shots in itertools.product(RATES, SHOTS):
        excesses, rates_seen, residuals, constancies, depths = [], [], [], [], []
        for rep in range(REPEATS if shots is not None else 1):
            rng = np.random.default_rng(cfg.seed + rep)
            k = depolarise(exact, rate, dim=dim)
            if shots is not None:
                k = sample_kernel(k, shots=shots, rng=rng)
            dist = 1.0 - k
            _, excess = strong_triangle_violations(dist)
            excesses.append(max(float(excess), 0.0))
            rates_seen.append(violation_rate(dist))
            residuals.append(profile_residual(k, lca, f))
            constancies.append(level_constancy(k, lca))
            depths.append(resolution_depth(k, lca, tol=1e-3))
        records.append(
            {
                "depolarising_rate": rate,
                "shots": shots if shots is not None else "exact",
                "max_excess_mean": float(np.mean(excesses)),
                "max_excess_std": float(np.std(excesses)),
                "violation_rate_mean": float(np.mean(rates_seen)),
                "profile_residual_mean": float(np.mean(residuals)),
                "level_constancy_mean": float(np.mean(constancies)),
                "resolution_depth_mean": float(np.mean(depths)),
            }
        )
        logger.info(
            "rate=%-5g shots=%-9s excess=%.3e viol_rate=%.3f resolution=%.1f",
            rate,
            shots,
            records[-1]["max_excess_mean"],
            records[-1]["violation_rate_mean"],
            records[-1]["resolution_depth_mean"],
        )

    noiseless = [
        r
        for r in records
        if r["depolarising_rate"] == 0.0
        and r["shots"] != "exact"
        and r["max_excess_mean"] < TARGET_EXCESS
    ]
    budget = min((int(r["shots"]) for r in noiseless), default=None)
    if budget is None:
        logger.warning(
            "no tested shot count kept the violation magnitude below %g; budget exceeds %s",
            TARGET_EXCESS,
            max(s for s in SHOTS if s is not None),
        )
    else:
        logger.info("shot budget: %d shots per entry for max excess < %g", budget, TARGET_EXCESS)

    tie_fraction = _tie_fraction(1.0 - exact)
    logger.info("tie fraction (triples meeting the inequality with equality): %.3f", tie_fraction)

    save_run(
        out,
        cfg,
        {
            "records": records,
            "target_excess": TARGET_EXCESS,
            "shot_budget": budget,
            "shot_budget_exceeds_tested": budget is None,
            "tie_fraction": tie_fraction,
            "depolarising_preserves_ultrametricity": all(
                r["max_excess_mean"] <= 1e-12 for r in records if r["shots"] == "exact"
            ),
            "levels_present": int(np.unique(lca).size),
            "dataset": cfg.dataset,
            "leaves": int(len(ds.leaves)),
            "support": int(psi.shape[1]),
            "register_dim": dim,
            "repeats": REPEATS,
        },
    )
    _plot(records, ds, cfg, out, budget)


def _tie_fraction(dist: np.ndarray, atol: float = 1e-12) -> float:
    """Fraction of triples meeting the strong triangle inequality with equality."""
    d_xy = dist[:, :, None]
    d_yz = dist[None, :, :]
    d_xz = dist[:, None, :]
    return float(np.isclose(d_xz, np.maximum(d_xy, d_yz), atol=atol).mean())


def _plot(records: list[dict], ds, cfg, out, budget: int | None) -> None:
    finite = sorted({int(r["shots"]) for r in records if r["shots"] != "exact"})
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.2))

    for rate in (0.0, 0.1, 0.5, 0.9):
        sel = sorted(
            (r for r in records if r["depolarising_rate"] == rate and r["shots"] != "exact"),
            key=lambda r: int(r["shots"]),
        )
        axes[0].loglog(
            [int(r["shots"]) for r in sel],
            [max(r["max_excess_mean"], 1e-9) for r in sel],
            marker="o",
            markersize=4,
            label=f"$p_{{dep}} = {rate:g}$",
        )
    axes[0].loglog(
        finite,
        [0.5 / np.sqrt(s) for s in finite],
        color="black",
        linestyle=":",
        label=r"$0.5/\sqrt{\mathrm{shots}}$",
    )
    axes[0].axhline(TARGET_EXCESS, color="grey", linestyle="--", linewidth=0.8)
    if budget is not None:
        axes[0].axvline(budget, color="#1b4965", linestyle="--", linewidth=0.8)
    axes[0].set_xlabel("shots per kernel entry")
    axes[0].set_ylabel("worst violation magnitude")
    axes[0].set_title("(a) sampling variance, and it decays", fontsize=9)
    axes[0].legend(fontsize=6)

    for rate in (0.0, 0.1, 0.5, 0.9):
        sel = sorted(
            (r for r in records if r["depolarising_rate"] == rate and r["shots"] != "exact"),
            key=lambda r: int(r["shots"]),
        )
        axes[1].semilogx(
            [int(r["shots"]) for r in sel],
            [r["violation_rate_mean"] for r in sel],
            marker="o",
            markersize=4,
            label=f"$p_{{dep}} = {rate:g}$",
        )
    axes[1].set_xlabel("shots per kernel entry")
    axes[1].set_ylabel("violation rate")
    axes[1].set_title("(b) but the count saturates: ties", fontsize=9)
    axes[1].legend(fontsize=6)

    for shots in (*finite, "exact"):
        sel = sorted(
            (r for r in records if r["shots"] == shots), key=lambda r: r["depolarising_rate"]
        )
        axes[2].semilogx(
            [max(r["depolarising_rate"], 1e-3) for r in sel],
            [r["resolution_depth_mean"] for r in sel],
            marker="s",
            markersize=4,
            label="noiseless estimate" if shots == "exact" else f"{shots:.0e} shots",
        )
    axes[2].set_xlabel("depolarising rate (drawn from $10^{-3}$)")
    axes[2].set_ylabel("resolution depth")
    axes[2].set_title("(c) depolarising bias flattens the profile", fontsize=9)
    axes[2].legend(fontsize=6)

    fig.suptitle(
        f"{cfg.dataset}: {len(ds.leaves)} leaves, height {ds.height}, dim {ds.tree.num_nodes}",
        fontsize=9,
    )
    savefig(fig, out, "e5_noise")


if __name__ == "__main__":
    main()
