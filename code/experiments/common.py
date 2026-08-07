"""Shared experiment plumbing: config parsing, dataset construction, output handling."""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from padic_kernel.datasets import DatasetFactory, DatasetSpec, LabelledTree  # noqa: E402
from padic_kernel.utils import env_info, set_seed, write_json  # noqa: E402

logger = logging.getLogger(__name__)

__all__ = [
    "ExperimentConfig",
    "build_dataset",
    "encoding_zoo",
    "parse_args",
    "save_run",
    "savefig",
]


@dataclass(frozen=True)
class ExperimentConfig:
    dataset: str = "synthetic"
    max_leaves: int = 128
    max_depth: int = 6
    seed: int = 42
    radix: int = 2
    depth: int = 6
    profile_s: float = 1.0
    profile_base: int = 2


def parse_args(description: str) -> ExperimentConfig:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--dataset", default="synthetic", choices=["synthetic", "wordnet", "go", "ncbi"]
    )
    parser.add_argument("--max-leaves", type=int, default=128)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--radix", type=int, default=2, help="synthetic branching factor")
    parser.add_argument("--depth", type=int, default=6, help="synthetic tree depth")
    parser.add_argument("--profile-s", type=float, default=1.0)
    parser.add_argument(
        "--profile-base",
        type=int,
        default=2,
        help=(
            "base of the geometric profile f(v) = base**(-(n - v) * s). Free design "
            "parameter; the p-adic specialisation is base = p. Pinning it to a large "
            "branching factor pushes the shallow levels below numerical resolution."
        ),
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s"
    )
    return ExperimentConfig(
        dataset=args.dataset,
        max_leaves=args.max_leaves,
        max_depth=args.max_depth,
        seed=args.seed,
        radix=args.radix,
        depth=args.depth,
        profile_s=args.profile_s,
        profile_base=args.profile_base,
    )


def build_dataset(cfg: ExperimentConfig) -> LabelledTree:
    set_seed(cfg.seed)
    spec = DatasetSpec(
        name=cfg.dataset, max_leaves=cfg.max_leaves, max_depth=cfg.max_depth, seed=cfg.seed
    )
    if cfg.dataset == "synthetic":
        return DatasetFactory("synthetic", p=cfg.radix, n=cfg.depth)(spec)
    return DatasetFactory(cfg.dataset)(spec)


def encoding_zoo(ds: LabelledTree, cfg: ExperimentConfig, profile: Any) -> dict[str, tuple]:
    """The encodings E1-E5 compare, with widths chosen so each one is representable.

    ``radix`` comes from the data, not the config: a real hierarchy's branching factor
    is what sets the angle-encoding rotation scale. ``random_product`` uses a local
    dimension of 2 so its width stays ``2**height`` rather than ``(radix + 2)**height``,
    and ``block_product`` is offered only when ``radix**height`` is representable --
    it is a Theorem A diagnostic on synthetic trees, not a headline baseline.
    """
    radix = ds.radix
    zoo: dict[str, tuple] = {
        "path_state": ("path_state", {"profile": profile}),
        "angle": ("angle", {"radix": radix}),
        "basis": ("basis", {"radix": radix}),
        "zz": ("zz", {"radix": radix, "reps": 2}),
        "random_product": ("random_product", {"radix": radix, "local_dim": 2, "seed": cfg.seed}),
    }
    if radix**ds.height <= 2**16:
        zoo["block_product"] = (
            "block_product",
            {"radix": radix, "block_size": 2, "seed": cfg.seed},
        )
    return zoo


def save_run(out: Path, cfg: ExperimentConfig, results: dict[str, Any]) -> None:
    write_json(out / "config.json", {"config": cfg, "env": env_info()})
    write_json(out / "results.json", results)


def savefig(fig: plt.Figure, out: Path, stem: str) -> None:
    fig.tight_layout()
    fig.savefig(out / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(out / f"{stem}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved figure %s", stem)
