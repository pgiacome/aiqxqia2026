"""Reproducibility and output-management helpers."""

from __future__ import annotations

import json
import logging
import os
import platform
import random
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["env_info", "make_output_dir", "set_seed", "write_json"]


def set_seed(seed: int = 42) -> None:
    """Seed every source of randomness used in this project."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    logger.info("Random seed set to %d", seed)


def env_info() -> dict[str, str]:
    """Record the environment for reproducibility."""
    import scipy
    import sklearn

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "sklearn": sklearn.__version__,
    }


def make_output_dir(name: str, root: Path = Path("outputs")) -> Path:
    """Create ``outputs/{name}_{timestamp}/`` and return it."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = root / f"{name}_{stamp}"
    path.mkdir(parents=True, exist_ok=False)
    logger.info("Output directory: %s", path)
    return path


def _default(obj: Any) -> Any:
    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Not JSON serialisable: {type(obj)!r}")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write ``payload`` as indented JSON."""
    path.write_text(json.dumps(payload, indent=2, default=_default), encoding="utf-8")
    logger.info("Wrote %s", path)
