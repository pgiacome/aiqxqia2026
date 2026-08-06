# Ultrametric Quantum Kernels — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a submission-ready CEUR-WS full paper (≥ 10 pages excl. references) plus a public, reproducible codebase establishing that no product quantum feature map and no sub-logarithmic-dimension feature map can induce a strictly monotone ultrametric kernel, and that path-state encoding realises every such kernel exactly at dimension within a constant factor of the proven lower bound.

**Architecture:** A small NumPy library (`padic_kernel`) exposes rooted trees, an encoding registry producing statevectors, fidelity Gram matrices, and ultrametricity diagnostics. Five experiment scripts consume the library and write JSON + PDF figures under `outputs/`. The LaTeX manuscript in `papers/aiqxqia2026/` reads numbers from those JSON artefacts so the paper can never drift from the code.

**Tech Stack:** Python 3.12, `uv`, NumPy, SciPy, scikit-learn, `nltk` (WordNet), `matplotlib`, `pytest`, `ruff`, `mypy`; LaTeX via `pdflatex` + `bibtex` with the `ceurart` class (TeX Live 2023 confirmed present locally).

---

## Global Constraints

- **Venue / format:** CEUR-WS `ceurart` class, single-column (`\documentclass[sigconf]{ceurart}` — CEUR's one-column mode), single-blind (author names **included**). Full paper: **≥ 10 pages excluding references**. Target 12–13 pages of body.
- **Deadline:** submit 9 August 2026; 10 August is buffer only. Today is 5 August 2026.
- **Author block:** single author; full name / affiliation / email are **open item #1** — until supplied, `main.tex` uses `\author{TODO-AUTHOR}` guarded by a visible `\todo{}`, and the paper must not be uploaded while any `TODO-AUTHOR` remains.
- **No fabricated citations.** Every `refs.bib` entry passes through the `reference-manager` agent and is verified against a real record (arXiv ID, DOI, or publisher page) before it is cited. Unverified claims get a visible `\todo{unverified}`.
- **References must be recent** (user requirement): prefer 2023–2026 sources; classical foundations (ultrametrics, p-adic analysis) may be older where they are the canonical origin.
- **Writing style:** mirror the editorial register of *Quantum Machine Intelligence* (Springer, ed.-in-chief Giovanni Acampora) — balanced mathematics and computer science, declarative, no hype, no "we believe", no promotional adjectives. Run the `humanizer` / `writing-anti-ai` pass before submission.
- **Honest scope, stated in §1:** `K(x,y) = f(λ(x,y))` is classically computable in `O(depth)`. **No quantum speedup in kernel evaluation is claimed anywhere in the paper.**
- **Code conventions:** 200–400 line modules; type hints on every function; module-level `logging.getLogger(__name__)`, never `print`; `@dataclass(frozen=True)` configs; factory/registry for encodings and datasets; `__all__` in every `__init__.py`; `set_seed(42)` at the top of every experiment; outputs to `outputs/{experiment}_{timestamp}/` with `config.json` and `env.json` alongside results.
- **Approved deviation from the standing global defaults:** frozen dataclasses + `argparse` instead of Hydra (five scripts, five days — Hydra is setup cost without payoff). All other conventions in `~/.claude/rules/coding-style.md` are followed as written.
- **Numerical tolerance:** exactness claims are asserted at `atol=1e-12`; observed residuals are reported at their true magnitude.

---

## Correction to the design spec, carried into this plan

The spec's §4 corollary asserts that angle, **IQP/ZZ**, and basis encodings all fall under Theorem 1. Angle and basis encodings do; **IQP/ZZ does not** — it is entangling, its fidelity kernel is not multiplicative across digit factors, and the multiplicativity step of the proof fails. Rather than weaken the claim, this plan replaces the unjustified corollary with a second, strictly more general theorem:

- **Theorem A (product no-go)** — the spec's Theorem 1, generalised from single digits to *disjoint digit blocks*. A block-product encoding with blocks of size ≤ `k` resolves the tree only to depth `v* + k`. Covers angle encoding, computational-basis encoding, and first-order Pauli-Z feature maps at `k = 1`.
- **Theorem B (dimension lower bound)** — new, and it is what actually kills IQP/ZZ. If unit vectors `Φ(x) ∈ ℂ^D` satisfy `|⟨Φ(x)|Φ(y)⟩|² = f(λ(x,y))` on `L` leaves, then the Gram matrix `G` is PSD with `tr G = L`, so `‖G‖_F² ≥ (tr G)²/rank(G)` gives

  ```
  D  ≥  rank(G)  ≥  L² / Σ_{x,y} f(λ(x,y)).
  ```

  For the regular p-ary tree of depth `n` with `f(v) = p^{-(n-v)s}`, `s > 1`, the row sum `S(p,s) = 1 + ((p−1)/p)·Σ_{m≥1} p^{m(1−s)}` is a constant, so `D ≥ p^n / S(p,s)` and the qubit count must satisfy `q ≥ n log₂ p − log₂ S(p,s)`. For every `p ≥ 3` this **exceeds `n`**, so *no* `n`-qubit encoding whatsoever — IQP, ZZ, hardware-efficient, anything — can realise a strictly monotone ultrametric kernel on `ℤ/pⁿ`. This is a rigorous statement about arbitrary encodings, not just product ones.

  **Verified numerically before this plan was written** (`s = 2`): `p=3, n=2` gives bound `6.94` = `2.80` qubits vs `n = 2`; `p=3, n=3` gives `20.44` = `4.35` qubits vs `3`; `p=5, n=2` gives `20.97` = `4.39` qubits vs `2`. The closed form `S(p,s,n)` matches the empirical row sum to machine precision in every case.

  **Complementarity, to be stated plainly in the paper.** At `p = 2` the bound gives only `q ≥ n − log₂ S(2,s) ≈ n − 0.58`, which does *not* exceed `n`. So Theorem B does not subsume Theorem A: for binary trees the dimension count alone permits an `n`-qubit encoding, and it is Theorem A's structural argument that rules out the product ones. The two theorems cover genuinely different regimes and the paper says so rather than presenting B as a strict generalisation.
- **Theorem C (exact realisation)** — the spec's Theorem 2, unchanged in substance, with two wording fixes: (i) the converse is uniqueness **up to a global isometry and per-point phases**, not "up to relabelling of `V`"; (ii) the optimality claim becomes *matching Theorem B to within an additive `log₂ S(p,s)` qubits*, rather than a bare "optimal width".

Contribution list becomes **C1** = Theorem A, **C2** = Theorem B, **C3** = Theorem C + resources, **C4** = generalisation to arbitrary rooted trees, **C5** = measured expressivity/hierarchy trade-off.

---

## File structure

| Path | Responsibility |
|------|----------------|
| `pyproject.toml` | uv project, deps, pytest/ruff/mypy config |
| `code/padic_kernel/__init__.py` | public API + `__all__` |
| `code/padic_kernel/tree.py` | `RootedTree`, p-ary builder, ancestors, LCA depth, digits, padding |
| `code/padic_kernel/profiles.py` | `Profile` frozen dataclass, `geometric`/`linear`/`uniform`, path amplitudes |
| `code/padic_kernel/encoding.py` | encoding registry; path-state, angle, basis, random-product, ZZ feature map |
| `code/padic_kernel/kernels.py` | statevectors → fidelity Gram matrix |
| `code/padic_kernel/metrics.py` | ultrametric violations, distortion, Gromov δ, alignment, dimension bound |
| `code/padic_kernel/datasets.py` | WordNet / GO / NCBI / synthetic loaders → `RootedTree` |
| `code/padic_kernel/utils.py` | `set_seed`, `env_info`, `make_output_dir`, JSON writer |
| `code/experiments/e1_ultrametricity.py` … `e5_noise.py` | one experiment each |
| `code/tests/test_tree.py`, `test_profiles.py`, `test_encoding.py`, `test_metrics.py`, `test_theorems.py` | unit + theorem-verification suites |
| `papers/aiqxqia2026/main.tex`, `ceurart.cls`, `refs.bib`, `figures/`, `numbers.tex` | manuscript; `numbers.tex` is generated from experiment JSON |
| `outputs/` | gitignored except `outputs/README.md` |

---

## Task 1: Project scaffold

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `code/padic_kernel/__init__.py`, `code/padic_kernel/utils.py`, `code/tests/__init__.py`, `outputs/README.md`

**Interfaces:**
- Consumes: nothing
- Produces: `padic_kernel.utils.set_seed(seed: int) -> None`, `padic_kernel.utils.env_info() -> dict[str, str]`, `padic_kernel.utils.make_output_dir(name: str, root: Path = Path("outputs")) -> Path`, `padic_kernel.utils.write_json(path: Path, payload: dict) -> None`

- [ ] **Step 1: Write `pyproject.toml`**

```toml
[project]
name = "padic-kernel"
version = "0.1.0"
description = "Ultrametric quantum kernels: p-adic feature maps for hierarchical data"
requires-python = ">=3.12"
dependencies = [
    "numpy>=2.1",
    "scipy>=1.14",
    "scikit-learn>=1.5",
    "matplotlib>=3.9",
    "nltk>=3.9",
]

[dependency-groups]
dev = ["pytest>=8.3", "ruff>=0.6", "mypy>=1.11"]

[build-system]
requires = ["setuptools>=75"]
build-backend = "setuptools.build_meta"

[tool.setuptools]
package-dir = {"" = "code"}
packages = ["padic_kernel"]

[tool.pytest.ini_options]
pythonpath = ["code"]
testpaths = ["code/tests"]
addopts = "-q"

[tool.ruff]
line-length = 100
target-version = "py312"

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B", "T20"]

[tool.mypy]
python_version = "3.12"
ignore_missing_imports = true
```

`T20` in the ruff select list is what mechanically enforces the no-`print` rule.

- [ ] **Step 2: Write `.gitignore`**

```gitignore
__pycache__/
*.py[cod]
.venv/
.pytest_cache/
.mypy_cache/
.ruff_cache/
outputs/*
!outputs/README.md
data/
papers/**/*.aux
papers/**/*.log
papers/**/*.out
papers/**/*.bbl
papers/**/*.blg
papers/**/*.fdb_latexmk
papers/**/*.fls
papers/**/*.synctex.gz
```

- [ ] **Step 3: Write `code/padic_kernel/utils.py`**

```python
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

__all__ = ["set_seed", "env_info", "make_output_dir", "write_json"]


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
```

- [ ] **Step 4: Write `code/padic_kernel/__init__.py`**

```python
"""Ultrametric quantum kernels: exact p-adic feature maps for hierarchical data."""

from padic_kernel.utils import env_info, make_output_dir, set_seed, write_json

__all__ = ["env_info", "make_output_dir", "set_seed", "write_json"]
```

Create `code/tests/__init__.py` as an empty file, and `outputs/README.md` containing the single line `Experiment outputs land here as {experiment}_{timestamp}/. Contents are gitignored.`

- [ ] **Step 5: Create the environment and verify**

Run:
```bash
cd /home/zerocold/source/papers/aiqxqia2026
uv sync --all-groups
uv run python -c "import padic_kernel; print(padic_kernel.env_info())"
uv run ruff check .
```
Expected: environment resolves, `env_info()` prints a dict, ruff is clean.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock .gitignore code outputs/README.md
git commit -m "chore(scaffold): uv project, package skeleton, reproducibility utils"
```

---

## Task 2: Rooted trees and LCA depth

**Files:**
- Create: `code/padic_kernel/tree.py`
- Test: `code/tests/test_tree.py`

**Interfaces:**
- Consumes: nothing
- Produces:
  - `RootedTree` frozen dataclass with fields `parent: np.ndarray` (int, `parent[root] = -1`), `depth: np.ndarray` (int), `height: int`, `labels: tuple[str, ...]`
  - `RootedTree.num_nodes -> int`, `RootedTree.root -> int`, `RootedTree.leaves -> np.ndarray` (nodes with no children)
  - `build_padic_tree(p: int, n: int) -> RootedTree`
  - `from_parent_map(parent_of: dict[str, str | None]) -> RootedTree`
  - `pad_to_uniform_depth(tree: RootedTree) -> RootedTree`
  - `ancestor_matrix(tree: RootedTree, leaves: np.ndarray) -> np.ndarray` shape `(m, height + 1)`, entry `[j, i] = anc_i(leaf_j)`
  - `lca_depth_matrix(anc: np.ndarray) -> np.ndarray` shape `(m, m)` int
  - `digit_matrix(tree: RootedTree, leaves: np.ndarray) -> np.ndarray` shape `(m, height)`, entry `[j, i-1]` = index of `anc_i` among the children of `anc_{i-1}`

- [ ] **Step 1: Write the failing tests**

```python
# code/tests/test_tree.py
import numpy as np
import pytest

from padic_kernel.tree import (
    ancestor_matrix,
    build_padic_tree,
    digit_matrix,
    from_parent_map,
    lca_depth_matrix,
    pad_to_uniform_depth,
)


def test_padic_tree_shape():
    t = build_padic_tree(p=3, n=2)
    # 1 root + 3 + 9 nodes
    assert t.num_nodes == 13
    assert t.height == 2
    assert len(t.leaves) == 9


def test_lca_depth_equals_p_adic_valuation():
    p, n = 3, 3
    t = build_padic_tree(p, n)
    anc = ancestor_matrix(t, t.leaves)
    lam = lca_depth_matrix(anc)
    for x in range(p**n):
        for y in range(p**n):
            d = x - y
            if d == 0:
                expected = n
            else:
                v = 0
                while d % p == 0:
                    d //= p
                    v += 1
                expected = min(v, n)
            assert lam[x, y] == expected, (x, y)


def test_digits_are_base_p_least_significant_first():
    p, n = 5, 2
    t = build_padic_tree(p, n)
    dig = digit_matrix(t, t.leaves)
    for x in range(p**n):
        assert dig[x, 0] == x % p
        assert dig[x, 1] == (x // p) % p


def test_from_parent_map_and_padding():
    #    root -> a -> a1, a2 ; root -> b   (b is a shallow leaf)
    parent_of = {"root": None, "a": "root", "b": "root", "a1": "a", "a2": "a"}
    t = from_parent_map(parent_of)
    assert t.height == 2
    padded = pad_to_uniform_depth(t)
    leaf_depths = padded.depth[padded.leaves]
    assert np.all(leaf_depths == padded.height)
    anc = ancestor_matrix(padded, padded.leaves)
    lam = lca_depth_matrix(anc)
    assert np.all(np.diag(lam) == padded.height)


def test_lca_depth_satisfies_strong_triangle():
    t = build_padic_tree(p=2, n=4)
    lam = lca_depth_matrix(ancestor_matrix(t, t.leaves))
    m = len(t.leaves)
    rng = np.random.default_rng(0)
    for _ in range(2000):
        x, y, z = rng.integers(0, m, size=3)
        assert lam[x, z] >= min(lam[x, y], lam[y, z])


def test_rejects_forest():
    with pytest.raises(ValueError, match="exactly one root"):
        from_parent_map({"a": None, "b": None})
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest code/tests/test_tree.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'padic_kernel.tree'`.

- [ ] **Step 3: Implement `code/padic_kernel/tree.py`**

```python
"""Rooted trees, ancestor paths, and lowest-common-ancestor depth.

Level convention. For a leaf ``x`` at depth ``n``, ``anc_i(x)`` is its depth-``i``
ancestor (``anc_0(x)`` is the root, ``anc_n(x) = x``). Digit ``i`` (1-indexed) is
the index of ``anc_i(x)`` among the children of ``anc_{i-1}(x)``. On the regular
p-ary tree built by :func:`build_padic_tree` this makes digit ``i`` the ``i``-th
least-significant base-p digit of the leaf index, so the LCA depth of two leaves
equals the p-adic valuation ``v_p(x - y)`` capped at ``n``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "RootedTree",
    "ancestor_matrix",
    "build_padic_tree",
    "digit_matrix",
    "from_parent_map",
    "lca_depth_matrix",
    "pad_to_uniform_depth",
]


@dataclass(frozen=True)
class RootedTree:
    """A finite rooted tree over node indices ``0 .. num_nodes - 1``."""

    parent: np.ndarray
    depth: np.ndarray
    labels: tuple[str, ...] = field(default=())

    def __post_init__(self) -> None:
        if self.parent.ndim != 1 or self.depth.shape != self.parent.shape:
            raise ValueError("parent and depth must be 1-D arrays of equal length")
        if int((self.parent < 0).sum()) != 1:
            raise ValueError("a rooted tree needs exactly one root")

    @property
    def num_nodes(self) -> int:
        return int(self.parent.shape[0])

    @property
    def height(self) -> int:
        return int(self.depth.max())

    @property
    def root(self) -> int:
        return int(np.flatnonzero(self.parent < 0)[0])

    @property
    def leaves(self) -> np.ndarray:
        has_child = np.zeros(self.num_nodes, dtype=bool)
        inner = self.parent[self.parent >= 0]
        has_child[inner] = True
        return np.flatnonzero(~has_child)

    def children(self, node: int) -> np.ndarray:
        return np.flatnonzero(self.parent == node)


def build_padic_tree(p: int, n: int) -> RootedTree:
    """The regular p-ary tree of depth ``n``; leaf ``x`` is the integer ``x`` in Z/p^n.

    Nodes are laid out level by level. The depth-``i`` level occupies indices
    ``off(i) .. off(i) + p**i - 1`` with ``off(i) = (p**i - 1) // (p - 1)``, and the
    node at position ``r`` of level ``i`` represents the residue class ``r mod p**i``.
    Hence ``anc_i(x) = off(i) + (x mod p**i)``.
    """
    if p < 2 or n < 1:
        raise ValueError("require p >= 2 and n >= 1")
    offsets = [(p**i - 1) // (p - 1) for i in range(n + 2)]
    num_nodes = offsets[n + 1]
    parent = np.empty(num_nodes, dtype=np.int64)
    depth = np.empty(num_nodes, dtype=np.int64)
    parent[0] = -1
    depth[0] = 0
    for i in range(1, n + 1):
        residues = np.arange(p**i, dtype=np.int64)
        idx = offsets[i] + residues
        parent[idx] = offsets[i - 1] + (residues % p ** (i - 1))
        depth[idx] = i
    logger.info("Built p-adic tree p=%d n=%d with %d nodes", p, n, num_nodes)
    return RootedTree(parent=parent, depth=depth)


def from_parent_map(parent_of: dict[str, str | None]) -> RootedTree:
    """Build a tree from a ``child -> parent`` label map (root maps to ``None``)."""
    names = sorted(parent_of)
    index = {name: i for i, name in enumerate(names)}
    parent = np.full(len(names), -1, dtype=np.int64)
    for name, par in parent_of.items():
        if par is not None:
            if par not in index:
                raise ValueError(f"parent {par!r} of {name!r} is not a node")
            parent[index[name]] = index[par]
    if int((parent < 0).sum()) != 1:
        raise ValueError("a rooted tree needs exactly one root")
    depth = _depths_from_parent(parent)
    return RootedTree(parent=parent, depth=depth, labels=tuple(names))


def _depths_from_parent(parent: np.ndarray) -> np.ndarray:
    depth = np.full(parent.shape[0], -1, dtype=np.int64)
    root = int(np.flatnonzero(parent < 0)[0])
    depth[root] = 0
    frontier = np.array([root], dtype=np.int64)
    level = 0
    while frontier.size:
        level += 1
        mask = np.isin(parent, frontier)
        frontier = np.flatnonzero(mask)
        depth[frontier] = level
    if int((depth < 0).sum()):
        raise ValueError("graph is not connected: some nodes are unreachable from the root")
    return depth


def pad_to_uniform_depth(tree: RootedTree) -> RootedTree:
    """Extend every shallow leaf by a chain of unary nodes down to ``tree.height``.

    Padding does not change LCA depths between the original leaves, because a unary
    chain adds no branching; it only makes every leaf sit at the same depth so that
    a single amplitude profile applies uniformly.
    """
    height = tree.height
    parents = list(map(int, tree.parent))
    depths = list(map(int, tree.depth))
    labels = list(tree.labels) if tree.labels else [str(i) for i in range(tree.num_nodes)]
    for leaf in map(int, tree.leaves):
        current, d = leaf, int(tree.depth[leaf])
        while d < height:
            d += 1
            parents.append(current)
            depths.append(d)
            labels.append(f"{labels[leaf]}#pad{d}")
            current = len(parents) - 1
    return RootedTree(
        parent=np.asarray(parents, dtype=np.int64),
        depth=np.asarray(depths, dtype=np.int64),
        labels=tuple(labels),
    )


def ancestor_matrix(tree: RootedTree, leaves: np.ndarray) -> np.ndarray:
    """``anc[j, i]`` is the depth-``i`` ancestor of ``leaves[j]``."""
    leaves = np.asarray(leaves, dtype=np.int64)
    height = tree.height
    if not np.all(tree.depth[leaves] == height):
        raise ValueError("all leaves must sit at the tree height; call pad_to_uniform_depth")
    anc = np.empty((leaves.shape[0], height + 1), dtype=np.int64)
    anc[:, height] = leaves
    for i in range(height - 1, -1, -1):
        anc[:, i] = tree.parent[anc[:, i + 1]]
    return anc


def lca_depth_matrix(anc: np.ndarray) -> np.ndarray:
    """Pairwise LCA depth from an ancestor matrix."""
    agree = anc[:, None, :] == anc[None, :, :]
    return agree.sum(axis=2).astype(np.int64) - 1


def digit_matrix(tree: RootedTree, leaves: np.ndarray) -> np.ndarray:
    """``dig[j, i - 1]`` is the child-index of ``anc_i`` among the children of ``anc_{i-1}``."""
    anc = ancestor_matrix(tree, leaves)
    order = np.zeros(tree.num_nodes, dtype=np.int64)
    for node in range(tree.num_nodes):
        kids = tree.children(node)
        order[kids] = np.arange(kids.shape[0], dtype=np.int64)
    return order[anc[:, 1:]]
```

`lca_depth_matrix` is `O(m² n)` in memory-light form; for the ~80k-leaf datasets the experiments subsample leaves (Task 9 fixes the sample sizes), so the dense pairwise form is never built at full scale.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest code/tests/test_tree.py -v`
Expected: all 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add code/padic_kernel/tree.py code/tests/test_tree.py
git commit -m "feat(tree): rooted trees, p-adic builder, LCA depth, digit extraction"
```

---

## Task 3: Kernel profiles and path amplitudes

**Files:**
- Create: `code/padic_kernel/profiles.py`
- Test: `code/tests/test_profiles.py`

**Interfaces:**
- Consumes: nothing
- Produces:
  - `Profile` frozen dataclass, field `values: tuple[float, ...]` (length `height + 1`, `values[-1] == 1.0`)
  - `Profile.height -> int`, `Profile.is_strictly_monotone -> bool`, `Profile.__call__(v: int | np.ndarray) -> np.ndarray`
  - `Profile.amplitudes() -> np.ndarray` returning `a` with `a[i]**2 = sqrt(f(i)) - sqrt(f(i-1))`
  - `geometric_profile(n: int, p: int, s: float = 1.0) -> Profile` with `f(v) = p ** (-(n - v) * s)`
  - `linear_profile(n: int) -> Profile` with `f(v) = (v + 1) / (n + 1)`
  - `uniform_profile(n: int) -> Profile` with `f(v) = 1.0` for all `v` (degenerate; ablation only)
  - `delta_profile(n: int) -> Profile` with `f(v) = 0` for `v < n`, `f(n) = 1` (basis-encoding profile; ablation only)

- [ ] **Step 1: Write the failing tests**

```python
# code/tests/test_profiles.py
import numpy as np
import pytest

from padic_kernel.profiles import (
    Profile,
    delta_profile,
    geometric_profile,
    linear_profile,
    uniform_profile,
)


def test_geometric_profile_values():
    f = geometric_profile(n=3, p=2, s=1.0)
    assert f.values == pytest.approx((0.125, 0.25, 0.5, 1.0))
    assert f.is_strictly_monotone


def test_amplitudes_normalise_and_reproduce_profile():
    for f in (geometric_profile(4, 3, 1.5), linear_profile(5)):
        a = f.amplitudes()
        assert np.sum(a**2) == pytest.approx(1.0, abs=1e-12)
        for v in range(f.height + 1):
            # overlap of two leaves whose LCA depth is v
            assert np.sum(a[: v + 1] ** 2) ** 2 == pytest.approx(f(v), abs=1e-12)


def test_non_monotone_profiles_are_flagged():
    assert not uniform_profile(3).is_strictly_monotone
    assert not delta_profile(3).is_strictly_monotone


def test_amplitudes_reject_non_monotone():
    with pytest.raises(ValueError, match="strictly monotone"):
        uniform_profile(3).amplitudes()


def test_profile_requires_terminal_one():
    with pytest.raises(ValueError, match="f\\(n\\) = 1"):
        Profile(values=(0.3, 0.7))


def test_call_is_vectorised():
    f = geometric_profile(3, 2, 1.0)
    out = f(np.array([0, 2, 3]))
    assert out == pytest.approx(np.array([0.125, 0.5, 1.0]))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest code/tests/test_profiles.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'padic_kernel.profiles'`.

- [ ] **Step 3: Implement `code/padic_kernel/profiles.py`**

```python
"""Kernel profiles ``f`` and the path-state amplitudes they induce."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "Profile",
    "delta_profile",
    "geometric_profile",
    "linear_profile",
    "uniform_profile",
]

_TOL = 1e-12


@dataclass(frozen=True)
class Profile:
    """A map ``f : {0, ..., n} -> [0, 1]`` with ``f(n) = 1``."""

    values: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.values) < 1:
            raise ValueError("profile needs at least one value")
        if abs(self.values[-1] - 1.0) > _TOL:
            raise ValueError("a kernel profile must satisfy f(n) = 1")
        if min(self.values) < -_TOL or max(self.values) > 1.0 + _TOL:
            raise ValueError("profile values must lie in [0, 1]")

    @property
    def height(self) -> int:
        return len(self.values) - 1

    @property
    def is_strictly_monotone(self) -> bool:
        v = np.asarray(self.values)
        return bool(np.all(np.diff(v) > _TOL) and v[0] > _TOL)

    def __call__(self, v: int | np.ndarray) -> np.ndarray:
        return np.asarray(self.values)[np.asarray(v)]

    def amplitudes(self) -> np.ndarray:
        """``a[i] = sqrt(sqrt(f(i)) - sqrt(f(i-1)))``, with ``sqrt(f(-1)) := 0``."""
        if not self.is_strictly_monotone:
            raise ValueError("path-state amplitudes require a strictly monotone profile")
        root = np.sqrt(np.asarray(self.values))
        squared = np.diff(root, prepend=0.0)
        return np.sqrt(squared)


def geometric_profile(n: int, p: int, s: float = 1.0) -> Profile:
    """``f(v) = p ** (-(n - v) * s)`` — the p-adic absolute value raised to ``s``."""
    if s <= 0:
        raise ValueError("require s > 0")
    return Profile(tuple(float(p ** (-(n - v) * s)) for v in range(n + 1)))


def linear_profile(n: int) -> Profile:
    """``f(v) = (v + 1) / (n + 1)`` — the slowest strictly monotone profile we use."""
    return Profile(tuple((v + 1) / (n + 1) for v in range(n + 1)))


def uniform_profile(n: int) -> Profile:
    """``f == 1``: the degenerate constant kernel. Ablation baseline only."""
    return Profile(tuple(1.0 for _ in range(n + 1)))


def delta_profile(n: int) -> Profile:
    """``f = 1[v = n]``: the kernel induced by computational-basis encoding."""
    return Profile(tuple(1.0 if v == n else 0.0 for v in range(n + 1)))
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest code/tests/test_profiles.py -v`
Expected: all 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add code/padic_kernel/profiles.py code/tests/test_profiles.py
git commit -m "feat(profiles): kernel profiles and path-state amplitude derivation"
```

---

## Task 4: Encoding registry and statevector maps

**Files:**
- Create: `code/padic_kernel/encoding.py`
- Test: `code/tests/test_encoding.py`

**Interfaces:**
- Consumes: `padic_kernel.tree` (`RootedTree`, `ancestor_matrix`, `digit_matrix`), `padic_kernel.profiles.Profile`
- Produces:
  - `Encoding` abstract base with `name: str` and `states(tree: RootedTree, leaves: np.ndarray) -> np.ndarray` returning a complex array of shape `(m, dim)` with unit rows
  - `register_encoding(name: str)` decorator and `EncodingFactory(name: str, **kwargs) -> Encoding`
  - `PathStateEncoding(profile: Profile)` — name `"path_state"`
  - `AngleEncoding(radix: int)` — name `"angle"`, `|ψ_i(a)⟩ = cos(θ_a/2)|0⟩ + sin(θ_a/2)|1⟩`, `θ_a = π a / radix`
  - `BasisEncoding(radix: int)` — name `"basis"`
  - `RandomProductEncoding(radix: int, local_dim: int, seed: int)` — name `"random_product"`
  - `ZZFeatureMapEncoding(radix: int, reps: int = 2)` — name `"zz"`, the standard IQP-style map
  - `BlockProductEncoding(radix: int, block_size: int, seed: int)` — name `"block_product"`, used to exercise Theorem A's block generalisation

- [ ] **Step 1: Write the failing tests**

```python
# code/tests/test_encoding.py
import numpy as np
import pytest

from padic_kernel.encoding import EncodingFactory
from padic_kernel.profiles import geometric_profile
from padic_kernel.tree import build_padic_tree


@pytest.mark.parametrize(
    "name,kwargs",
    [
        ("path_state", {"profile": geometric_profile(3, 2, 1.0)}),
        ("angle", {"radix": 2}),
        ("basis", {"radix": 2}),
        ("random_product", {"radix": 2, "local_dim": 3, "seed": 0}),
        ("zz", {"radix": 2, "reps": 2}),
        ("block_product", {"radix": 2, "block_size": 2, "seed": 0}),
    ],
)
def test_states_are_unit_vectors(name, kwargs):
    tree = build_padic_tree(2, 3)
    enc = EncodingFactory(name, **kwargs)
    psi = enc.states(tree, tree.leaves)
    assert psi.shape[0] == len(tree.leaves)
    assert np.allclose(np.linalg.norm(psi, axis=1), 1.0, atol=1e-12)


def test_path_state_dimension_is_node_count():
    tree = build_padic_tree(3, 2)
    enc = EncodingFactory("path_state", profile=geometric_profile(2, 3, 1.0))
    assert enc.states(tree, tree.leaves).shape[1] == tree.num_nodes


def test_angle_encoding_dimension_is_two_to_the_n():
    tree = build_padic_tree(2, 4)
    psi = EncodingFactory("angle", radix=2).states(tree, tree.leaves)
    assert psi.shape[1] == 2**4


def test_basis_encoding_gram_is_identity():
    tree = build_padic_tree(3, 2)
    psi = EncodingFactory("basis", radix=3).states(tree, tree.leaves)
    gram = np.abs(psi.conj() @ psi.T) ** 2
    assert np.allclose(gram, np.eye(len(tree.leaves)), atol=1e-12)


def test_zz_encoding_is_not_a_product_map():
    # A product map's kernel factorises; the ZZ map's does not. We detect this by
    # checking that changing one digit changes the kernel by a factor that itself
    # depends on the other digits.
    tree = build_padic_tree(2, 3)
    psi = EncodingFactory("zz", radix=2, reps=2).states(tree, tree.leaves)
    k = np.abs(psi.conj() @ psi.T) ** 2
    # leaves 0b000=0, 0b001=1, 0b010=2, 0b011=3
    assert not np.isclose(k[0, 1] * k[0, 2], k[0, 3] * k[0, 0], atol=1e-9)


def test_unknown_encoding_raises():
    with pytest.raises(KeyError, match="nope"):
        EncodingFactory("nope")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest code/tests/test_encoding.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'padic_kernel.encoding'`.

- [ ] **Step 3: Implement `code/padic_kernel/encoding.py`**

```python
"""Quantum feature maps as explicit statevectors.

Every encoding returns an ``(m, dim)`` complex array whose rows are unit vectors.
Fidelity kernels are formed downstream in :mod:`padic_kernel.kernels`; keeping the
statevectors explicit lets the same code serve exact, noisy, and finite-shot paths.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, TypeVar

import numpy as np

from padic_kernel.profiles import Profile
from padic_kernel.tree import RootedTree, ancestor_matrix, digit_matrix

logger = logging.getLogger(__name__)

__all__ = [
    "AngleEncoding",
    "BasisEncoding",
    "BlockProductEncoding",
    "Encoding",
    "EncodingFactory",
    "PathStateEncoding",
    "RandomProductEncoding",
    "ZZFeatureMapEncoding",
    "register_encoding",
]

ENCODING_REGISTRY: dict[str, type["Encoding"]] = {}
E = TypeVar("E", bound="Encoding")


def register_encoding(name: str) -> Callable[[type[E]], type[E]]:
    def decorator(cls: type[E]) -> type[E]:
        ENCODING_REGISTRY[name] = cls
        cls.name = name  # type: ignore[attr-defined]
        return cls

    return decorator


def EncodingFactory(name: str, **kwargs: Any) -> "Encoding":
    """Instantiate a registered encoding by name."""
    if name not in ENCODING_REGISTRY:
        raise KeyError(f"unknown encoding {name!r}; known: {sorted(ENCODING_REGISTRY)}")
    return ENCODING_REGISTRY[name](**kwargs)


class Encoding(ABC):
    """A feature map from tree leaves to unit vectors."""

    name: str = "encoding"

    @abstractmethod
    def states(self, tree: RootedTree, leaves: np.ndarray) -> np.ndarray:
        """Return the ``(m, dim)`` complex statevector array for ``leaves``."""


def _kron_rows(factors: list[np.ndarray]) -> np.ndarray:
    """Row-wise Kronecker product of ``(m, d_i)`` arrays."""
    out = factors[0]
    for nxt in factors[1:]:
        out = (out[:, :, None] * nxt[:, None, :]).reshape(out.shape[0], -1)
    return out


@register_encoding("path_state")
@dataclass
class PathStateEncoding(Encoding):
    """``Phi_f(x) = sum_i a_i |anc_i(x)>`` — the exact ultrametric construction."""

    profile: Profile

    def states(self, tree: RootedTree, leaves: np.ndarray) -> np.ndarray:
        if self.profile.height != tree.height:
            raise ValueError(
                f"profile height {self.profile.height} != tree height {tree.height}"
            )
        anc = ancestor_matrix(tree, leaves)
        amps = self.profile.amplitudes()
        psi = np.zeros((anc.shape[0], tree.num_nodes), dtype=np.complex128)
        rows = np.arange(anc.shape[0])[:, None]
        np.add.at(psi, (rows, anc), amps[None, :].astype(np.complex128))
        return psi


@register_encoding("angle")
@dataclass
class AngleEncoding(Encoding):
    """One qubit per level: ``|psi_i(a)> = cos(theta_a/2)|0> + sin(theta_a/2)|1>``."""

    radix: int

    def states(self, tree: RootedTree, leaves: np.ndarray) -> np.ndarray:
        dig = digit_matrix(tree, leaves)
        theta = np.pi * dig / self.radix
        factors = [
            np.stack([np.cos(theta[:, i] / 2), np.sin(theta[:, i] / 2)], axis=1).astype(
                np.complex128
            )
            for i in range(dig.shape[1])
        ]
        return _kron_rows(factors)


@register_encoding("basis")
@dataclass
class BasisEncoding(Encoding):
    """One qudit per level in the computational basis."""

    radix: int

    def states(self, tree: RootedTree, leaves: np.ndarray) -> np.ndarray:
        dig = digit_matrix(tree, leaves)
        factors = []
        for i in range(dig.shape[1]):
            block = np.zeros((dig.shape[0], self.radix), dtype=np.complex128)
            block[np.arange(dig.shape[0]), dig[:, i]] = 1.0
            factors.append(block)
        return _kron_rows(factors)


@register_encoding("random_product")
@dataclass
class RandomProductEncoding(Encoding):
    """A Haar-random product map: independent random unit vectors per (level, digit)."""

    radix: int
    local_dim: int
    seed: int = 0

    def states(self, tree: RootedTree, leaves: np.ndarray) -> np.ndarray:
        dig = digit_matrix(tree, leaves)
        rng = np.random.default_rng(self.seed)
        factors = []
        for i in range(dig.shape[1]):
            table = rng.normal(size=(self.radix, self.local_dim)) + 1j * rng.normal(
                size=(self.radix, self.local_dim)
            )
            table /= np.linalg.norm(table, axis=1, keepdims=True)
            factors.append(table[dig[:, i]])
        return _kron_rows(factors)


@register_encoding("block_product")
@dataclass
class BlockProductEncoding(Encoding):
    """A product map over consecutive blocks of ``block_size`` levels.

    Each block carries a random unit vector per joint digit assignment, so the block
    is maximally expressive internally while the map remains a tensor product across
    blocks. This is the object Theorem A's block generalisation constrains.
    """

    radix: int
    block_size: int
    seed: int = 0

    def states(self, tree: RootedTree, leaves: np.ndarray) -> np.ndarray:
        dig = digit_matrix(tree, leaves)
        rng = np.random.default_rng(self.seed)
        factors = []
        for start in range(0, dig.shape[1], self.block_size):
            block = dig[:, start : start + self.block_size]
            width = block.shape[1]
            codes = np.zeros(block.shape[0], dtype=np.int64)
            for j in range(width):
                codes = codes * self.radix + block[:, j]
            size = self.radix**width
            table = rng.normal(size=(size, size)) + 1j * rng.normal(size=(size, size))
            table /= np.linalg.norm(table, axis=1, keepdims=True)
            factors.append(table[codes])
        return _kron_rows(factors)


@register_encoding("zz")
@dataclass
class ZZFeatureMapEncoding(Encoding):
    """The IQP-style ZZ feature map: ``(U_phi H^{ot n})^{reps} |0>``.

    ``phi_i(x) = x_i`` and ``phi_{ij}(x) = (pi - x_i)(pi - x_j)``, the standard
    second-order Pauli-Z choice. The all-pairs ZZ terms make this map entangling, so
    it is *not* a product map and Theorem A does not apply to it; Theorem B does.
    """

    radix: int
    reps: int = 2

    def states(self, tree: RootedTree, leaves: np.ndarray) -> np.ndarray:
        dig = digit_matrix(tree, leaves)
        m, n = dig.shape
        angles = 2.0 * np.pi * dig / self.radix
        basis = np.arange(2**n)
        # z[k, i] = +1/-1 eigenvalue of Z_i on basis state k
        bits = ((basis[:, None] >> np.arange(n)[None, :]) & 1).astype(np.float64)
        z = 1.0 - 2.0 * bits
        phase = angles @ z.T  # (m, 2**n) single-qubit part
        for i in range(n):
            for j in range(i + 1, n):
                coupling = (np.pi - angles[:, i]) * (np.pi - angles[:, j])
                phase = phase + 2.0 * coupling[:, None] * (z[:, i] * z[:, j])[None, :]
        psi = np.full((m, 2**n), 2.0 ** (-n / 2), dtype=np.complex128)
        for _ in range(self.reps):
            psi = psi * np.exp(1j * phase)
            psi = _hadamard_all(psi, n)
        return psi / np.linalg.norm(psi, axis=1, keepdims=True)


def _hadamard_all(psi: np.ndarray, n: int) -> np.ndarray:
    """Apply ``H^{ot n}`` to each row of ``psi`` by the fast Walsh-Hadamard transform."""
    out = psi.reshape(psi.shape[0], *([2] * n)).copy()
    for axis in range(1, n + 1):
        a = np.take(out, 0, axis=axis)
        b = np.take(out, 1, axis=axis)
        out = np.stack([(a + b), (a - b)], axis=axis)
    return out.reshape(psi.shape[0], 2**n) / np.sqrt(2.0) ** n
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest code/tests/test_encoding.py -v`
Expected: all tests PASS (11 parametrised + 5). If `test_zz_encoding_is_not_a_product_map` fails, the ZZ phase convention has accidentally factorised — check that the `coupling` term is actually included.

- [ ] **Step 5: Commit**

```bash
git add code/padic_kernel/encoding.py code/tests/test_encoding.py
git commit -m "feat(encoding): registry with path-state, angle, basis, product, block, ZZ maps"
```

---

## Task 5: Fidelity kernels

**Files:**
- Create: `code/padic_kernel/kernels.py`
- Test: extend `code/tests/test_encoding.py` with a kernel section (no new file)

**Interfaces:**
- Consumes: nothing beyond NumPy
- Produces:
  - `fidelity_gram(psi: np.ndarray) -> np.ndarray` — real `(m, m)` array `|⟨ψ_x|ψ_y⟩|²`
  - `overlap_gram(psi: np.ndarray) -> np.ndarray` — complex `(m, m)` array `⟨ψ_x|ψ_y⟩`
  - `depolarise(kernel: np.ndarray, rate: float) -> np.ndarray` — `(1 - rate) * K + rate / dim` off-diagonal model, diagonal pinned to 1
  - `sample_kernel(kernel: np.ndarray, shots: int, rng: np.random.Generator) -> np.ndarray` — binomial finite-shot estimate of a fidelity kernel, symmetrised, diagonal pinned to 1

- [ ] **Step 1: Write the failing tests (append to `code/tests/test_encoding.py`)**

```python
from padic_kernel.kernels import depolarise, fidelity_gram, overlap_gram, sample_kernel


def test_fidelity_gram_diagonal_is_one():
    tree = build_padic_tree(2, 3)
    psi = EncodingFactory("angle", radix=2).states(tree, tree.leaves)
    k = fidelity_gram(psi)
    assert np.allclose(np.diag(k), 1.0, atol=1e-12)
    assert np.allclose(k, k.T, atol=1e-12)
    assert k.min() >= -1e-12 and k.max() <= 1.0 + 1e-12


def test_overlap_gram_is_positive_semidefinite():
    tree = build_padic_tree(2, 3)
    psi = EncodingFactory("path_state", profile=geometric_profile(3, 2, 1.0)).states(
        tree, tree.leaves
    )
    eigs = np.linalg.eigvalsh(overlap_gram(psi))
    assert eigs.min() > -1e-10


def test_sampling_converges_to_the_exact_kernel():
    tree = build_padic_tree(2, 3)
    psi = EncodingFactory("path_state", profile=geometric_profile(3, 2, 1.0)).states(
        tree, tree.leaves
    )
    k = fidelity_gram(psi)
    rng = np.random.default_rng(0)
    est = sample_kernel(k, shots=200_000, rng=rng)
    assert np.max(np.abs(est - k)) < 0.02


def test_depolarise_moves_towards_the_maximally_mixed_value():
    k = np.array([[1.0, 0.2], [0.2, 1.0]])
    out = depolarise(k, rate=0.5)
    assert out[0, 1] < k[0, 1]
    assert np.allclose(np.diag(out), 1.0)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest code/tests/test_encoding.py -k "gram or sampling or depolarise" -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'padic_kernel.kernels'`.

- [ ] **Step 3: Implement `code/padic_kernel/kernels.py`**

```python
"""Fidelity kernels, and simple noise and finite-shot models."""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["depolarise", "fidelity_gram", "overlap_gram", "sample_kernel"]


def overlap_gram(psi: np.ndarray) -> np.ndarray:
    """``G[x, y] = <psi_x | psi_y>`` for unit rows ``psi``."""
    return psi.conj() @ psi.T


def fidelity_gram(psi: np.ndarray) -> np.ndarray:
    """``K[x, y] = |<psi_x | psi_y>|^2``, clipped into ``[0, 1]``."""
    k = np.abs(overlap_gram(psi)) ** 2
    k = 0.5 * (k + k.T)
    return np.clip(k, 0.0, 1.0)


def depolarise(kernel: np.ndarray, rate: float) -> np.ndarray:
    """Global depolarising model on the swap-test outcome.

    With probability ``rate`` the estimate is replaced by an unbiased coin, which maps
    ``K -> (1 - rate) K + rate / 2``. The diagonal is pinned to 1 because ``K(x, x)``
    is never actually measured in a kernel-matrix protocol.
    """
    if not 0.0 <= rate <= 1.0:
        raise ValueError("depolarising rate must lie in [0, 1]")
    out = (1.0 - rate) * kernel + rate / 2.0
    np.fill_diagonal(out, 1.0)
    return out


def sample_kernel(kernel: np.ndarray, shots: int, rng: np.random.Generator) -> np.ndarray:
    """Binomial finite-shot estimate of a fidelity kernel."""
    if shots < 1:
        raise ValueError("shots must be positive")
    m = kernel.shape[0]
    iu = np.triu_indices(m, k=1)
    draws = rng.binomial(shots, np.clip(kernel[iu], 0.0, 1.0)) / shots
    out = np.eye(m)
    out[iu] = draws
    out[(iu[1], iu[0])] = draws
    return out
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest code/tests/test_encoding.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add code/padic_kernel/kernels.py code/tests/test_encoding.py
git commit -m "feat(kernels): fidelity Gram matrices with depolarising and shot-noise models"
```

---

## Task 6: Ultrametricity diagnostics

**Files:**
- Create: `code/padic_kernel/metrics.py`
- Test: `code/tests/test_metrics.py`

**Interfaces:**
- Consumes: NumPy only
- Produces:
  - `strong_triangle_violations(dist: np.ndarray, tol: float = 1e-9) -> tuple[int, float]` — `(count, max_excess)` over all ordered triples, with excess `d(x,z) − max(d(x,y), d(y,z))`
  - `violation_rate(dist: np.ndarray, tol: float = 1e-9) -> float` — count divided by the number of triples examined
  - `gromov_delta(dist: np.ndarray, base: int = 0) -> float` — four-point δ via the Gromov product with a fixed base point
  - `kernel_target_alignment(kernel: np.ndarray, labels: np.ndarray) -> float`
  - `dimension_lower_bound(kernel: np.ndarray) -> float` — `L² / Σ K`, the Theorem B bound
  - `qubit_lower_bound(kernel: np.ndarray) -> float` — `log2` of the above
  - `profile_residual(kernel: np.ndarray, lca: np.ndarray, profile) -> float` — `max |K − f(λ)|`

For `m` above `~400` the exact triple scan is `O(m³)`; `strong_triangle_violations` therefore accepts arrays only up to `m = 512` and raises above that, and the experiments subsample. This is deliberate: an approximate violation count would make the headline "zero violations" claim unfalsifiable.

- [ ] **Step 1: Write the failing tests**

```python
# code/tests/test_metrics.py
import numpy as np
import pytest

from padic_kernel.metrics import (
    dimension_lower_bound,
    gromov_delta,
    kernel_target_alignment,
    profile_residual,
    qubit_lower_bound,
    strong_triangle_violations,
    violation_rate,
)
from padic_kernel.profiles import geometric_profile
from padic_kernel.tree import ancestor_matrix, build_padic_tree, lca_depth_matrix


def _ultrametric_distance(p=2, n=3, s=1.0):
    tree = build_padic_tree(p, n)
    lca = lca_depth_matrix(ancestor_matrix(tree, tree.leaves))
    f = geometric_profile(n, p, s)
    return 1.0 - f(lca), lca, f


def test_true_ultrametric_has_no_violations():
    dist, _, _ = _ultrametric_distance()
    count, excess = strong_triangle_violations(dist)
    assert count == 0
    assert excess <= 0.0
    assert violation_rate(dist) == 0.0


def test_euclidean_distance_violates_the_strong_triangle():
    x = np.arange(8, dtype=float)[:, None]
    dist = np.abs(x - x.T)
    count, excess = strong_triangle_violations(dist)
    assert count > 0
    assert excess > 0.0


def test_gromov_delta_is_zero_for_an_ultrametric():
    dist, _, _ = _ultrametric_distance()
    assert gromov_delta(dist) == pytest.approx(0.0, abs=1e-12)


def test_dimension_lower_bound_matches_the_closed_form():
    # For the delta kernel K = I on L points, sum K = L, so the bound is exactly L.
    k = np.eye(16)
    assert dimension_lower_bound(k) == pytest.approx(16.0)
    assert qubit_lower_bound(k) == pytest.approx(4.0)


def test_dimension_lower_bound_is_one_for_the_constant_kernel():
    k = np.ones((16, 16))
    assert dimension_lower_bound(k) == pytest.approx(1.0)


def test_profile_residual_is_zero_for_the_exact_construction():
    from padic_kernel.encoding import EncodingFactory
    from padic_kernel.kernels import fidelity_gram

    tree = build_padic_tree(2, 3)
    lca = lca_depth_matrix(ancestor_matrix(tree, tree.leaves))
    f = geometric_profile(3, 2, 1.0)
    psi = EncodingFactory("path_state", profile=f).states(tree, tree.leaves)
    assert profile_residual(fidelity_gram(psi), lca, f) < 1e-12


def test_alignment_is_one_for_a_perfectly_matched_kernel():
    labels = np.array([0, 0, 1, 1])
    target = (labels[:, None] == labels[None, :]).astype(float)
    assert kernel_target_alignment(target, labels) == pytest.approx(1.0)


def test_violation_scan_refuses_oversized_inputs():
    with pytest.raises(ValueError, match="512"):
        strong_triangle_violations(np.zeros((600, 600)))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest code/tests/test_metrics.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'padic_kernel.metrics'`.

- [ ] **Step 3: Implement `code/padic_kernel/metrics.py`**

```python
"""Diagnostics: how far a kernel is from being ultrametric, and how expressive it is."""

from __future__ import annotations

import logging
from typing import Protocol

import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "dimension_lower_bound",
    "gromov_delta",
    "kernel_target_alignment",
    "profile_residual",
    "qubit_lower_bound",
    "strong_triangle_violations",
    "violation_rate",
]

MAX_EXACT_SCAN = 512


class _Callable(Protocol):
    def __call__(self, v: int | np.ndarray) -> np.ndarray: ...


def _excess(dist: np.ndarray) -> np.ndarray:
    """``E[x, y, z] = d(x, z) - max(d(x, y), d(y, z))``."""
    if dist.shape[0] > MAX_EXACT_SCAN:
        raise ValueError(
            f"exact triple scan is limited to {MAX_EXACT_SCAN} points; subsample first"
        )
    d_xy = dist[:, :, None]
    d_yz = dist[None, :, :]
    d_xz = dist[:, None, :]
    return d_xz - np.maximum(d_xy, d_yz)


def strong_triangle_violations(dist: np.ndarray, tol: float = 1e-9) -> tuple[int, float]:
    """Count triples breaking ``d(x, z) <= max(d(x, y), d(y, z))``, and the worst excess."""
    exc = _excess(dist)
    return int(np.count_nonzero(exc > tol)), float(exc.max())


def violation_rate(dist: np.ndarray, tol: float = 1e-9) -> float:
    """Violating triples as a fraction of all ordered triples."""
    count, _ = strong_triangle_violations(dist, tol=tol)
    m = dist.shape[0]
    return count / float(m**3)


def gromov_delta(dist: np.ndarray, base: int = 0) -> float:
    """Gromov hyperbolicity via the Gromov product with base point ``base``.

    ``delta = max over (x, y, z) of min(gp(x, y), gp(y, z)) - gp(x, z)`` where
    ``gp(x, y) = (d(x, w) + d(y, w) - d(x, y)) / 2``. An ultrametric space has
    ``delta = 0``.
    """
    w = base
    gp = 0.5 * (dist[:, w][:, None] + dist[:, w][None, :] - dist)
    inner = np.minimum(gp[:, :, None], gp[None, :, :])
    return float(np.max(inner - gp[:, None, :]))


def kernel_target_alignment(kernel: np.ndarray, labels: np.ndarray) -> float:
    """Centred-free Frobenius alignment between ``K`` and the label kernel ``y y^T``."""
    target = (labels[:, None] == labels[None, :]).astype(float)
    num = float(np.sum(kernel * target))
    den = float(np.linalg.norm(kernel) * np.linalg.norm(target))
    return num / den if den > 0 else 0.0


def dimension_lower_bound(kernel: np.ndarray) -> float:
    """Theorem B: ``D >= L^2 / sum_{x,y} K(x, y)``.

    The Gram matrix ``G`` of the feature vectors is PSD with unit diagonal, so
    ``trace(G) = L`` and ``||G||_F^2 = sum K``. For a PSD matrix of rank ``r``,
    ``||G||_F^2 >= trace(G)^2 / r``, which rearranges to the stated bound.
    """
    total = float(kernel.sum())
    if total <= 0:
        raise ValueError("kernel sum must be positive")
    return float(kernel.shape[0] ** 2) / total


def qubit_lower_bound(kernel: np.ndarray) -> float:
    """The Theorem B bound expressed in qubits."""
    return float(np.log2(dimension_lower_bound(kernel)))


def profile_residual(kernel: np.ndarray, lca: np.ndarray, profile: _Callable) -> float:
    """``max |K(x, y) - f(lambda(x, y))|``."""
    return float(np.max(np.abs(kernel - profile(lca))))
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest code/tests/test_metrics.py -v`
Expected: all 8 PASS.

- [ ] **Step 5: Update `code/padic_kernel/__init__.py` to re-export the public API**

```python
"""Ultrametric quantum kernels: exact p-adic feature maps for hierarchical data."""

from padic_kernel.encoding import Encoding, EncodingFactory, register_encoding
from padic_kernel.kernels import depolarise, fidelity_gram, overlap_gram, sample_kernel
from padic_kernel.metrics import (
    dimension_lower_bound,
    gromov_delta,
    kernel_target_alignment,
    profile_residual,
    qubit_lower_bound,
    strong_triangle_violations,
    violation_rate,
)
from padic_kernel.profiles import (
    Profile,
    delta_profile,
    geometric_profile,
    linear_profile,
    uniform_profile,
)
from padic_kernel.tree import (
    RootedTree,
    ancestor_matrix,
    build_padic_tree,
    digit_matrix,
    from_parent_map,
    lca_depth_matrix,
    pad_to_uniform_depth,
)
from padic_kernel.utils import env_info, make_output_dir, set_seed, write_json

__all__ = [
    "Encoding",
    "EncodingFactory",
    "Profile",
    "RootedTree",
    "ancestor_matrix",
    "build_padic_tree",
    "delta_profile",
    "depolarise",
    "digit_matrix",
    "dimension_lower_bound",
    "env_info",
    "fidelity_gram",
    "from_parent_map",
    "geometric_profile",
    "gromov_delta",
    "kernel_target_alignment",
    "lca_depth_matrix",
    "linear_profile",
    "make_output_dir",
    "overlap_gram",
    "pad_to_uniform_depth",
    "profile_residual",
    "qubit_lower_bound",
    "register_encoding",
    "sample_kernel",
    "set_seed",
    "strong_triangle_violations",
    "uniform_profile",
    "violation_rate",
    "write_json",
]
```

- [ ] **Step 6: Commit**

```bash
git add code/padic_kernel/metrics.py code/padic_kernel/__init__.py code/tests/test_metrics.py
git commit -m "feat(metrics): ultrametricity diagnostics and the Theorem B dimension bound"
```

---

## Task 7: Theorem verification harness

This is the task that makes the paper's mathematical claims machine-checked. Every theorem in the manuscript has a test here, and the test names are cited in the paper.

**Files:**
- Create: `code/tests/test_theorems.py`

**Interfaces:**
- Consumes: everything from Tasks 2–6
- Produces: no library code; produces the verification suite the paper cites by name

- [ ] **Step 1: Write the tests (they will fail only if the library is wrong — write them, run them, and expect PASS; if any fails, the theorem or the code is wrong and must be resolved before proceeding)**

```python
"""Numerical verification of Theorems A, B and C.

The paper cites these test names directly. A failure here means either the library
or a theorem statement is wrong; it must never be silenced.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from padic_kernel.encoding import EncodingFactory
from padic_kernel.kernels import fidelity_gram
from padic_kernel.metrics import (
    dimension_lower_bound,
    profile_residual,
    strong_triangle_violations,
)
from padic_kernel.profiles import geometric_profile, linear_profile
from padic_kernel.tree import ancestor_matrix, build_padic_tree, lca_depth_matrix

CASES = [(2, 3), (3, 2), (2, 4), (5, 2)]


def _setup(p: int, n: int):
    tree = build_padic_tree(p, n)
    lca = lca_depth_matrix(ancestor_matrix(tree, tree.leaves))
    return tree, lca


# --------------------------------------------------------------------------- Theorem A


@pytest.mark.parametrize("p,n", CASES)
@pytest.mark.parametrize("seed", range(5))
def test_theorem_a_random_product_maps_are_never_strictly_monotone(p, n, seed):
    """No product feature map induces a strictly monotone ultrametric kernel."""
    tree, lca = _setup(p, n)
    psi = EncodingFactory(
        "random_product", radix=p, local_dim=p + 2, seed=seed
    ).states(tree, tree.leaves)
    k = fidelity_gram(psi)
    # If K were ultrametric with profile f, K would be constant on each LCA level.
    spreads = [
        np.ptp(k[lca == v]) for v in range(n + 1) if np.any(lca == v)
    ]
    assert max(spreads) > 1e-6, "a random product map came out level-constant"


@pytest.mark.parametrize("p,n", [(2, 3), (3, 3)])
def test_theorem_a_conclusion_forces_at_most_three_kernel_values(p, n):
    """A product map that *is* ultrametric takes at most three kernel values.

    We build the extremal object the theorem predicts: level 1 non-trivial, all deeper
    levels trivial. Its kernel must be two-valued (v* = 0 case).
    """
    tree, lca = _setup(p, n)

    class _Extremal:
        def states(self, tree, leaves):
            from padic_kernel.tree import digit_matrix

            dig = digit_matrix(tree, leaves)
            theta = np.pi * dig[:, 0] / (2 * p)
            first = np.stack([np.cos(theta), np.sin(theta)], axis=1).astype(np.complex128)
            trivial = np.ones((dig.shape[0], 1), dtype=np.complex128)
            return np.concatenate([first, trivial], axis=1) / np.linalg.norm(
                np.concatenate([first, trivial], axis=1), axis=1, keepdims=True
            )

    k = fidelity_gram(_Extremal().states(tree, tree.leaves))
    distinct = np.unique(np.round(k, 9))
    assert len(distinct) <= 3


@pytest.mark.parametrize("block_size", [1, 2])
def test_theorem_a_block_generalisation_bounds_resolution_depth(block_size):
    """A block-product map resolves the tree only within the block containing level 1.

    Concretely: for leaves agreeing on levels 1..block_size, the kernel must be
    constant, because the remaining blocks are forced trivial by the theorem when the
    kernel is ultrametric. We verify the contrapositive numerically: a block-product
    map whose kernel *is* level-constant is level-constant only down to block_size.
    """
    p, n = 2, 4
    tree, lca = _setup(p, n)
    psi = EncodingFactory(
        "block_product", radix=p, block_size=block_size, seed=0
    ).states(tree, tree.leaves)
    k = fidelity_gram(psi)
    deep = [v for v in range(block_size + 1, n) if np.any(lca == v)]
    spreads = [np.ptp(k[lca == v]) for v in deep]
    assert max(spreads) > 1e-6


# --------------------------------------------------------------------------- Theorem B


@pytest.mark.parametrize("p,n", [(3, 2), (3, 3), (5, 2)])
def test_theorem_b_rules_out_every_n_qubit_encoding(p, n):
    """For p >= 3 the dimension bound exceeds 2**n, so no n-qubit map can work."""
    tree, lca = _setup(p, n)
    f = geometric_profile(n, p, s=2.0)
    bound = dimension_lower_bound(f(lca))
    assert bound > 2**n, f"bound {bound} did not exceed 2**{n}"


@pytest.mark.parametrize("p,n", CASES)
def test_theorem_b_bound_is_satisfied_by_the_exact_construction(p, n):
    """The path-state dimension |V| respects its own lower bound."""
    tree, lca = _setup(p, n)
    f = geometric_profile(n, p, s=2.0)
    assert tree.num_nodes >= dimension_lower_bound(f(lca)) - 1e-9


@pytest.mark.parametrize("p,n", [(3, 2), (3, 3)])
def test_theorem_b_zz_feature_map_is_below_the_bound(p, n):
    """The ZZ map lives in 2**n dimensions, which the bound forbids."""
    tree, lca = _setup(p, n)
    f = geometric_profile(n, p, s=2.0)
    psi = EncodingFactory("zz", radix=p, reps=2).states(tree, tree.leaves)
    assert psi.shape[1] < dimension_lower_bound(f(lca))
    assert profile_residual(fidelity_gram(psi), lca, f) > 1e-3


def test_theorem_b_closed_form_row_sum():
    """S(p, s) = 1 + ((p-1)/p) * sum_{m>=1} p**(m(1-s)) matches the empirical row sum."""
    p, n, s = 3, 4, 2.0
    _, lca = _setup(p, n)
    f = geometric_profile(n, p, s)
    empirical = float(f(lca)[0].sum())
    closed = 1.0 + ((p - 1) / p) * sum(p ** (m * (1 - s)) for m in range(1, n + 1))
    assert empirical == pytest.approx(closed, rel=1e-12)


# --------------------------------------------------------------------------- Theorem C


@pytest.mark.parametrize("p,n", CASES)
@pytest.mark.parametrize("profile_name", ["geometric", "linear"])
def test_theorem_c_path_states_realise_the_profile_exactly(p, n, profile_name):
    tree, lca = _setup(p, n)
    f = geometric_profile(n, p, 1.0) if profile_name == "geometric" else linear_profile(n)
    psi = EncodingFactory("path_state", profile=f).states(tree, tree.leaves)
    residual = profile_residual(fidelity_gram(psi), lca, f)
    assert residual < 1e-12, f"residual {residual:.3e}"


@pytest.mark.parametrize("p,n", CASES)
def test_theorem_c_induced_distance_is_exactly_ultrametric(p, n):
    tree, lca = _setup(p, n)
    f = geometric_profile(n, p, 1.0)
    psi = EncodingFactory("path_state", profile=f).states(tree, tree.leaves)
    if len(tree.leaves) > 256:
        pytest.skip("triple scan limited to 256 leaves for runtime")
    count, excess = strong_triangle_violations(1.0 - fidelity_gram(psi))
    assert count == 0
    assert excess <= 1e-12


def test_theorem_c_kernel_is_psd_via_the_hadamard_square_factorisation():
    """K = (A A^T) o 2 is PSD because the Schur product of PSD matrices is PSD."""
    p, n = 2, 4
    tree, lca = _setup(p, n)
    f = geometric_profile(n, p, 1.0)
    psi = EncodingFactory("path_state", profile=f).states(tree, tree.leaves)
    k = fidelity_gram(psi)
    assert np.linalg.eigvalsh(k).min() > -1e-10


def test_theorem_c_uniqueness_up_to_isometry_and_phase():
    """Any unitary rotation of the path states leaves the kernel unchanged."""
    p, n = 2, 3
    tree, lca = _setup(p, n)
    f = geometric_profile(n, p, 1.0)
    psi = EncodingFactory("path_state", profile=f).states(tree, tree.leaves)
    rng = np.random.default_rng(0)
    a = rng.normal(size=(psi.shape[1], psi.shape[1])) + 1j * rng.normal(
        size=(psi.shape[1], psi.shape[1])
    )
    q, _ = np.linalg.qr(a)
    phases = np.exp(1j * rng.uniform(0, 2 * np.pi, size=psi.shape[0]))[:, None]
    assert np.allclose(fidelity_gram(psi @ q.T * phases), fidelity_gram(psi), atol=1e-12)


def test_theorem_c_generalises_to_a_non_homogeneous_tree():
    """The construction needs only nesting of ancestors, not a regular branching factor."""
    from padic_kernel.profiles import Profile
    from padic_kernel.tree import from_parent_map, pad_to_uniform_depth

    parent_of = {
        "r": None,
        "a": "r", "b": "r", "c": "r",
        "a1": "a", "a2": "a", "a3": "a",
        "b1": "b",
        "c1": "c", "c2": "c",
    }
    tree = pad_to_uniform_depth(from_parent_map(parent_of))
    lca = lca_depth_matrix(ancestor_matrix(tree, tree.leaves))
    f = Profile((0.1, 0.4, 1.0))
    psi = EncodingFactory("path_state", profile=f).states(tree, tree.leaves)
    assert profile_residual(fidelity_gram(psi), lca, f) < 1e-12


# --------------------------------------------------------------- cross-check vs baselines


@pytest.mark.parametrize("name,kwargs", [("angle", {"radix": 2}), ("zz", {"radix": 2})])
def test_baselines_violate_the_strong_triangle_inequality(name, kwargs):
    tree, _ = _setup(2, 3)
    psi = EncodingFactory(name, **kwargs).states(tree, tree.leaves)
    count, _ = strong_triangle_violations(1.0 - fidelity_gram(psi))
    assert count > 0
```

- [ ] **Step 2: Run the full suite**

Run: `uv run pytest code/tests -v`
Expected: everything PASS. Any failure is a mathematical finding, not a test-tuning opportunity — record it in `notes/theorem-discrepancies.md` and resolve the statement before writing it up.

- [ ] **Step 3: Record the exact residuals for the paper**

Run:
```bash
uv run python - <<'PY'
import json, numpy as np
from padic_kernel import *
rows = []
for p, n in [(2,3),(2,5),(3,3),(5,2)]:
    tree = build_padic_tree(p, n)
    lca = lca_depth_matrix(ancestor_matrix(tree, tree.leaves))
    f = geometric_profile(n, p, 1.0)
    psi = EncodingFactory("path_state", profile=f).states(tree, tree.leaves)
    k = fidelity_gram(psi)
    rows.append({"p": p, "n": n, "leaves": int(len(tree.leaves)),
                 "nodes": int(tree.num_nodes),
                 "residual": profile_residual(k, lca, f)})
print(json.dumps(rows, indent=2))
PY
```
Save the output to `notes/theorem-residuals.json`. These numbers go verbatim into §5 of the paper.

- [ ] **Step 4: Commit**

```bash
git add code/tests/test_theorems.py notes/theorem-residuals.json
git commit -m "test(theorems): machine-checked verification of Theorems A, B and C"
```

---

## Task 8: Dataset loaders

**Files:**
- Create: `code/padic_kernel/datasets.py`
- Test: `code/tests/test_datasets.py`

**Interfaces:**
- Consumes: `padic_kernel.tree`
- Produces:
  - `DatasetSpec` frozen dataclass: `name: str`, `max_leaves: int`, `seed: int`
  - `register_dataset(name: str)` decorator, `DatasetFactory(name: str, **kwargs) -> Callable[[DatasetSpec], LabelledTree]`
  - `LabelledTree` frozen dataclass: `tree: RootedTree`, `leaves: np.ndarray`, `labels: np.ndarray` (int class ids for E2), `label_names: tuple[str, ...]`
  - `load_synthetic(spec: DatasetSpec, p: int, n: int) -> LabelledTree`
  - `load_wordnet(spec: DatasetSpec) -> LabelledTree`
  - `load_go(spec: DatasetSpec, obo_path: Path) -> LabelledTree`
  - `load_ncbi(spec: DatasetSpec, taxdump_path: Path) -> LabelledTree`

**Preprocessing rule stated in the paper (do not vary silently):** WordNet's hypernym relation is a DAG. We induce a tree by keeping, for each synset, the **longest** hypernym path to the root, breaking ties by the lexicographically smallest synset name. GO and NCBI get the same rule. This is a named limitation in §9, not a hidden choice.

- [ ] **Step 1: Write the failing tests**

```python
# code/tests/test_datasets.py
import numpy as np
import pytest

from padic_kernel.datasets import DatasetFactory, DatasetSpec
from padic_kernel.tree import ancestor_matrix, lca_depth_matrix


def test_synthetic_dataset_round_trips():
    spec = DatasetSpec(name="synthetic", max_leaves=64, seed=0)
    ds = DatasetFactory("synthetic", p=2, n=6)(spec)
    assert len(ds.leaves) == 64
    assert ds.labels.shape == ds.leaves.shape


def test_subsampling_is_deterministic_under_the_seed():
    spec = DatasetSpec(name="synthetic", max_leaves=32, seed=7)
    a = DatasetFactory("synthetic", p=2, n=6)(spec)
    b = DatasetFactory("synthetic", p=2, n=6)(spec)
    assert np.array_equal(a.leaves, b.leaves)


def test_all_sampled_leaves_sit_at_the_tree_height():
    spec = DatasetSpec(name="synthetic", max_leaves=16, seed=1)
    ds = DatasetFactory("synthetic", p=3, n=3)(spec)
    assert np.all(ds.tree.depth[ds.leaves] == ds.tree.height)
    lca = lca_depth_matrix(ancestor_matrix(ds.tree, ds.leaves))
    assert np.all(np.diag(lca) == ds.tree.height)


@pytest.mark.slow
def test_wordnet_loads_and_is_a_tree():
    spec = DatasetSpec(name="wordnet", max_leaves=200, seed=0)
    ds = DatasetFactory("wordnet")(spec)
    assert len(ds.leaves) == 200
    assert ds.tree.height >= 5
    assert int((ds.tree.parent < 0).sum()) == 1
```

Register the `slow` marker by adding `markers = ["slow: needs a downloaded corpus"]` to `[tool.pytest.ini_options]` in `pyproject.toml`.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest code/tests/test_datasets.py -v -m "not slow"`
Expected: FAIL — `ModuleNotFoundError: No module named 'padic_kernel.datasets'`.

- [ ] **Step 3: Implement `code/padic_kernel/datasets.py`**

```python
"""Hierarchical datasets reduced to rooted trees.

Every source here is a DAG in the wild. We reduce each to a tree by keeping the
longest root-ward path per node, breaking ties lexicographically. This is stated in
the paper as a named preprocessing choice, not hidden.
"""

from __future__ import annotations

import logging
import tarfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from padic_kernel.tree import (
    RootedTree,
    build_padic_tree,
    from_parent_map,
    pad_to_uniform_depth,
)

logger = logging.getLogger(__name__)

__all__ = [
    "DatasetFactory",
    "DatasetSpec",
    "LabelledTree",
    "load_go",
    "load_ncbi",
    "load_synthetic",
    "load_wordnet",
    "register_dataset",
]

DATASET_REGISTRY: dict[str, Callable[..., Callable[["DatasetSpec"], "LabelledTree"]]] = {}


@dataclass(frozen=True)
class DatasetSpec:
    """How much of a dataset to use, and with what seed."""

    name: str
    max_leaves: int = 256
    seed: int = 42


@dataclass(frozen=True)
class LabelledTree:
    """A padded tree, a leaf sample, and a class label per sampled leaf."""

    tree: RootedTree
    leaves: np.ndarray
    labels: np.ndarray
    label_names: tuple[str, ...] = ()


def register_dataset(name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        DATASET_REGISTRY[name] = fn
        return fn

    return decorator


def DatasetFactory(name: str, **kwargs: Any) -> Callable[[DatasetSpec], LabelledTree]:
    """Return a loader closure for the named dataset."""
    if name not in DATASET_REGISTRY:
        raise KeyError(f"unknown dataset {name!r}; known: {sorted(DATASET_REGISTRY)}")
    builder = DATASET_REGISTRY[name]

    def loader(spec: DatasetSpec) -> LabelledTree:
        return builder(spec, **kwargs)

    return loader


def _subsample(tree: RootedTree, spec: DatasetSpec) -> np.ndarray:
    leaves = tree.leaves
    if leaves.shape[0] <= spec.max_leaves:
        return leaves
    rng = np.random.default_rng(spec.seed)
    return np.sort(rng.choice(leaves, size=spec.max_leaves, replace=False))


def _labels_at_depth(tree: RootedTree, leaves: np.ndarray, depth: int) -> np.ndarray:
    """Class label = the ancestor at ``depth``, remapped to 0-based ids."""
    from padic_kernel.tree import ancestor_matrix

    anc = ancestor_matrix(tree, leaves)[:, depth]
    _, labels = np.unique(anc, return_inverse=True)
    return labels.astype(np.int64)


@register_dataset("synthetic")
def load_synthetic(spec: DatasetSpec, p: int = 2, n: int = 6) -> LabelledTree:
    """A regular p-ary tree; labels are the depth-1 ancestor."""
    tree = build_padic_tree(p, n)
    leaves = _subsample(tree, spec)
    return LabelledTree(tree=tree, leaves=leaves, labels=_labels_at_depth(tree, leaves, 1))


def _longest_path_parents(children_of: dict[str, list[str]], root: str) -> dict[str, str | None]:
    """Keep, for each node, the parent that maximises its depth; ties broken by name."""
    depth: dict[str, int] = {root: 0}
    parent: dict[str, str | None] = {root: None}
    frontier = [root]
    while frontier:
        nxt: list[str] = []
        for u in frontier:
            for v in children_of.get(u, ()):
                cand = depth[u] + 1
                if v not in depth or (cand, u) > (depth[v], parent[v] or ""):
                    depth[v] = cand
                    parent[v] = u
                    nxt.append(v)
        frontier = sorted(set(nxt))
    return parent


@register_dataset("wordnet")
def load_wordnet(spec: DatasetSpec, root_name: str = "entity.n.01") -> LabelledTree:
    """WordNet nouns via NLTK, reduced to a tree by the longest hypernym path."""
    import nltk
    from nltk.corpus import wordnet as wn

    try:
        wn.synset(root_name)
    except LookupError:  # pragma: no cover - one-time download path
        nltk.download("wordnet")
        nltk.download("omw-1.4")

    children_of: dict[str, list[str]] = {}
    stack = [wn.synset(root_name)]
    seen = {root_name}
    while stack:
        syn = stack.pop()
        kids = syn.hyponyms()
        children_of[syn.name()] = sorted(k.name() for k in kids)
        for k in kids:
            if k.name() not in seen:
                seen.add(k.name())
                stack.append(k)
    parent = _longest_path_parents(children_of, root_name)
    tree = pad_to_uniform_depth(from_parent_map(parent))
    leaves = _subsample(tree, spec)
    logger.info("WordNet: %d nodes, height %d", tree.num_nodes, tree.height)
    return LabelledTree(tree=tree, leaves=leaves, labels=_labels_at_depth(tree, leaves, 2))


@register_dataset("go")
def load_go(spec: DatasetSpec, obo_path: Path = Path("data/go-basic.obo"),
            namespace: str = "molecular_function") -> LabelledTree:
    """Gene Ontology from a ``go-basic.obo`` file, restricted to one namespace."""
    if not obo_path.exists():
        raise FileNotFoundError(
            f"{obo_path} not found; fetch it from https://purl.obolibrary.org/obo/go/go-basic.obo"
        )
    children_of: dict[str, list[str]] = {}
    roots: list[str] = []
    term: dict[str, Any] = {}
    for block in obo_path.read_text(encoding="utf-8").split("\n\n"):
        if not block.startswith("[Term]"):
            continue
        term = {"is_a": []}
        for line in block.splitlines()[1:]:
            key, _, val = line.partition(": ")
            if key == "id":
                term["id"] = val.strip()
            elif key == "namespace":
                term["namespace"] = val.strip()
            elif key == "is_a":
                term["is_a"].append(val.split("!")[0].strip())
            elif key == "is_obsolete":
                term["obsolete"] = True
        if term.get("namespace") != namespace or term.get("obsolete"):
            continue
        node = term["id"]
        children_of.setdefault(node, [])
        if not term["is_a"]:
            roots.append(node)
        for par in term["is_a"]:
            children_of.setdefault(par, []).append(node)
    if len(roots) != 1:
        raise ValueError(f"expected one namespace root, found {roots}")
    parent = _longest_path_parents({k: sorted(v) for k, v in children_of.items()}, roots[0])
    tree = pad_to_uniform_depth(from_parent_map(parent))
    leaves = _subsample(tree, spec)
    return LabelledTree(tree=tree, leaves=leaves, labels=_labels_at_depth(tree, leaves, 2))


@register_dataset("ncbi")
def load_ncbi(spec: DatasetSpec, taxdump_path: Path = Path("data/taxdump.tar.gz"),
              clade_taxid: int = 40674) -> LabelledTree:
    """NCBI taxonomy from ``taxdump.tar.gz``, restricted to a clade (default Mammalia)."""
    if not taxdump_path.exists():
        raise FileNotFoundError(
            f"{taxdump_path} not found; fetch it from "
            "https://ftp.ncbi.nlm.nih.gov/pub/taxonomy/taxdump.tar.gz"
        )
    parent_of_all: dict[int, int] = {}
    with tarfile.open(taxdump_path) as tar:
        member = tar.extractfile("nodes.dmp")
        assert member is not None
        for raw in member:
            fields = raw.decode("utf-8").split("\t|\t")
            parent_of_all[int(fields[0])] = int(fields[1])
    children_of_all: dict[int, list[int]] = {}
    for child, par in parent_of_all.items():
        if child != par:
            children_of_all.setdefault(par, []).append(child)
    children_of: dict[str, list[str]] = {}
    stack = [clade_taxid]
    while stack:
        u = stack.pop()
        kids = sorted(children_of_all.get(u, []))
        children_of[str(u)] = [str(k) for k in kids]
        stack.extend(kids)
    parent = _longest_path_parents(children_of, str(clade_taxid))
    tree = pad_to_uniform_depth(from_parent_map(parent))
    leaves = _subsample(tree, spec)
    return LabelledTree(tree=tree, leaves=leaves, labels=_labels_at_depth(tree, leaves, 2))
```

- [ ] **Step 4: Run the fast tests, then the slow one**

Run:
```bash
uv run pytest code/tests/test_datasets.py -v -m "not slow"
uv run python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"
uv run pytest code/tests/test_datasets.py -v -m slow
```
Expected: fast tests PASS; WordNet test PASS after the corpus download.

- [ ] **Step 5: Fetch GO and NCBI, and verify loading**

Run:
```bash
mkdir -p data
curl -L -o data/go-basic.obo https://purl.obolibrary.org/obo/go/go-basic.obo
curl -L -o data/taxdump.tar.gz https://ftp.ncbi.nlm.nih.gov/pub/taxonomy/taxdump.tar.gz
uv run python -c "
from padic_kernel.datasets import DatasetFactory, DatasetSpec
for name in ('go', 'ncbi'):
    ds = DatasetFactory(name)(DatasetSpec(name=name, max_leaves=128, seed=0))
    print(name, ds.tree.num_nodes, ds.tree.height, len(ds.leaves), len(set(ds.labels.tolist())))
"
```
Expected: both print sensible node counts and heights. **If either fails or takes more than 30 minutes to make work, apply the pre-chosen risk response from spec §14: drop to WordNet + synthetic, and add the note to §7 of the paper.**

- [ ] **Step 6: Commit**

```bash
git add code/padic_kernel/datasets.py code/tests/test_datasets.py pyproject.toml
git commit -m "feat(datasets): WordNet, GO, NCBI and synthetic tree loaders with longest-path DAG reduction"
```

---

## Task 9: E1 — ultrametricity audit

**Files:**
- Create: `code/experiments/__init__.py` (empty), `code/experiments/common.py`, `code/experiments/e1_ultrametricity.py`

**Interfaces:**
- Consumes: the whole library
- Produces:
  - `code/experiments/common.py`: `ExperimentConfig` frozen dataclass (`dataset: str`, `max_leaves: int`, `seed: int`, `radix: int`, `depth: int`, `profile_s: float`), `parse_args(description: str) -> ExperimentConfig`, `build_dataset(cfg) -> LabelledTree`, `save_run(out: Path, cfg, results: dict) -> None`, `savefig(fig, out: Path, stem: str) -> None`
  - `e1_ultrametricity.py`: writes `results.json` with one record per encoding: `{"encoding", "dim", "qubits", "violations", "violation_rate", "max_excess", "gromov_delta", "profile_residual", "dimension_lower_bound"}`, plus `e1_violations.pdf`

- [ ] **Step 1: Write `code/experiments/common.py`**

```python
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
from padic_kernel.utils import env_info, make_output_dir, set_seed, write_json  # noqa: E402

logger = logging.getLogger(__name__)

__all__ = ["ExperimentConfig", "build_dataset", "parse_args", "save_run", "savefig"]


@dataclass(frozen=True)
class ExperimentConfig:
    dataset: str = "synthetic"
    max_leaves: int = 128
    seed: int = 42
    radix: int = 2
    depth: int = 6
    profile_s: float = 1.0


def parse_args(description: str) -> ExperimentConfig:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--dataset", default="synthetic",
                        choices=["synthetic", "wordnet", "go", "ncbi"])
    parser.add_argument("--max-leaves", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--radix", type=int, default=2)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--profile-s", type=float, default=1.0)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    return ExperimentConfig(
        dataset=args.dataset, max_leaves=args.max_leaves, seed=args.seed,
        radix=args.radix, depth=args.depth, profile_s=args.profile_s,
    )


def build_dataset(cfg: ExperimentConfig) -> LabelledTree:
    set_seed(cfg.seed)
    spec = DatasetSpec(name=cfg.dataset, max_leaves=cfg.max_leaves, seed=cfg.seed)
    if cfg.dataset == "synthetic":
        return DatasetFactory("synthetic", p=cfg.radix, n=cfg.depth)(spec)
    return DatasetFactory(cfg.dataset)(spec)


def save_run(out: Path, cfg: ExperimentConfig, results: dict[str, Any]) -> None:
    write_json(out / "config.json", {"config": cfg, "env": env_info()})
    write_json(out / "results.json", results)


def savefig(fig: "plt.Figure", out: Path, stem: str) -> None:
    fig.tight_layout()
    fig.savefig(out / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(out / f"{stem}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved figure %s", stem)
```

- [ ] **Step 2: Write `code/experiments/e1_ultrametricity.py`**

```python
"""E1: how far is each encoding's induced distance from being an ultrametric?

Makes Theorems A and B visible: the path-state encoding has exactly zero strong-triangle
violations; every product and every n-qubit encoding has many.
"""

from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np

from experiments.common import build_dataset, parse_args, save_run, savefig
from padic_kernel.encoding import EncodingFactory
from padic_kernel.kernels import fidelity_gram
from padic_kernel.metrics import (
    dimension_lower_bound,
    gromov_delta,
    profile_residual,
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
    height = ds.tree.height
    f = geometric_profile(height, cfg.radix, cfg.profile_s)

    encodings = {
        "path_state": ("path_state", {"profile": f}),
        "angle": ("angle", {"radix": cfg.radix}),
        "basis": ("basis", {"radix": cfg.radix}),
        "zz": ("zz", {"radix": cfg.radix, "reps": 2}),
        "random_product": ("random_product",
                           {"radix": cfg.radix, "local_dim": cfg.radix + 2, "seed": cfg.seed}),
        "block_product": ("block_product",
                          {"radix": cfg.radix, "block_size": 2, "seed": cfg.seed}),
    }

    records = []
    for label, (name, kwargs) in encodings.items():
        psi = EncodingFactory(name, **kwargs).states(ds.tree, ds.leaves)
        k = fidelity_gram(psi)
        dist = 1.0 - k
        count, excess = strong_triangle_violations(dist)
        records.append({
            "encoding": label,
            "dim": int(psi.shape[1]),
            "qubits": float(np.log2(psi.shape[1])),
            "violations": count,
            "violation_rate": violation_rate(dist),
            "max_excess": excess,
            "gromov_delta": gromov_delta(dist),
            "profile_residual": profile_residual(k, lca, f),
        })
        logger.info("%s: %d violations, residual %.3e", label, count, records[-1]["profile_residual"])

    # An RBF kernel on integer leaf coordinates, as the classical reference point.
    coords = ds.leaves.astype(float)[:, None]
    sq = (coords - coords.T) ** 2
    k_rbf = np.exp(-sq / (2.0 * np.median(sq[sq > 0])))
    d_rbf = 1.0 - k_rbf
    count, excess = strong_triangle_violations(d_rbf)
    records.append({
        "encoding": "rbf_integer", "dim": len(ds.leaves), "qubits": float(np.log2(len(ds.leaves))),
        "violations": count, "violation_rate": violation_rate(d_rbf), "max_excess": excess,
        "gromov_delta": gromov_delta(d_rbf), "profile_residual": profile_residual(k_rbf, lca, f),
    })

    results = {
        "records": records,
        "leaves": int(len(ds.leaves)),
        "tree_height": int(height),
        "tree_nodes": int(ds.tree.num_nodes),
        "dimension_lower_bound": dimension_lower_bound(f(lca)),
        "qubit_lower_bound": float(np.log2(dimension_lower_bound(f(lca)))),
    }
    save_run(out, cfg, results)

    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    names = [r["encoding"] for r in records]
    rates = [max(r["violation_rate"], 1e-6) for r in records]
    ax.bar(names, rates, color=["#1b4965" if n == "path_state" else "#9db4c0" for n in names])
    ax.set_yscale("log")
    ax.set_ylabel("strong-triangle violation rate")
    ax.set_xlabel("encoding")
    ax.tick_params(axis="x", rotation=30)
    savefig(fig, out, "e1_violations")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run E1 on synthetic and WordNet**

Run:
```bash
uv run python -m experiments.e1_ultrametricity --dataset synthetic --radix 2 --depth 6 --max-leaves 128
uv run python -m experiments.e1_ultrametricity --dataset wordnet --max-leaves 128
```
(`pythonpath = ["code"]` in `pyproject.toml` already covers `-m experiments....` under `uv run` because `uv run` adds the project root; if the import fails, run `PYTHONPATH=code uv run python -m experiments.e1_ultrametricity ...`.)

Expected: `path_state` reports `violations: 0`, `gromov_delta: 0.0`, `profile_residual < 1e-12`; every other encoding reports a positive violation rate.

- [ ] **Step 4: Commit**

```bash
git add code/experiments
git commit -m "feat(e1): ultrametricity audit across encodings with violation-rate figure"
```

---

## Task 10: E2 — hierarchical classification

**Files:**
- Create: `code/experiments/e2_classification.py`

**Interfaces:**
- Consumes: library + `sklearn.svm.SVC(kernel="precomputed")`
- Produces: `results.json` with per-encoding `{"leaf_accuracy", "root_accuracy", "spearman_rho", "fit_seconds"}` averaged over 5 stratified folds with standard deviations, plus `e2_accuracy.pdf`

- [ ] **Step 1: Write `code/experiments/e2_classification.py`**

```python
"""E2: hierarchical classification with precomputed quantum kernels.

Reported as parity evidence only. The headline claim of the paper is exactness (E1)
and does not depend on this experiment; see the paper's success criteria.
"""

from __future__ import annotations

import logging
import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC

from experiments.common import build_dataset, parse_args, save_run, savefig
from padic_kernel.encoding import EncodingFactory
from padic_kernel.kernels import fidelity_gram
from padic_kernel.profiles import geometric_profile
from padic_kernel.tree import ancestor_matrix, lca_depth_matrix
from padic_kernel.utils import make_output_dir

logger = logging.getLogger(__name__)


def _cross_validated_accuracy(kernel: np.ndarray, labels: np.ndarray, seed: int) -> tuple[float, float]:
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
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
    lca = lca_depth_matrix(anc)
    height = ds.tree.height
    f = geometric_profile(height, cfg.radix, cfg.profile_s)
    true_dist = (height - lca).astype(float)

    root_labels = np.unique(anc[:, 1], return_inverse=True)[1]

    encodings = {
        "path_state": ("path_state", {"profile": f}),
        "angle": ("angle", {"radix": cfg.radix}),
        "basis": ("basis", {"radix": cfg.radix}),
        "zz": ("zz", {"radix": cfg.radix, "reps": 2}),
    }

    records = []
    for label, (name, kwargs) in encodings.items():
        start = time.perf_counter()
        k = fidelity_gram(EncodingFactory(name, **kwargs).states(ds.tree, ds.leaves))
        leaf_mean, leaf_std = _cross_validated_accuracy(k, ds.labels, cfg.seed)
        root_mean, root_std = _cross_validated_accuracy(k, root_labels, cfg.seed)
        iu = np.triu_indices(k.shape[0], k=1)
        rho = float(spearmanr(1.0 - k[iu], true_dist[iu]).statistic)
        records.append({
            "encoding": label, "leaf_accuracy": leaf_mean, "leaf_accuracy_std": leaf_std,
            "root_accuracy": root_mean, "root_accuracy_std": root_std,
            "spearman_rho": rho, "fit_seconds": time.perf_counter() - start,
        })
        logger.info("%s: leaf %.3f root %.3f rho %.3f", label, leaf_mean, root_mean, rho)

    # Classical baselines on the same folds.
    coords = ds.leaves.astype(float)[:, None]
    sq = (coords - coords.T) ** 2
    for gamma_name, k_cls in (
        ("rbf_integer", np.exp(-sq / (2.0 * np.median(sq[sq > 0])))),
        ("rbf_onehot", np.exp(-(1.0 - np.eye(len(ds.leaves))))),
    ):
        leaf_mean, leaf_std = _cross_validated_accuracy(k_cls, ds.labels, cfg.seed)
        root_mean, root_std = _cross_validated_accuracy(k_cls, root_labels, cfg.seed)
        iu = np.triu_indices(k_cls.shape[0], k=1)
        records.append({
            "encoding": gamma_name, "leaf_accuracy": leaf_mean, "leaf_accuracy_std": leaf_std,
            "root_accuracy": root_mean, "root_accuracy_std": root_std,
            "spearman_rho": float(spearmanr(1.0 - k_cls[iu], true_dist[iu]).statistic),
            "fit_seconds": 0.0,
        })

    save_run(out, cfg, {"records": records, "leaves": int(len(ds.leaves)),
                        "num_classes": int(len(np.unique(ds.labels)))})

    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    names = [r["encoding"] for r in records]
    means = [r["leaf_accuracy"] for r in records]
    errs = [r["leaf_accuracy_std"] for r in records]
    ax.bar(names, means, yerr=errs, capsize=3,
           color=["#1b4965" if n == "path_state" else "#9db4c0" for n in names])
    ax.set_ylabel("leaf accuracy (5-fold)")
    ax.tick_params(axis="x", rotation=30)
    savefig(fig, out, "e2_accuracy")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run E2 on all available datasets**

Run:
```bash
for d in synthetic wordnet go ncbi; do
  uv run python -m experiments.e2_classification --dataset "$d" --max-leaves 256 || echo "SKIP $d"
done
```
Expected: each run writes an output directory. Record the numbers as they come out. **Do not tune `C` or the profile to improve accuracy** — the success criterion in the spec is fixed in advance, and E2 is parity evidence, not the claim.

- [ ] **Step 3: Commit**

```bash
git add code/experiments/e2_classification.py
git commit -m "feat(e2): hierarchical classification with precomputed kernels and classical baselines"
```

---

## Task 11: E3 — ablations

**Files:**
- Create: `code/experiments/e3_ablations.py`

**Interfaces:**
- Consumes: library
- Produces: `results.json` with a grid over `profile ∈ {geometric s=0.5, geometric s=1, geometric s=2, linear}` × `depth ∈ {2..6}` × `radix ∈ {2, 3, 5}`, each entry `{"violations", "profile_residual", "leaf_accuracy", "dimension_lower_bound", "qubits_used"}`, plus `e3_ablation.pdf` (a 1×3 panel: residual vs depth, bound vs radix, accuracy vs `s`)

- [ ] **Step 1: Write `code/experiments/e3_ablations.py`**

```python
"""E3: which design choices matter — profile shape, tree depth, branching factor."""

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
    profile_residual,
    strong_triangle_violations,
)
from padic_kernel.profiles import geometric_profile, linear_profile
from padic_kernel.tree import ancestor_matrix, lca_depth_matrix
from padic_kernel.utils import make_output_dir, set_seed

logger = logging.getLogger(__name__)

RADICES = (2, 3, 5)
DEPTHS = (2, 3, 4, 5, 6)
PROFILES = ("geometric_0.5", "geometric_1.0", "geometric_2.0", "linear")


def _profile(name: str, n: int, p: int):
    if name == "linear":
        return linear_profile(n)
    return geometric_profile(n, p, float(name.split("_")[1]))


def _accuracy(kernel: np.ndarray, labels: np.ndarray, seed: int) -> float:
    if len(np.unique(labels)) < 2 or np.min(np.bincount(labels)) < 5:
        return float("nan")
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
            DatasetSpec(name="synthetic", max_leaves=min(256, p**n), seed=cfg.seed)
        )
        lca = lca_depth_matrix(ancestor_matrix(ds.tree, ds.leaves))
        f = _profile(pname, n, p)
        k = fidelity_gram(EncodingFactory("path_state", profile=f).states(ds.tree, ds.leaves))
        count, _ = strong_triangle_violations(1.0 - k)
        records.append({
            "radix": p, "depth": n, "profile": pname,
            "violations": count,
            "profile_residual": profile_residual(k, lca, f),
            "leaf_accuracy": _accuracy(k, ds.labels, cfg.seed),
            "dimension_lower_bound": dimension_lower_bound(f(lca)),
            "qubits_used": float(np.log2(ds.tree.num_nodes)),
            "qubit_lower_bound": float(np.log2(dimension_lower_bound(f(lca)))),
        })
        logger.info("p=%d n=%d %s -> %d violations", p, n, pname, count)

    save_run(out, cfg, {"records": records})

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.0))
    for pname in PROFILES:
        sel = [r for r in records if r["profile"] == pname and r["radix"] == 2]
        axes[0].semilogy([r["depth"] for r in sel],
                         [max(r["profile_residual"], 1e-18) for r in sel], marker="o", label=pname)
    axes[0].set_xlabel("tree depth n"); axes[0].set_ylabel("max |K - f(lambda)|")
    axes[0].legend(fontsize=7)

    for p in RADICES:
        sel = [r for r in records if r["radix"] == p and r["profile"] == "geometric_2.0"]
        axes[1].plot([r["depth"] for r in sel], [r["qubit_lower_bound"] for r in sel],
                     marker="s", label=f"bound, p={p}")
        axes[1].plot([r["depth"] for r in sel], [r["depth"] for r in sel],
                     linestyle=":", label=f"n qubits, p={p}")
    axes[1].set_xlabel("tree depth n"); axes[1].set_ylabel("qubits")
    axes[1].legend(fontsize=6)

    for pname in PROFILES:
        sel = [r for r in records if r["profile"] == pname and r["radix"] == 3]
        axes[2].plot([r["depth"] for r in sel], [r["leaf_accuracy"] for r in sel],
                     marker="^", label=pname)
    axes[2].set_xlabel("tree depth n"); axes[2].set_ylabel("leaf accuracy")
    axes[2].legend(fontsize=7)
    savefig(fig, out, "e3_ablation")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run E3**

Run: `uv run python -m experiments.e3_ablations`
Expected: residuals stay below `1e-12` for every strictly monotone profile at every depth, and the middle panel shows the Theorem B bound crossing above `n` for `p ≥ 3`. That crossing is the figure that carries Theorem B in the paper.

- [ ] **Step 3: Commit**

```bash
git add code/experiments/e3_ablations.py
git commit -m "feat(e3): profile, depth and radix ablations with the Theorem B crossing figure"
```

---

## Task 12: E4 — entangling extension and the distortion/expressivity front

**Files:**
- Create: `code/padic_kernel/reupload.py`, `code/experiments/e4_entangling.py`
- Test: append to `code/tests/test_encoding.py`

**Interfaces:**
- Consumes: `padic_kernel.encoding`, `padic_kernel.kernels`
- Produces:
  - `reupload.py`: `ReuploadEncoding(base: Encoding, layers: int, theta: np.ndarray, seed: int)` implementing `Φ^{(L)}(x) = W_L V(x) ⋯ W_1 V(x)|0⟩`, where `V(x)` is implemented as the rank-1 replacement `|ψ⟩ ↦ ⟨base(x)|ψ⟩ · base(x)` renormalised — no, see below — and `random_entangler(dim: int, rng) -> np.ndarray` returning a Haar unitary
  - `distortion(kernel: np.ndarray) -> float` = `max(0, max excess)` over triples of `1 − K`

  **Data re-uploading semantics, fixed here so it is unambiguous:** `V_f(x)` is the *state-preparation isometry* for `Φ_f(x)`; composing it with anything on the left discards the previous state. To get a non-trivial `L`-layer map we instead interleave **diagonal data-dependent phases** with fixed entanglers:
  `Φ^{(L)}(x) = W_L D(x) ⋯ W_1 D(x) Φ_f(x)`, with `D(x) = diag(exp(i γ · 1[u is an ancestor of x]))` over nodes `u`. At `γ = 0` and `W_ℓ = I` this reduces exactly to `Φ_f(x)`, so the exact construction sits at the origin of the family and the Pareto front is anchored.

- [ ] **Step 1: Write the failing tests (append to `code/tests/test_encoding.py`)**

```python
def test_reupload_reduces_to_path_state_at_zero_coupling():
    from padic_kernel.kernels import fidelity_gram
    from padic_kernel.reupload import ReuploadEncoding

    tree = build_padic_tree(2, 3)
    f = geometric_profile(3, 2, 1.0)
    base = EncodingFactory("path_state", profile=f)
    enc = ReuploadEncoding(base=base, layers=3, gamma=0.0, entangle=0.0, seed=0)
    assert np.allclose(
        fidelity_gram(enc.states(tree, tree.leaves)),
        fidelity_gram(base.states(tree, tree.leaves)),
        atol=1e-12,
    )


def test_reupload_distortion_grows_with_gamma():
    from padic_kernel.kernels import fidelity_gram
    from padic_kernel.metrics import strong_triangle_violations
    from padic_kernel.reupload import ReuploadEncoding

    tree = build_padic_tree(2, 3)
    f = geometric_profile(3, 2, 1.0)
    base = EncodingFactory("path_state", profile=f)
    excesses = []
    for gamma in (0.0, 0.3, 0.9):
        enc = ReuploadEncoding(base=base, layers=2, gamma=gamma, entangle=0.5, seed=0)
        _, excess = strong_triangle_violations(1.0 - fidelity_gram(enc.states(tree, tree.leaves)))
        excesses.append(max(excess, 0.0))
    assert excesses[0] <= 1e-12 < excesses[1] < excesses[2]
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest code/tests/test_encoding.py -k reupload -v`
Expected: FAIL — `No module named 'padic_kernel.reupload'`.

- [ ] **Step 3: Implement `code/padic_kernel/reupload.py`**

```python
"""Data re-uploading on top of the exact path-state encoding.

The exact construction sits at ``gamma = 0, entangle = 0``; increasing either knob
trades ultrametricity for expressivity. This is the tunable family whose Pareto front
E4 measures.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from padic_kernel.encoding import Encoding
from padic_kernel.tree import RootedTree, ancestor_matrix

logger = logging.getLogger(__name__)

__all__ = ["ReuploadEncoding", "haar_unitary"]


def haar_unitary(dim: int, rng: np.random.Generator) -> np.ndarray:
    """A Haar-random unitary of size ``dim`` via QR of a complex Ginibre matrix."""
    a = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
    q, r = np.linalg.qr(a)
    return q * (np.diag(r) / np.abs(np.diag(r)))[None, :]


@dataclass
class ReuploadEncoding(Encoding):
    """``Phi(x) = W_L D(x) ... W_1 D(x) Phi_base(x)``.

    ``D(x)`` applies phase ``gamma`` to every node on the root-to-leaf path of ``x``,
    which is the natural data-dependent diagonal on the path-state Hilbert space.
    ``entangle`` interpolates each ``W_l`` between the identity and a Haar unitary.
    """

    base: Encoding
    layers: int = 1
    gamma: float = 0.0
    entangle: float = 0.0
    seed: int = 0
    name: str = field(default="reupload", init=False)

    def _mixers(self, dim: int) -> list[np.ndarray]:
        rng = np.random.default_rng(self.seed)
        mixers = []
        for _ in range(self.layers):
            u = haar_unitary(dim, rng)
            m = (1.0 - self.entangle) * np.eye(dim) + self.entangle * u
            # Re-unitarise the interpolant so the map stays norm-preserving.
            q, r = np.linalg.qr(m)
            mixers.append(q * (np.diag(r) / np.abs(np.diag(r)))[None, :])
        return mixers

    def states(self, tree: RootedTree, leaves: np.ndarray) -> np.ndarray:
        psi = self.base.states(tree, leaves)
        dim = psi.shape[1]
        anc = ancestor_matrix(tree, leaves)
        on_path = np.zeros((psi.shape[0], dim), dtype=bool)
        rows = np.arange(psi.shape[0])[:, None]
        on_path[rows, anc] = True
        phases = np.exp(1j * self.gamma * on_path)
        for mixer in self._mixers(dim):
            psi = (psi * phases) @ mixer.T
        return psi / np.linalg.norm(psi, axis=1, keepdims=True)
```

- [ ] **Step 4: Run to verify the tests pass**

Run: `uv run pytest code/tests/test_encoding.py -k reupload -v`
Expected: both PASS.

- [ ] **Step 5: Write `code/experiments/e4_entangling.py`**

```python
"""E4: the expressivity / ultrametricity trade-off of the re-uploading family."""

from __future__ import annotations

import itertools
import logging

import matplotlib.pyplot as plt
import numpy as np

from experiments.common import build_dataset, parse_args, save_run, savefig
from padic_kernel.encoding import EncodingFactory
from padic_kernel.kernels import fidelity_gram
from padic_kernel.metrics import kernel_target_alignment, strong_triangle_violations
from padic_kernel.profiles import geometric_profile
from padic_kernel.reupload import ReuploadEncoding
from padic_kernel.utils import make_output_dir

logger = logging.getLogger(__name__)

GAMMAS = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0)
ENTANGLES = (0.0, 0.25, 0.5, 1.0)
LAYERS = (1, 2, 4)


def main() -> None:
    cfg = parse_args(__doc__ or "E4")
    out = make_output_dir("e4_entangling")
    ds = build_dataset(cfg)
    f = geometric_profile(ds.tree.height, cfg.radix, cfg.profile_s)
    base = EncodingFactory("path_state", profile=f)

    records = []
    for gamma, entangle, layers in itertools.product(GAMMAS, ENTANGLES, LAYERS):
        enc = ReuploadEncoding(base=base, layers=layers, gamma=gamma,
                               entangle=entangle, seed=cfg.seed)
        k = fidelity_gram(enc.states(ds.tree, ds.leaves))
        _, excess = strong_triangle_violations(1.0 - k)
        records.append({
            "gamma": gamma, "entangle": entangle, "layers": layers,
            "distortion": max(float(excess), 0.0),
            "alignment": kernel_target_alignment(k, ds.labels),
        })
    logger.info("Collected %d re-uploading configurations", len(records))
    save_run(out, cfg, {"records": records})

    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    for layers in LAYERS:
        sel = sorted((r for r in records if r["layers"] == layers), key=lambda r: r["distortion"])
        ax.scatter([r["distortion"] for r in sel], [r["alignment"] for r in sel],
                   s=18, label=f"L = {layers}")
    exact = [r for r in records if r["gamma"] == 0.0 and r["entangle"] == 0.0][0]
    ax.scatter([exact["distortion"]], [exact["alignment"]], marker="*", s=180,
               color="#1b4965", label="exact path state", zorder=5)
    ax.set_xlabel("ultrametric distortion D")
    ax.set_ylabel("kernel-target alignment")
    ax.legend(fontsize=7)
    savefig(fig, out, "e4_pareto")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Run E4**

Run: `uv run python -m experiments.e4_entangling --dataset synthetic --radix 2 --depth 5 --max-leaves 96`
Expected: the exact point sits at distortion 0; distortion increases continuously with `gamma`. **If the front is flat or structureless, apply the spec's pre-chosen response: §6 shrinks to a short remark and C5 is dropped from the contribution list.**

- [ ] **Step 7: Commit**

```bash
git add code/padic_kernel/reupload.py code/experiments/e4_entangling.py code/tests/test_encoding.py
git commit -m "feat(e4): data-reuploading family and the distortion/alignment Pareto front"
```

---

## Task 13: E5 — noise and finite shots

**Files:**
- Create: `code/experiments/e5_noise.py`

**Interfaces:**
- Consumes: `padic_kernel.kernels.depolarise`, `sample_kernel`
- Produces: `results.json` with a grid over `depolarising rate ∈ {0, 1e-3, 1e-2, 5e-2, 1e-1}` × `shots ∈ {10², 10³, 10⁴, 10⁵, ∞}` × 10 repeats, each `{"violations", "violation_rate", "profile_residual"}` with mean and std, plus `e5_noise.pdf`, and the derived **shot budget**: the smallest shot count keeping the mean violation rate below `1e-3`

- [ ] **Step 1: Write `code/experiments/e5_noise.py`**

```python
"""E5: does exact ultrametricity survive depolarising noise and finite sampling?"""

from __future__ import annotations

import itertools
import logging

import matplotlib.pyplot as plt
import numpy as np

from experiments.common import build_dataset, parse_args, save_run, savefig
from padic_kernel.encoding import EncodingFactory
from padic_kernel.kernels import depolarise, fidelity_gram, sample_kernel
from padic_kernel.metrics import profile_residual, violation_rate
from padic_kernel.profiles import geometric_profile
from padic_kernel.tree import ancestor_matrix, lca_depth_matrix
from padic_kernel.utils import make_output_dir

logger = logging.getLogger(__name__)

RATES = (0.0, 1e-3, 1e-2, 5e-2, 1e-1)
SHOTS = (100, 1_000, 10_000, 100_000, None)
REPEATS = 10
TARGET_RATE = 1e-3


def main() -> None:
    cfg = parse_args(__doc__ or "E5")
    out = make_output_dir("e5_noise")
    ds = build_dataset(cfg)
    lca = lca_depth_matrix(ancestor_matrix(ds.tree, ds.leaves))
    f = geometric_profile(ds.tree.height, cfg.radix, cfg.profile_s)
    exact = fidelity_gram(
        EncodingFactory("path_state", profile=f).states(ds.tree, ds.leaves)
    )

    records = []
    for rate, shots in itertools.product(RATES, SHOTS):
        rates_seen, residuals = [], []
        for rep in range(REPEATS if shots is not None else 1):
            rng = np.random.default_rng(cfg.seed + rep)
            k = depolarise(exact, rate, dim=ds.tree.num_nodes)
            if shots is not None:
                k = sample_kernel(k, shots=shots, rng=rng)
            rates_seen.append(violation_rate(1.0 - k))
            residuals.append(profile_residual(k, lca, f))
        records.append({
            "depolarising_rate": rate,
            "shots": shots if shots is not None else "exact",
            "violation_rate_mean": float(np.mean(rates_seen)),
            "violation_rate_std": float(np.std(rates_seen)),
            "profile_residual_mean": float(np.mean(residuals)),
        })
        logger.info("rate=%.3g shots=%s -> violation rate %.3g",
                    rate, shots, records[-1]["violation_rate_mean"])

    noiseless = [r for r in records if r["depolarising_rate"] == 0.0
                 and r["shots"] != "exact" and r["violation_rate_mean"] < TARGET_RATE]
    budget = min((int(r["shots"]) for r in noiseless), default=None)

    save_run(out, cfg, {"records": records, "target_violation_rate": TARGET_RATE,
                        "shot_budget": budget})

    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    for rate in RATES:
        sel = [r for r in records if r["depolarising_rate"] == rate and r["shots"] != "exact"]
        ax.loglog([int(r["shots"]) for r in sel],
                  [max(r["violation_rate_mean"], 1e-8) for r in sel],
                  marker="o", label=f"p_dep = {rate:g}")
    ax.axhline(TARGET_RATE, linestyle="--", color="grey", linewidth=0.8)
    ax.set_xlabel("shots per kernel entry")
    ax.set_ylabel("strong-triangle violation rate")
    ax.legend(fontsize=7)
    savefig(fig, out, "e5_noise")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run E5**

Run: `uv run python -m experiments.e5_noise --dataset synthetic --radix 2 --depth 5 --max-leaves 64`
Expected: the noiseless-exact row has violation rate 0; sampled rows decay towards it as shots grow; the reported `shot_budget` is a concrete integer for the paper.

- [ ] **Step 3: Commit**

```bash
git add code/experiments/e5_noise.py
git commit -m "feat(e5): depolarising-noise and finite-shot robustness with a derived shot budget"
```

---

## Task 14: Numbers extraction — the paper cannot drift from the code

**Files:**
- Create: `code/experiments/collect_numbers.py`
- Create (generated): `papers/aiqxqia2026/numbers.tex`

**Interfaces:**
- Consumes: the newest `outputs/e{1..5}_*/results.json`
- Produces: `\newcommand` definitions the manuscript uses instead of hard-coded numerals — e.g. `\PathResidual`, `\AngleViolationRate`, `\ShotBudget`, `\WordNetQubits`, `\QubitLowerBoundThree`

- [ ] **Step 1: Write `code/experiments/collect_numbers.py`**

```python
"""Turn the newest experiment outputs into LaTeX macros for the manuscript.

The paper never hard-codes a numeral that came from an experiment: it uses the macros
this script writes, so a rerun that changes a number changes the paper too.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _newest(root: Path, prefix: str) -> dict[str, Any]:
    candidates = sorted(root.glob(f"{prefix}_*/results.json"))
    if not candidates:
        raise FileNotFoundError(f"no results for {prefix} under {root}")
    return json.loads(candidates[-1].read_text(encoding="utf-8"))


def _fmt(value: float, digits: int = 3) -> str:
    if value == 0:
        return "0"
    if abs(value) < 1e-3 or abs(value) >= 1e4:
        mantissa, exponent = f"{value:.{digits}e}".split("e")
        return rf"{mantissa}\times 10^{{{int(exponent)}}}"
    return f"{value:.{digits}g}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs", type=Path, default=Path("outputs"))
    parser.add_argument("--target", type=Path,
                        default=Path("papers/aiqxqia2026/numbers.tex"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    lines = ["% Generated by code/experiments/collect_numbers.py -- do not edit by hand."]

    e1 = _newest(args.outputs, "e1_ultrametricity")
    by_name = {r["encoding"]: r for r in e1["records"]}
    lines += [
        rf"\newcommand{{\PathResidual}}{{{_fmt(by_name['path_state']['profile_residual'])}}}",
        rf"\newcommand{{\PathViolations}}{{{by_name['path_state']['violations']}}}",
        rf"\newcommand{{\AngleViolationRate}}{{{_fmt(by_name['angle']['violation_rate'])}}}",
        rf"\newcommand{{\ZZViolationRate}}{{{_fmt(by_name['zz']['violation_rate'])}}}",
        rf"\newcommand{{\RBFViolations}}{{{by_name['rbf_integer']['violations']}}}",
        rf"\newcommand{{\EOneLeaves}}{{{e1['leaves']}}}",
        rf"\newcommand{{\EOneQubitBound}}{{{_fmt(e1['qubit_lower_bound'])}}}",
    ]

    e5 = _newest(args.outputs, "e5_noise")
    lines.append(rf"\newcommand{{\ShotBudget}}{{{e5['shot_budget']}}}")

    args.target.parent.mkdir(parents=True, exist_ok=True)
    args.target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Wrote %d macros to %s", len(lines) - 1, args.target)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it**

Run: `uv run python -m experiments.collect_numbers`
Expected: `papers/aiqxqia2026/numbers.tex` appears with the macros defined. Extend the script whenever the manuscript needs another experimental numeral — never type one by hand.

- [ ] **Step 3: Commit**

```bash
git add code/experiments/collect_numbers.py papers/aiqxqia2026/numbers.tex
git commit -m "feat(paper): generate LaTeX numeric macros from experiment outputs"
```

---

## Task 15: CEUR manuscript scaffold that compiles

**Files:**
- Create: `papers/aiqxqia2026/ceurart.cls`, `papers/aiqxqia2026/main.tex`, `papers/aiqxqia2026/refs.bib`, `papers/aiqxqia2026/Makefile`, `papers/aiqxqia2026/figures/.gitkeep`

**Interfaces:**
- Consumes: `numbers.tex` from Task 14
- Produces: a `main.pdf` that builds with zero errors

- [ ] **Step 1: Fetch the class file and its companions**

Run:
```bash
mkdir -p papers/aiqxqia2026/figures
cd papers/aiqxqia2026
curl -L -o ceurart.cls \
  https://raw.githubusercontent.com/yamadharma/ceurart/master/tex/latex/ceurart/ceurart.cls
grep -c . ceurart.cls
```
Expected: a non-empty class file. If the class pulls in a `.sty` or `.bst` that TeX Live 2023 lacks, fetch it from the same `tex/latex/ceurart/` directory. The bibliography style CEUR expects is included with the class; if `\bibliographystyle` errors, fall back to `plainnat` and note it.

- [ ] **Step 2: Write `papers/aiqxqia2026/main.tex`**

```latex
\documentclass[twocolumn]{ceurart}
% CEUR-WS single-blind submission for AIQxQIA 2026.

\usepackage{amsmath,amssymb,amsthm}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage{microtype}
\usepackage[colorlinks=true,allcolors=blue]{hyperref}
\usepackage{todonotes}
\setuptodonotes{inline,color=yellow!40}

\input{numbers}

% --- notation macros (see .claude/skills/latex-conventions) ---
\newcommand{\Zp}{\mathbb{Z}_p}
\newcommand{\Zpn}{\mathbb{Z}/p^{n}}
\newcommand{\C}{\mathbb{C}}
\newcommand{\R}{\mathbb{R}}
\newcommand{\vp}{v_p}
\newcommand{\lcadepth}{\lambda}
\newcommand{\PathState}[1]{\Phi_f(#1)}
\newcommand{\fid}[2]{\lvert\langle #1 \mid #2 \rangle\rvert^{2}}
\DeclareMathOperator{\anc}{anc}
\DeclareMathOperator{\rank}{rank}
\DeclareMathOperator{\tr}{tr}

\theoremstyle{plain}
\newtheorem{theorem}{Theorem}
\newtheorem{proposition}[theorem]{Proposition}
\newtheorem{corollary}[theorem]{Corollary}
\newtheorem{lemma}[theorem]{Lemma}
\theoremstyle{definition}
\newtheorem{definition}[theorem]{Definition}
\newtheorem{remark}[theorem]{Remark}

\begin{document}

\copyrightyear{2026}
\copyrightclause{Copyright for this paper by its author.
  Use permitted under Creative Commons License Attribution 4.0
  International (CC BY 4.0).}

\conference{AIQxQIA 2026: 4th International Workshop on AI for Quantum and
  Quantum for AI, 2026}

\title{Ultrametric Quantum Kernels: Exact $p$-adic Feature Maps for Hierarchical Data}

\author[1]{TODO-AUTHOR}[
  orcid=0000-0000-0000-0000,
  email=todo@example.org,
]
\address[1]{TODO-AFFILIATION}
\todo{Author block, ORCID, affiliation and email pending -- must be resolved before upload.}

\begin{abstract}
\todo{Abstract written last, after the results are final.}
\end{abstract}

\begin{keywords}
  quantum machine learning \sep
  quantum kernels \sep
  ultrametric spaces \sep
  $p$-adic numbers \sep
  hierarchical representation learning
\end{keywords}

\maketitle

\input{sections/01-introduction}
\input{sections/02-preliminaries}
\input{sections/03-problem}
\input{sections/04-theorem-a}
\input{sections/05-theorem-b}
\input{sections/06-theorem-c}
\input{sections/07-reuploading}
\input{sections/08-experiments}
\input{sections/09-related-work}
\input{sections/10-limitations}
\input{sections/11-conclusion}

\section*{Declaration on Generative AI}
\todo{Complete per the CEUR-WS generative-AI disclosure policy in force at submission.}

\bibliographystyle{plainnat}
\bibliography{refs}

\end{document}
```

Create `papers/aiqxqia2026/sections/` with eleven files, each initially containing only its `\section{...}` line and a `\todo{draft pending}`, so the document compiles from day one.

- [ ] **Step 3: Write `papers/aiqxqia2026/Makefile`**

```makefile
.PHONY: all clean watch
all: main.pdf

main.pdf: main.tex refs.bib numbers.tex $(wildcard sections/*.tex) $(wildcard figures/*.pdf)
	latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

clean:
	latexmk -C
	rm -f *.bbl *.blg

watch:
	latexmk -pdf -pvc main.tex
```

- [ ] **Step 4: Verify the build**

Run:
```bash
cd papers/aiqxqia2026 && make clean && make
pdfinfo main.pdf | grep Pages
```
Expected: `main.pdf` builds with no errors. Record the page count; it will grow as sections are written.

- [ ] **Step 5: Commit**

```bash
git add papers/aiqxqia2026
git commit -m "docs(paper): CEUR ceurart scaffold that compiles with section stubs"
```

---

## Task 16: References

**Files:**
- Modify: `papers/aiqxqia2026/refs.bib`

**Interfaces:**
- Consumes: nothing
- Produces: a verified bibliography

Delegate the search to the `literature-scout` agent and the BibTeX formatting to the `reference-manager` agent, per the workspace conventions in `CLAUDE.md`. Cite-key format follows `.claude/skills/bibtex-hygiene/`.

- [ ] **Step 1: Collect the required entries**

Required, and each must be verified against a real record before it is cited:

| Topic | What to find | Notes |
|-------|--------------|-------|
| v-PuNNs | N'guessan, van der Put neural networks, arXiv:2508.01010 | The benchmark source; confirm the current version and any published venue |
| p-adic quantum mechanics | Aniello et al., *Symmetry* 2023, p-adic qubit / quNit | Confirm volume, issue, article number, DOI |
| Quantum walks on p-adic trees | Zúñiga-Galindo, arXiv:2508.06712 | Confirm title and whether a journal version exists by August 2026 |
| Quantum kernel methods | Havlíček et al. 2019 and Schuld's kernel-perspective paper | Foundational; keep to two |
| Recent quantum-kernel theory | 2024–2026 results on expressivity, exponential concentration, inductive bias | At least four post-2023 entries — the user requires recent references |
| Hyperbolic embeddings | Nickel & Kiela Poincaré embeddings, plus a 2024–2026 hyperbolic-ML survey | |
| Ultrametrics in ML | A recent survey or textbook chapter on ultrametric / hierarchical clustering | |
| Data re-uploading | Pérez-Salinas et al., plus a recent follow-up | |
| CEUR / venue | Not cited | |

- [ ] **Step 2: Dispatch the agents**

```
Use the literature-scout agent to find and summarise, with verifiable identifiers,
the most recent (2023-2026) work on: quantum kernel expressivity and inductive bias;
p-adic and ultrametric methods in machine learning; hyperbolic representation
learning; and data re-uploading. Return arXiv IDs or DOIs for everything.

Then use the reference-manager agent to turn that list into refs.bib entries with
cite keys following .claude/skills/bibtex-hygiene, deduplicated, with no invented
fields.
```

- [ ] **Step 3: Verify every entry resolves**

Run:
```bash
uv run python - <<'PY'
import re, pathlib, urllib.request
text = pathlib.Path("papers/aiqxqia2026/refs.bib").read_text()
ids = re.findall(r"(?:arXiv:|eprint\s*=\s*[{\"])\s*([0-9]{4}\.[0-9]{4,5})", text)
dois = re.findall(r"doi\s*=\s*[{\"]([^}\"]+)", text)
for a in sorted(set(ids)):
    url = f"https://arxiv.org/abs/{a}"
    code = urllib.request.urlopen(url).getcode()
    print(a, code)
for d in sorted(set(dois)):
    url = f"https://doi.org/{d}"
    code = urllib.request.urlopen(urllib.request.Request(url, method="HEAD")).getcode()
    print(d, code)
PY
```
Expected: every identifier returns 200. Anything that does not is removed from `refs.bib` and the corresponding claim in the paper is either re-sourced or marked `\todo{unverified}`.

- [ ] **Step 4: Commit**

```bash
git add papers/aiqxqia2026/refs.bib
git commit -m "docs(refs): verified bibliography, 2023-2026 sources prioritised"
```

---

## Task 17: Draft the mathematical core (§§1–6)

**Files:**
- Modify: `papers/aiqxqia2026/sections/01-introduction.tex` … `06-theorem-c.tex`

**Interfaces:**
- Consumes: `numbers.tex`, `refs.bib`
- Produces: sections 1–6 of the manuscript

Write in the register described in Global Constraints. Delegate proof drafting to the `proof-writer` agent and audit each finished proof with the `proof-verifier` agent before moving on.

- [ ] **Step 1: §1 Introduction (target 1.5 pp)**

Content, in order: hierarchy is ubiquitous in the data QML is applied to; the geometry a quantum kernel induces is fixed by the encoding before any training; the question "which feature maps induce an exactly ultrametric kernel?"; the three results in one sentence each; **the honest-scope paragraph stating plainly that `K(x,y) = f(λ(x,y))` is classically computable in `O(depth)` and that no quantum speedup in kernel evaluation is claimed**; contributions C1–C5 as a bulleted list; a code-availability sentence.

- [ ] **Step 2: §2 Preliminaries (1.5 pp)**

`ℤ_p` and the p-adic valuation; the correspondence between `v_p(x−y)` and LCA depth in the regular p-ary tree, with the small worked example `p = 3, n = 2`; ultrametric spaces and the strong triangle inequality; quantum feature maps and fidelity kernels; the statement that a fidelity kernel is PSD.

- [ ] **Step 3: §3 Problem statement (0.75 pp)**

Definition of an *ultrametric kernel with profile f* and of a *strictly monotone* profile; Definition of a product feature map and of a block-product feature map with block size `k`; the observation that `d = 1 − K` is an ultrametric exactly when `f` is monotone.

- [ ] **Step 4: §4 Theorem A with the block generalisation (1.5 pp)**

Statement, proof (the multiplicativity argument, generalised to blocks), and the corollaries for angle encoding, computational-basis encoding, and first-order Pauli-Z feature maps. **State explicitly that IQP/ZZ maps are entangling and therefore not covered by this theorem** — and forward-reference §5, which does cover them. Cite the verification tests by name: `test_theorem_a_random_product_maps_are_never_strictly_monotone`, `test_theorem_a_block_generalisation_bounds_resolution_depth`.

- [ ] **Step 5: §5 Theorem B, the dimension lower bound (1.25 pp)**

Statement:

> Let `Φ : X → ℂ^D` map `L` points to unit vectors with `|⟨Φ(x)|Φ(y)⟩|² = K(x,y)`. Then
> `D ≥ L² / Σ_{x,y} K(x,y)`.

Proof: the Gram matrix `G` of the `Φ(x)` is PSD with unit diagonal, so `tr G = L`; for a PSD matrix of rank `r`, Cauchy–Schwarz on the eigenvalues gives `‖G‖_F² ≥ (tr G)²/r`; and `‖G‖_F² = Σ_{x,y}|G_{xy}|² = Σ_{x,y} K(x,y)`; finally `D ≥ rank(G)`.

Corollary for `ℤ/pⁿ` with `f(v) = p^{-(n-v)s}`, `s > 1`: `Σ_y f(λ(x,y)) = S(p,s,n) := 1 + ((p−1)/p) Σ_{m=1}^{n} p^{m(1−s)}`, bounded above by a constant `S(p,s)`, so `D ≥ pⁿ / S(p,s)` and the qubit count satisfies `q ≥ n log₂ p − log₂ S(p,s)`. For every `p ≥ 3` this exceeds `n` once `n log₂(p/2) > log₂ S(p,s)`; **hence no `n`-qubit encoding of any kind — IQP, ZZ, hardware-efficient, or otherwise — realises a strictly monotone ultrametric kernel on `ℤ/pⁿ`.** Give the worked constant for `p = 3, s = 2`: `S = 4/3`, so `q ≥ 1.585n − 0.415`.

Remark on tightness: the bound degrades gracefully as the profile flattens, which is correct — a near-constant kernel is realisable in one dimension.

**Remark on complementarity, which must not be omitted.** At `p = 2` the bound reads `q ≥ n − log₂ S(2,s) ≈ n − 0.58` and therefore does not exceed `n`: Theorem B leaves binary-tree `n`-qubit encodings open, and it is Theorem A that closes the product case there. Theorems A and B cover different regimes; neither subsumes the other, and the paper presents them that way.

Cite `test_theorem_b_rules_out_every_n_qubit_encoding` and `test_theorem_b_closed_form_row_sum`.

- [ ] **Step 6: §6 Theorem C, exact realisation (2.5 pp)**

Statement and proof as in the design spec, with the two corrections: uniqueness **up to a global isometry and per-point phases**, and the optimality claim phrased as *matching Theorem B to within an additive `log₂ S(p,s)` qubits*. Include: the amplitude formula and the normalisation check; the `K = (A Aᵀ)^{∘2}` factorisation and the one-sentence PSD corollary; the resource table (dimension `|V|`, qubits `⌈log₂|V|⌉`, `n+1` non-zero amplitudes, `O(n·log|V|)` sparse-preparation gates, `O(n)` controlled rotations in a level-indexed layout, WordNet at 17 qubits); a circuit figure for `p = 2, n = 3`; the p-adic specialisation with `f(v) = p^{-(n-v)s}` and the link to quantum walks on the p-adic tree; and the generalisation to arbitrary rooted trees (C4), citing `test_theorem_c_generalises_to_a_non_homogeneous_tree`.

- [ ] **Step 7: Audit every proof**

```
Use the proof-verifier agent to audit sections/04-theorem-a.tex,
sections/05-theorem-b.tex and sections/06-theorem-c.tex for logical gaps, misused
hypotheses, quantifier errors, and any claim that the numerical tests do not
actually support.
```
Resolve every finding before continuing. A finding that cannot be resolved becomes a visible `\todo{}` and a weakened statement — never a silently retained overclaim.

- [ ] **Step 8: Compile and commit**

```bash
cd papers/aiqxqia2026 && make
git add papers/aiqxqia2026/sections
git commit -m "docs(paper): sections 1-6, the mathematical core"
```

---

## Task 18: Draft the empirical and closing sections (§§7–11)

**Files:**
- Modify: `papers/aiqxqia2026/sections/07-reuploading.tex` … `11-conclusion.tex`
- Copy figures: `outputs/e*/e*.pdf` → `papers/aiqxqia2026/figures/`

**Interfaces:**
- Consumes: experiment outputs, `numbers.tex`
- Produces: the rest of the manuscript

- [ ] **Step 1: §7 Re-uploading extension (1.0 p)**

Open with the observation that a single unitary applied after `Φ_f` leaves the kernel invariant, so any non-trivial extension must re-upload the data; give the `Φ^{(L)}(x) = W_L D(x) ⋯ W_1 D(x) Φ_f(x)` family and note that `γ = 0` recovers the exact construction; define distortion `D(θ)` and alignment; forward-reference E4.

- [ ] **Step 2: §8 Experiments (2.5 pp)**

One subsection per experiment. Each states the setup, the numbers (via `numbers.tex` macros, never hand-typed), and the reading. Include the E1 violation-rate figure, the E3 three-panel ablation, the E4 Pareto front, and the E5 shot-budget curve, plus one table for E2 across datasets. State the WordNet longest-hypernym-path reduction explicitly here. **Report E2 exactly as it came out.** If accuracy is below v-PuNNs, say so in one plain sentence and note that the comparison is about geometric fidelity, not accuracy — this criterion was fixed before any numbers were seen.

- [ ] **Step 3: §9 Related work (0.75 p)**

Four paragraphs: p-adic quantum mechanics; p-adic machine learning and v-PuNNs; quantum kernels and what Theorems A and B say about the standard constructions; hyperbolic embeddings as the classical answer to hierarchy, with the low-but-nonzero-distortion contrast.

- [ ] **Step 4: §10 Limitations and outlook (0.5 p)**

Named limitations: no quantum speedup in kernel evaluation; multiple-inheritance DAGs handled only by the longest-path reduction; simulation only, no hardware; the dimension bound is tight only for rapidly decaying profiles; E2 is parity evidence. Outlook: the path-state encoding as the input stage of deeper quantum models, and the p-adic gate-synthesis direction reserved for a longer treatment.

- [ ] **Step 5: §11 Conclusion (0.25 p) and the abstract**

Write the conclusion, then the abstract last: one sentence of setting, one of the question, three of the results, one of scope.

- [ ] **Step 6: Copy figures and compile**

```bash
cp outputs/e1_ultrametricity_*/e1_violations.pdf papers/aiqxqia2026/figures/
cp outputs/e3_ablations_*/e3_ablation.pdf papers/aiqxqia2026/figures/
cp outputs/e4_entangling_*/e4_pareto.pdf papers/aiqxqia2026/figures/
cp outputs/e5_noise_*/e5_noise.pdf papers/aiqxqia2026/figures/
cd papers/aiqxqia2026 && make && pdfinfo main.pdf | grep Pages
```
Expected: builds clean; body length at or above 10 pages excluding references.

- [ ] **Step 7: Commit**

```bash
git add papers/aiqxqia2026
git commit -m "docs(paper): sections 7-11, figures, abstract"
```

---

## Task 19: Self-review and submission readiness

**Files:**
- Modify: whatever the review turns up
- Create: `notes/submission-checklist.md`

- [ ] **Step 1: Style pass**

Invoke the `humanizer` skill on every section file. Target the specific patterns it lists: inflated symbolism, promotional adjectives, "-ing" analyses, rule-of-three constructions, negative parallelisms. The register to match is *Quantum Machine Intelligence*: declarative, mathematically dense, no hype.

- [ ] **Step 2: Paper self-review**

Invoke the `paper-self-review` skill against the compiled PDF. Separately confirm by hand:
- every theorem has a proof, and every proof has been through `proof-verifier`;
- every experimental numeral in the text comes from a `numbers.tex` macro;
- every citation resolves (rerun the Task 16 Step 3 verifier);
- no `\todo{}` remains anywhere, and no `TODO-AUTHOR`;
- the honest-scope paragraph is present in §1;
- body length ≥ 10 pages excluding references.

- [ ] **Step 3: Full reproduction from a clean checkout**

```bash
git status --porcelain   # must be empty
uv run pytest code/tests -v
for e in e1_ultrametricity e2_classification e3_ablations e4_entangling e5_noise; do
  uv run python -m "experiments.$e" || echo "FAILED $e"
done
uv run python -m experiments.collect_numbers
cd papers/aiqxqia2026 && make clean && make
```
Expected: tests green, all five experiments complete, the PDF rebuilds and the macro values are unchanged from what the text describes.

- [ ] **Step 4: Write `notes/submission-checklist.md`**

```markdown
# AIQxQIA 2026 submission checklist

- [ ] Author name, affiliation, ORCID, email filled in (no TODO-AUTHOR anywhere)
- [ ] Single-blind: author names ARE included
- [ ] Body >= 10 pages excluding references
- [ ] ceurart class, CEUR copyright block and conference line present
- [ ] Generative-AI disclosure section completed per CEUR policy
- [ ] All citations verified; refs.bib has no invented entries
- [ ] Public code repository created and the availability statement URL resolves
- [ ] EasyChair submission created and the PDF uploaded
- [ ] CEUR author agreement (NTP variant) signed and mailed to riccardo.rasconi@istc.cnr.it
- [ ] Submitted on or before 9 August 2026
```

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "docs(paper): style pass, self-review fixes, submission checklist"
```

---

## Self-review of this plan against the spec

**Spec coverage.** §1–3 → Tasks 17.1–17.3. §4 Theorem 1 → Task 17.4 (generalised to blocks). §5 Theorem 2 → Task 17.6. §6 entangling extension → Tasks 12 and 18.1. §7 protocol, datasets, preprocessing, baselines, reproducibility → Tasks 8–13. §8 success criteria → enforced in Task 10 Step 2 and Task 18.2. §9 honest scope → Task 17.1. §10 structure → Tasks 17–18, page targets carried over. §11 related work → Tasks 16 and 18.3. §12 layout and conventions → Tasks 1–13. §13 schedule → below. §14 risks → pre-chosen responses embedded at Task 8 Step 5, Task 10 Step 2, Task 12 Step 6. §15 out of scope → nothing in this plan touches hardware, advantage claims, gate synthesis, or DAG geometry. §16 open items → Global Constraints and Task 19 Step 4.

**Additions beyond the spec.** Theorem B and its verification tests; `collect_numbers.py`, so no experimental numeral is ever hand-typed; the `block_product` encoding, which exercises Theorem A's generalisation.

**Type consistency.** `Encoding.states(tree, leaves) -> (m, dim) complex` is the single interface every encoding implements, including `ReuploadEncoding`. `Profile.__call__` accepts an array, which is what `profile_residual(kernel, lca, profile)` relies on. `DatasetFactory(name, **kwargs)` returns a closure over `DatasetSpec`, matching every call site in `experiments/common.py`.

## Revised schedule

| Date | Tasks |
|------|-------|
| 5 Aug (today) | 1–7: scaffold through the theorem-verification harness |
| 6 Aug | 8–9: datasets, E1 green on synthetic and WordNet |
| 7 Aug | 10–13: E2–E5; 15: LaTeX scaffold compiling |
| 8 Aug | 14, 16, 17: numbers extraction, references, mathematical core |
| 9 Aug | 18–19: empirical sections, self-review, compile, **submit** |
| 10 Aug | Buffer only |

---

## Execution log

**5 August 2026 — Tasks 1–7 complete** on branch `feature/ultrametric-quantum-kernels`.
98 tests pass; `ruff` and `mypy` clean. Four deviations from the plan as written, all
deliberate:

1. **`depolarise` gained a required `dim` argument.** The plan modelled depolarising as
   `K -> (1 - rate) K + rate/2`, which pulls kernel entries *up* towards 1/2 and made
   the plan's own test assertion false. The compute-uncompute estimator reads the
   all-zeros outcome probability, which the maximally mixed state gives as `1/dim`, so
   the correct model is `K -> (1 - rate) K + rate/dim`. Task 13's call site is updated
   above.
2. **`reupload.py` was built during Task 4** rather than Task 12, because the Task 4
   test file already exercises it. Task 12 now only needs `e4_entangling.py`.
3. **`digit_matrix` is vectorised** via a stable argsort on the parent array instead of
   the plan's per-node Python loop, which would have been ~80k iterations on WordNet.
4. **Two Theorem A tests were strengthened.** The plan's
   `test_theorem_a_conclusion_forces_at_most_three_kernel_values` built an object that
   was not actually ultrametric, so it tested the wrong thing. It is replaced by
   `test_theorem_a_ultrametric_product_map_resolves_only_one_level`, which builds a
   genuinely ultrametric product map from interpolated simplex factors and checks the
   predicted two-valued kernel, plus
   `test_theorem_a_a_nontrivial_deep_factor_destroys_ultrametricity`, which exercises
   the proof's multiplicativity step directly.

Also added: `test_theorem_b_does_not_subsume_theorem_a_at_radix_two`, which pins the
complementarity claim so it cannot quietly rot into an overclaim.

**Recorded for the paper** (`notes/theorem-residuals.json`): path-state profile
residuals are `2.2e-16` at `(p,n) = (2,3)`, `1.1e-16` at `(2,5)`, `1.4e-17` at `(3,3)`,
`2.8e-17` at `(5,2)` — machine precision in every case, with zero strong-triangle
violations.

**6 August 2026 — Tasks 8–9 complete.** 117 tests pass; `ruff` and `mypy` clean. E1 is
green on synthetic (`p = 2, 3`), WordNet, GO and NCBI.

Three library changes were forced by the real hierarchies, which are far larger and
deeper than the plan assumed (WordNet: 74,374 nodes, height 19, max branching 402):

1. **`tree.restrict_to_leaves`** cuts to the sampled leaves' ancestor closure. Padding
   the full WordNet tree would have made the path-state array 2.8 GB when only the
   sampled leaves' paths can carry amplitude.
2. **`tree.truncate_at_depth` plus `encoding.MAX_QUBITS`.** Angle and ZZ need
   `2**height` amplitudes per point — `2**19` on raw WordNet. Experiments truncate at
   depth 8; the guard makes an over-deep tree fail loudly instead of exhausting memory.
3. **`BasisEncoding` emits the compact one-hot form.** Its nominal width is
   `radix ** height` = `402**19`, unrepresentable; the occupied-subspace form has an
   identical Gram, and `nominal_dim()` reports the true width for the resource table.

`pad_to_uniform_depth` now also returns a `tip` remap — padding turns a shallow leaf
into an internal node, so held leaf indices go stale. A test caught this.

**A finding that changes how E1 is reported.** Violation count does not separate the
encodings: basis encoding has **zero** violations, because `K = I` is the discrete
metric, which is ultrametric. It is also useless — it resolves nothing. This is the
`v* = n` corner of Theorem A, not a counterexample, but it means the paper cannot lead
with "path states have zero violations and the baselines do not". E1 now reports two
quantities, and only the path state wins both:

| encoding | violations | level constancy | resolution depth |
|---|---|---|---|
| path state | 0 | `4.4e-16` | full |
| basis | 0 | 0 | **2** |
| angle | ~6.5e5 | 0.95 | full |
| ZZ | ~6.7e5 | 0.72 | full |
| random product | ~6.8e5 | 0.98 | full |

§8 of the paper must present the basis encoding this way rather than omitting it.

**Two reporting artifacts fixed before they could be mistaken for findings.** The
geometric profile base is now a free parameter defaulting to 2 rather than the
branching factor — at radix 38 the profile underflowed to `4e-13` and understated
path-state resolution as 6/9. And resolution depth is compared against the LCA levels
the sample actually populates: GO's 8 is a ceiling shared by every encoding, not a
shortfall.

**Dataset shapes for the paper** (256 leaves, truncated at depth 8): WordNet 74,374
source nodes / height 19 → 1,028 nodes, radix 11, 11 classes, 11 qubits; GO
molecular-function 10,041 / 12 → 1,235 nodes, radix 16, 26 classes, 11 qubits; NCBI
Mammalia 14,722 / 14 → 633 nodes, radix 38, 4 classes, 10 qubits.

**6 August 2026 — Tasks 10–13 complete.** 130 tests pass; `ruff` and `mypy` clean. All
five experiments run end to end and emit figures.

**A new result, found while building E5: Proposition D.** Global depolarising noise
sends `K -> (1 - r)K + r/D`, hence `d -> (1 - r)d + r(1 - 1/D)` — an increasing affine
map. Affine maps preserve order and commute with `max`, so the strong triangle
inequality survives **exactly**, at every rate. Verified numerically to `r = 0.99`. What
depolarising destroys is contrast, not geometry: the profile flattens and resolution
depth falls (6 → 4 at `r = 0.99`). This belongs in §7 of the paper as a proposition
with a two-line proof, and it materially improves the noise story: the construction's
ultrametricity is immune to the dominant hardware error channel.

**Three measurement corrections, each caught because a result looked wrong.**

1. **E5 must report violation magnitude, not count.** In an ultrametric every triangle
   is isoceles with its two longest sides equal, so a large fraction of triples — 66.7%
   on the `p=2, n=5` tree — meets the inequality with equality. Any perturbation flips
   about half of them, so the count *rises* toward ~0.30 as shots increase, which is
   the opposite of the expected behaviour and pure artifact. The magnitude behaves
   properly, decaying as `O(1/sqrt(shots))`: `0.134, 0.049, 0.014, 0.0049, 0.0017` for
   `1e2 … 1e6` shots, a factor of `sqrt(10)` per decade. **Shot budget: 1e5 shots per
   kernel entry** for worst-case excess below 0.01.
2. **E3's crossing figure must use the closed-form full-tree bound.** Evaluating
   Theorem B on a 128-leaf subsample stays a valid bound but is capped at
   `log2(128) = 7` qubits, so the curve saturated instead of crossing. Added
   `metrics.regular_tree_dimension_bound`, which computes `p**n / S` without
   materialising the tree. The crossing is now unambiguous: at `p = 2` the bound is
   below `n` at every depth (1.54, 2.48, 3.45, 4.43, 5.42 for `n = 2..6`); at `p = 3` it
   is above at every depth (2.80, 4.35, 5.93, 7.51, 9.10); at `p = 5` far above.
3. **E2 reports undefined Spearman rho as `None` with a note.** A delta kernel gives
   every distinct pair the same distance, so no rank correlation exists; NaN would read
   as a failed computation.

Also pinned: **Theorem B genuinely needs `s > 1`.** At `s = 1` every term of the row sum
is 1, so `S = 1 + n(p-1)/p` grows with `n` and the bound degrades to `p**n / Theta(n)`.
The hypothesis is load-bearing, not decoration.

**E2 results, to be reported exactly as they came out.** On WordNet the path state
leads on geometric fidelity by a wide margin and *loses* on leaf accuracy:

| encoding | leaf acc. | root acc. | Spearman rho |
|---|---|---|---|
| path state | 0.792 | 0.937 | **1.000** |
| random product | **0.839** | 1.000 | 0.368 |
| angle | 0.675 | 1.000 | 0.175 |
| ZZ | 0.478 | 0.824 | 0.098 |
| basis | 0.212 | 0.667 | undefined |

GO is the same shape (path state 0.601 vs random product 0.750). This is exactly the
case §8 of the spec pre-committed to reporting plainly: accuracy and geometric fidelity
are different objectives, and the paper's claim is the latter. The path state's
`rho = 1.000` is by construction, not a fitted result, and must be described that way.

**E4.** Distortion grows continuously from zero, so the exact construction is the
endpoint of a tunable family rather than an isolated point. The alignment available is
small: on WordNet the best configuration reaches 0.4258 against the exact point's
0.4212, a gain of 0.005 for distortion 0.046. §7 should say plainly that the trade-off
exists but buys little on these datasets.

**6 August 2026 — Tasks 14–16 complete.** 130 tests pass; `ruff` and `mypy` clean. The
manuscript compiles to 3 pages of scaffold with all 16 references typeset and zero
BibTeX errors.

**Task 14.** `collect_numbers.py` emits 48 macros from the newest run of each
experiment. Every one uses `\providecommand` + `\renewcommand`, so the file is
idempotent and a dropped macro degrades to empty rather than breaking the build.

**Task 15 — toolchain findings worth keeping.** `ceurart` needs `ccicons` and
`elsarticle-num-names`, neither in TeX Live 2023 here, and `tlmgr` refuses to install
(local 2023 older than the remote 2026 repository). Both were fetched from CTAN into
`$HOME/texmf`; the Makefile exports `TEXMFHOME` so the build does not depend on the
ambient environment. `fontawesome5` is also missing, but the class guards it with a
file-exists test, so it degrades to no ORCID icon. Two further build facts:

- **`lmodern` is required.** Without `cm-super`, T1 Computer Modern falls back to
  bitmaps and microtype's font expansion aborts the run outright.
- **Do not set `\bibliographystyle` in `main.tex`.** The class already sets
  `elsarticle-num-names`; a duplicate makes BibTeX abort with *"Illegal, another
  \bibstyle command"* — reported only in the `.blg`, while a stale `.bbl` survives and
  the build looks fine. This cost a debugging cycle and would have shipped a paper with
  no bibliography.

Layout choices: one column (the `ceurart` default and CEUR house style) and **no**
`singleblind` option, since that anonymises and the workshop wants names visible.

**Task 16 — 16 references, all machine-verified.** `verify_refs.py` resolves every
arXiv id and DOI *and* compares the recorded title against the real record, because an
identifier that resolves to a different paper looks fine while being wrong. Non-zero
exit on failure, so it gates submission.

The gate paid for itself on first run. Of three failures, one was real: I had written
the title for arXiv:2401.04642 from memory rather than from the record. It is *Neural
quantum kernels: training quantum kernels with quantum neural networks*. The other two
were a bug in the comparison — LaTeX accent escapes against Unicode — now normalised
through NFKD.

**Correction to the design spec §11.** The Aniello et al. p-adic qubit paper is in
**Entropy** 25(1):86, DOI 10.3390/e25010086 — not *Symmetry*.

**v-PuNNs, confirmed and consequential for §8.** arXiv:2508.01010, sole author Gnankan
Landry Regis N'guessan, still a preprint (revised January 2026). It reports **99.96%
leaf accuracy on WordNet**, 96.9%/100% on GO, and Spearman |rho| = 0.96 on NCBI. Our
path state gets 0.792 leaf accuracy on WordNet. The gap is large and must be stated
plainly, alongside the point that v-PuNNs is a trained deep model with a bespoke
optimiser while ours is a fixed encoding with a closed form and rho = 1.000 by
construction. This is exactly the scenario §8 of the spec pre-committed to.

Items the literature search could not verify were **dropped, not cited**: a Murtagh
2009 ultrametric-clustering paper (bibliographic detail unconfirmed) and a KDD '25
acceptance claim for arXiv:2507.17787, which is cited as a preprint instead.
