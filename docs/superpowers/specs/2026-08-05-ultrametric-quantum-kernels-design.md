# Ultrametric Quantum Kernels — Design Specification

**Working title.** Ultrametric Quantum Kernels: Exact p-adic Feature Maps for Hierarchical Data

**Venue.** AIQxQIA 2026 — 4th International Workshop on AI for Quantum and Quantum for AI (CNR / ISTC), CEUR-WS proceedings.

**Category.** Full paper, ≥ 10 pages excluding references.

**Deadlines.** Paper submission 10 August 2026 (extended). Notification 31 August 2026. Internal target: submit 9 August 2026, leaving 10 August as buffer. Abstract deadline (3 August 2026) has lapsed; the Instructions-for-Authors page states no mandatory-abstract rule, but EasyChair submission creation must be confirmed by the author.

**Submission.** EasyChair; CEUR-WS Overleaf template (`ceurart`); single-blind, author names included. Corresponding author signs the CEUR CC-BY author agreement (NTP variant, assuming no third-party copyrighted material) and mails it to riccardo.rasconi@istc.cnr.it.

**Authorship.** Single author. Full name, affiliation, and contact email pending from the author — tracked in *Open items* below.

**Code release.** Public repository, linked from the paper with an availability statement.

---

## 1. Problem and motivation

Quantum kernel methods embed classical data through a feature map `Φ` and compare points by fidelity,
`K(x, y) = |⟨Φ(x)|Φ(y)⟩|²`. The geometry that `Φ` induces on the data is therefore fixed entirely by the
encoding, before any training happens.

Hierarchical data — taxonomies, ontologies, phylogenies, file systems, type lattices — is not Euclidean.
Its natural geometry is **ultrametric**: distances satisfy the strong triangle inequality
`d(x, z) ≤ max(d(x, y), d(y, z))`. Equivalently, the data sits at the leaves of a tree and distance is a
decreasing function of the depth of the lowest common ancestor.

The p-adic integers `ℤ_p` are the canonical ultrametric space. Truncating at depth `n` gives `ℤ/pⁿ`, and the
p-adic valuation `v_p(x − y)` — the number of matching least-significant base-p digits — is exactly the
LCA depth in the regular p-ary tree.

The paper asks a single question:

> Which quantum feature maps induce an exactly ultrametric kernel?

It answers with a matched no-go and construction.

## 2. Contributions

- **C1 — A no-go theorem for product encodings.** Angle encoding, IQP/ZZ-style encoding, and basis encoding
  are all product feature maps. None of them can induce a non-degenerate ultrametric kernel. This is a
  constraint on current QML practice, not a statement about one particular map.
- **C2 — An exact construction at optimal width.** The *path-state encoding* realises any strictly monotone
  ultrametric kernel exactly, using `⌈log₂|V|⌉` qubits for a tree with `|V|` nodes and `O(n)` preparation
  operations. The family is complete: it realises exactly the strictly monotone ultrametric kernels, and
  nothing else.
- **C3 — A generalisation beyond the homogeneous case.** The construction needs only that ancestors nest,
  so it applies to arbitrary rooted trees. `ℤ_p` is the homogeneous specialisation in which the profile ties
  directly to `|·|_p` and to quantum walks on the p-adic tree.
- **C4 — A measured expressivity/hierarchy trade-off.** Stacking trainable entangling layers on the path
  state raises expressivity and degrades ultrametricity. We map the resulting Pareto front empirically.

## 3. Formal setup

Let `p` be prime and `n ≥ 1`. Write `X = ℤ/pⁿ`, and for `x ∈ X` let `x = Σᵢ₌₁ⁿ xᵢ p^{i-1}` be its base-p
expansion with digits `xᵢ ∈ {0, …, p−1}`. Define

```
v(x, y) = #{ leading matching least-significant digits of x and y }  =  v_p(x − y),   capped at n
```

A kernel `K : X × X → [0, 1]` is **ultrametric with profile `f`** if there is `f : {0, …, n} → [0, 1]` with
`f(n) = 1` and `K(x, y) = f(v(x, y))` for all `x, y`. The profile is **strictly monotone** if `f` is strictly
increasing. `d = 1 − K` is then an ultrametric, because LCA depth satisfies
`λ(x, z) ≥ min(λ(x, y), λ(y, z))`.

A **product feature map** is `Φ(x) = ⊗ᵢ₌₁ⁿ |ψᵢ(xᵢ)⟩` with unit vectors `|ψᵢ(a)⟩ ∈ ℂ^{dᵢ}`, factor `i`
depending only on digit `i`, and local dimensions `dᵢ` arbitrary. Write `gᵢ(a, b) = |⟨ψᵢ(a)|ψᵢ(b)⟩|²`, so
`gᵢ(a, a) = 1` and `K(x, y) = ∏ᵢ gᵢ(xᵢ, yᵢ)`.

## 4. Theorem 1 — no-go for product encodings

**Statement.** Let `Φ` be a product feature map on `ℤ/pⁿ` whose fidelity kernel is ultrametric with profile
`f`, and let `v* = min{ v : f(v) > 0 }`. Then for every level `i ≥ v* + 2` and all digits `a, b`,
`|⟨ψᵢ(a)|ψᵢ(b)⟩| = 1`. Consequently `f` takes at most the three values `{0, f(v*), 1}`, and `Φ` resolves the
tree only to depth `v* + 1`. In particular, for `n ≥ 2` no product feature map realises a strictly monotone
profile.

**Proof.** Take the generic case `v* = 0`; the general case shifts every index by `v*`. Fix a level `i ≥ 2`
and digits `a ≠ b`. Choose `x, y ∈ X` differing in digit 1 and agreeing everywhere else, with `xᵢ = a`. Then
`v(x, y) = 0`, and since every other factor contributes `gⱼ(xⱼ, xⱼ) = 1`,

```
f(0) = K(x, y) = g₁(x₁, y₁).
```

Now let `y'` agree with `y` except at digit `i`, where `y'ᵢ = b`. Since `x` and `y'` still first differ at
digit 1, `v(x, y') = 0` and so `K(x, y') = f(0)`. But multiplicativity gives
`K(x, y') = g₁(x₁, y₁) · gᵢ(a, b) = f(0) · gᵢ(a, b)`. As `f(0) > 0`, this forces `gᵢ(a, b) = 1`, i.e. the
states at level `i` coincide up to phase. Levels `≥ 2` therefore carry no information, `K` depends only on
digit 1, and `f(v) = 1` for all `v ≥ 1`. ∎

The proof turns on exactly one property — fidelity is multiplicative across tensor factors — which is why it
holds for arbitrary local dimensions and rules out an entire class of encodings rather than specific
instances.

**Corollaries to state explicitly in the paper:** angle encoding, IQP/ZZ feature maps, and computational-basis
encoding all fall under the hypothesis, hence all distort hierarchy.

**Numerical confirmation** (`code/tests/test_theorems.py`): randomly drawn product maps are never ultrametric
for `(p, n) ∈ {(2,3), (3,2), (2,4)}`; a map constructed to satisfy the theorem's conclusion yields exactly two
distinct kernel values, as predicted.

## 5. Theorem 2 — exact realisation by path states

Let `T` be a finite rooted tree with node set `V` and all leaves at depth `n` (pad shallower leaves). For a
leaf `x` and `0 ≤ i ≤ n`, let `ancᵢ(x)` denote its depth-`i` ancestor, and let `λ(x, y)` be the LCA depth.

**Statement.** For every strictly increasing `f : {0, …, n} → (0, 1]` with `f(n) = 1`, set

```
aᵢ² = √f(i) − √f(i−1)        (i = 0, …, n;  √f(−1) := 0)
Φ_f(x) = Σᵢ₌₀ⁿ aᵢ |ancᵢ(x)⟩  ∈ ℂ^V
```

Then `Φ_f(x)` is a unit vector and

```
|⟨Φ_f(x)|Φ_f(y)⟩|² = f(λ(x, y))     exactly, for all leaves x, y.
```

Conversely, every ultrametric kernel with strictly monotone profile arises from such a `Φ_f`, unique up to
relabelling of `V`. The correspondence `f ↔ Φ_f` is therefore a bijection onto the strictly monotone
ultrametric kernels.

**Proof.** Normalisation: `Σᵢ aᵢ² = √f(n) = 1`. Two leaves share exactly their ancestors up to the LCA, so

```
⟨Φ_f(x)|Φ_f(y)⟩ = Σᵢ aᵢ² · 1[ancᵢ(x) = ancᵢ(y)] = Σ_{i ≤ λ(x,y)} aᵢ² = √f(λ(x, y)),
```

and squaring gives the claim. For the converse, `aᵢ² = √f(i) − √f(i−1) > 0` is forced by strict monotonicity
and determines `Φ_f` up to the labelling of `V`. ∎

**Structural remark.** The kernel matrix factorises as `K = (A Aᵀ)∘²` where `A` is the leaf–ancestor incidence
matrix weighted by `aᵢ`. Positive semi-definiteness is therefore automatic, which re-derives the classical
fact that monotone ultrametric kernels are PSD — a pleasant by-product worth one sentence in the paper.

**Resources.** Dimension `|V|`, hence `⌈log₂|V|⌉` qubits. WordNet nouns (~82k leaves, ~117k nodes) fit in 17
qubits. The state carries `n + 1` non-zero amplitudes, so sparse state preparation costs `O(n · log|V|)`
gates; in a level-indexed layout it is `O(n)` controlled rotations.

**p-adic specialisation.** For the regular p-ary tree, `λ = v_p(x − y)` and the profile can be tied to the
p-adic absolute value, e.g. `f(v) = p^{-(n-v)·s}` for a scale parameter `s > 0`. In this case `Φ_f` is a
level-weighted superposition over the root-to-leaf path in the p-adic tree, connecting directly to the
quantum-walk-on-`ℤ_p` literature.

**Numerical confirmation.** `max |K − f(v)| ≤ 1.4 × 10⁻¹⁷` across `(p, n) ∈ {(2,3), (2,5), (3,3), (5,2)}`.
Induced metric shows 0 strong-triangle violations, against 112 for an RBF kernel on the same 8 points.

## 6. Beyond exactness — the entangling extension

Applying a single parametrised entangling circuit `U(θ)` to `Φ_f(x)` leaves the kernel unchanged, since
`|⟨Φ_f(x)|U(θ)†U(θ)|Φ_f(y)⟩|² = |⟨Φ_f(x)|Φ_f(y)⟩|²`. Any non-trivial extension must therefore **re-upload the
data**: the extended map alternates `L` path-state injection layers with entangling layers,
`Φ^{(L)}(x) = W_L(θ_L) V_f(x) ⋯ W_1(θ_1) V_f(x) |0⟩`, where `V_f(x)` prepares `Φ_f(x)` and each `W_ℓ` is a
hardware-efficient entangling block. We then measure

- **ultrametric distortion** `D(θ) = max over triples (x,y,z) of max(0, d(x,z) − max(d(x,y), d(y,z)))`,
  with `d = 1 − K_θ`, so `D = 0` exactly when `K_θ` is ultrametric; and
- **kernel–target alignment** as the expressivity proxy,

and reports the Pareto front. The expected and interesting finding is that distortion grows continuously from
zero, so the exact construction is the endpoint of a tunable family rather than an isolated special case.

## 7. Experimental protocol

All experiments are analytic or statevector, CPU-only, NumPy + scikit-learn, minutes per run.

**Datasets.** WordNet nouns via `nltk` (82,115 synsets, verified loading locally); Gene Ontology
molecular-function via `go-basic.obo` (32 MB, verified reachable); NCBI Mammalia via `taxdump.tar.gz`
(77 MB, verified reachable); synthetic regular p-ary trees for controlled ablations.

**Preprocessing declared in the paper.** WordNet's hypernym structure is a DAG. We induce a tree by taking the
longest hypernym path per synset and state this explicitly; multiple-inheritance handling is a named
limitation, not a silent choice.

| ID | Experiment | Measures | Purpose |
|----|-----------|----------|---------|
| E1 | Ultrametricity audit | strong-triangle violation rate, Gromov δ | Makes Theorem 1 visible: 0 for path states, > 0 for every product encoding |
| E2 | Hierarchical classification | leaf accuracy, root accuracy, Spearman ρ vs. true tree distance | Parity evidence against v-PuNNs, hyperbolic embeddings, Euclidean baselines |
| E3 | Ablations | profile `f` (uniform / geometric `p^{-i}` / alignment-learned), depth, radix | Shows which design choices matter |
| E4 | Entangling extension | distortion vs. alignment | Pareto front of C4 |
| E5 | Noise and shots | violations under depolarising noise and finite sampling | Shot budget preserving ultrametricity within tolerance |

**Baselines.** Angle encoding, IQP/ZZ feature map, RBF on integer coordinates, RBF on one-hot, Poincaré
embeddings, and v-PuNNs' published numbers.

**Reproducibility.** Fixed seeds, recorded environment and library versions, `uv`-managed lockfile, outputs
under `outputs/{experiment}_{timestamp}/`.

## 8. Success criteria

The headline claim is **exact ultrametricity at optimal qubit count**, established by Theorems 1 and 2 and
demonstrated by E1. It does not depend on E2.

E2 is reported as parity evidence. If kernel-SVM accuracy lands below v-PuNNs, that is reported plainly and
without reframing: v-PuNNs is a trained deep model, this is a fixed encoding with a closed form, and the
comparison is about geometric fidelity rather than accuracy. This criterion is fixed now, in advance of
seeing any numbers, precisely so that a weak E2 cannot tempt a retrofit of the paper's claims.

## 9. Honest scope — stated in the paper's introduction

`K(x, y) = f(λ(x, y))` is classically computable in `O(depth)`. There is **no quantum speedup in kernel
evaluation**, and the paper says so up front rather than leaving a reviewer to find it. The contribution is
(i) a no-go that constrains QML encoding practice, (ii) an optimal-width encoding usable as the input stage
of deeper quantum models where no closed form survives, and (iii) the measured expressivity/hierarchy
trade-off of C4.

## 10. Paper structure

| § | Content | pp |
|---|---------|-----|
| 1 | Introduction — hierarchy in QML, the two results, honest scope | 1.50 |
| 2 | Preliminaries — `ℤ_p`, ultrametricity, quantum kernels | 1.50 |
| 3 | Problem statement — ultrametric quantum kernels defined | 0.75 |
| 4 | Theorem 1 and corollaries for angle/IQP/basis encodings | 1.50 |
| 5 | Theorem 2 — path states, circuit, resources, p-adic case | 2.50 |
| 6 | Entangling extension and distortion | 1.00 |
| 7 | Experiments E1–E5 | 2.50 |
| 8 | Related work | 0.75 |
| 9 | Limitations and outlook | 0.50 |
| 10 | Conclusion | 0.25 |
|   | **Total (excluding references)** | **12.75** |

## 11. Related-work positioning

- **p-adic quantum mechanics.** The p-adic qubit / quNit model over quadratic extensions of `ℚ_p`
  (Aniello et al., *Symmetry*, 2023); p-adic Schrödinger equations as scaling limits of continuous-time
  quantum Markov chains and quantum walks on p-adic trees (Zúñiga-Galindo, arXiv:2508.06712, 2025).
- **p-adic machine learning.** v-PuNNs — van der Put neural networks with p-adic ball neurons, a Finite
  Hierarchical Approximation Theorem, and results on WordNet / GO / NCBI Mammalia (N'guessan,
  arXiv:2508.01010, 2025, rev. 2026). This work supplies our benchmarks and explicitly gestures at an
  undeveloped quantum extension.
- **Quantum kernels.** Standard feature-map constructions, all of which fall under Theorem 1.
- **Hyperbolic embeddings.** The classical answer to hierarchy in ML, which achieves low but non-zero
  distortion; our construction achieves zero by design.

All citations go through `reference-manager` for cite-key consistency, and every entry is verified against a
real record before it enters `refs.bib`.

## 12. Repository layout

```
papers/aiqxqia2026/        main.tex, ceurart.cls, refs.bib, figures/
code/padic_kernel/         tree.py, encoding.py, kernels.py, metrics.py
code/experiments/          e1_ultrametricity.py … e5_noise.py
code/tests/                test_theorems.py (the Theorem 1/2 verification harness)
docs/superpowers/specs/    this document
outputs/                   {experiment}_{timestamp}/
```

**Engineering conventions.** 200–400 line modules, type hints throughout, module-level loggers (no `print`),
frozen dataclass configs, factory/registry for encodings and datasets, `__all__` in every package
`__init__.py`, fixed seeds.

**One deviation from the standing defaults:** frozen dataclasses plus `argparse` instead of Hydra. Hydra's
composition and override machinery pays off across large sweeps; for five experiment scripts on a five-day
schedule it is setup cost without payoff. Every other convention is followed as written.

## 13. Schedule

| Date | Work |
|------|------|
| 5–6 Aug | Repo scaffold, CEUR template, tree loaders (WordNet/GO/NCBI), path-state encoding, product baselines, E1 green |
| 7 Aug | Theorems written formally with complete proofs; E2 across all three datasets |
| 8 Aug | E3/E4/E5, figures, complete first draft |
| 9 Aug | References, self-review, compile, **submit** |
| 10 Aug | Buffer only |

## 14. Risks and pre-chosen responses

| Risk | Response |
|------|----------|
| E2 accuracy below v-PuNNs | Report plainly; headline claim is independent (§ 8) |
| GO or NCBI preprocessing proves fiddly | Drop to WordNet + synthetic trees with a stated note; do not spend a day on it |
| Reviewer objects "no quantum speedup" | Pre-empted in the introduction (§ 9) |
| EasyChair will not accept a new submission after the lapsed abstract date | Author confirms early; if blocked, the work retargets to a later venue with no wasted effort |
| Entangling extension (E4) shows no interesting structure | Section 6 shrinks to a short remark; C1–C3 carry the paper |

## 15. Out of scope

- Hardware execution — simulation only.
- Quantum advantage claims of any kind.
- The p-adic-descent gate-synthesis direction (candidate B), which is mentioned in § 9 as the natural
  continuation and reserved for a later, longer treatment.
- Multiple-inheritance DAG geometry beyond the stated longest-path reduction.

## 16. Open items requiring the author

1. **Author block** — full name, affiliation, and contact email for the single-blind submission and the CEUR
   copyright form. Drafting proceeds meanwhile; this must be resolved before upload.
2. **EasyChair confirmation** — that a new submission can still be created given the lapsed abstract date.
3. **Public repository** — remote created before submission so the availability statement resolves.
