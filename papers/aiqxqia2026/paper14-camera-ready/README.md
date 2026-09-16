# $p$-adic Variational Quantum Circuits Have No Barren Plateaus

Workshop paper targeting **AIQxQIA 2026** — International Workshop on AI for
Quantum and Quantum for AI (CNR, Italy), <https://aiqxqia2026.cnr.it/>.

## Submission facts

| Item | Value |
|---|---|
| Abstract deadline | 3 August 2026 |
| Paper deadline | 10 August 2026 (extended) |
| Notification | 31 August 2026 |
| System | EasyChair, `aiqxqia2026` |
| Review | Single-blind — author names stay in |
| Format | CEUR-WS single column (`ceurart`) |
| Length | Full paper ≥10 pp. excl. refs; this draft is **12 pp. excl. refs** |
| Proceedings | CEUR-WS; selected papers invited to extend in Springer *Quantum Machine Intelligence* |

## Relationship to the CSUR survey

This paper resolves open problem **O13** from
`papers/PAdicQML-CSUR/sections/11_agenda.tex` (barren plateaus over $p$-adic
unitary groups). Since the camera-ready revision the paper does not cite the
survey: it states the problem as Question 1.1, in its own corrected form. It is new content, not a condensation of the survey — there is
no textual overlap beyond shared preliminaries, so it does not create a
dual-submission conflict with the CSUR manuscript.

It also **corrects** the O13 framing in the survey: the survey assumes a Haar
measure on the full $p$-adic unitary group. Proposition 3.1 here shows that
group is non-compact, so no Haar *probability* measure exists and the ensemble
must be the integral unitary group. The survey's O13 text and
`sections/10_neural_networks.tex` §"Trainability and barren plateaus" should be
updated to match if this paper is accepted.

## Results

Numbering follows the current `main.tex` (revised 2026-08-09).

1. **Prop. 3.1 / Ex. 3.2** — $\mathrm{U}(K^N)$ is non-compact for $N \ge 2$,
   with an explicit unbounded family at $p=3$. The plateau question as usually
   posed is therefore ill-defined.
2. **Def. 3.3 / Prop. 3.4–3.5** — the integral unitary group
   $G_N = \mathrm{U}_N(\mathcal{O})$ is compact and contains every circuit
   realisable from known $p$-adic gate sets.
3. **Remark 2.4** — the scale $p^{-1}$, not $1$, is the correct benchmark: the
   $p$-adic convergence bound is *equivalent* to $\lVert X\rVert \le p^{-1}$
   (the value group is discrete), so every $\mathbb{Z}_p$-analytic
   one-parameter subgroup lies in the principal congruence subgroup and
   $\lvert\partial_\theta\mathcal{L}\rvert_p \le p^{-1}$ always. Results are
   stated for the normalised gradient $\nabla\mathcal{L}=p^{-1}\partial_\theta\mathcal{L}$.
4. **Lemma 4.1 / 4.2** — Haar averages over $G_N$ reduce to counting in
   $\mathrm{U}_N(\mathcal{O}/\mathfrak{m}^k)$, and Haar-random states reduce to
   the uniform measure on the finite Hermitian sphere. This replaces the
   missing $t$-design combinatorics. Both proofs are self-contained (Newton
   iteration; lattice splitting) — no scheme theory.
5. **Lemma 5.3** — explicit character-sum count for a Hermitian form on the
   finite Hermitian sphere, proved from scratch.
6. **Theorem 5.6** — the gradient attains its maximal attainable absolute value
   $p^{-1}$ with probability $\ge 1 - 1/p - 10p^{1-r}$, independent of the
   qubit count. No barren plateau. Stated for **arbitrary fixed $U_B$**, only
   $U_A$ Haar-random; Cor. 5.7 gives the random-$U_B$ version.
   **Conditional** on the splitting hypothesis — see below.
7. **Ex. 5.8** — a witness: $M = X_1$, $A = -\tfrac{i}{2}(\mathbf{1}-Z_1)$ over
   $\mathbb{Q}_p(i)$ for $p\equiv 3 \bmod 4$ gives $\Gamma = Y_1$ and
   $r = 2^{n-1}$.
8. **Cor. 5.9** — the degenerate locus is exactly
   $[\widehat M, A] \equiv 0 \bmod p$. Note it has *small but positive*
   measure, not measure zero (§6.1).

Limitations are stated explicitly in §6.3.

## Known limitation: the splitting hypothesis

Theorem 5.6 is **conditional** on the diagonalisability clause of Definition
5.2, and that clause is a genuine restriction, not a formality. Claim-level
citation checking on 2026-07-28 found that the paper had originally justified
it as "the spectral theorem for Hermitian matrices over a finite field" citing
Grove — which is false. Over $\mathbb{F}_{q^2}$ a Hermitian matrix can have
eigenvalues outside $\mathbb{F}_q$; Remark 5.4 now gives the counterexample

$$\Gamma = \begin{pmatrix} 0 & 1+\iota \\ 1-\iota & 0\end{pmatrix} \text{ over }
\mathbb{F}_9, \qquad \chi_\Gamma(x) = x^2+1,$$

whose roots $\pm\iota$ lie in $\mathbb{F}_9 \setminus \mathbb{F}_3$ (verified by
enumeration). The abstract, introduction, §6.3 and the conclusion all carry the
hypothesis explicitly. Closing this gap needs the classification of *pencils*
of Hermitian forms over a finite field, not a single diagonalisation, and is
the main open follow-up.

Ex. 5.8 shows the restricted class is non-empty and contains an entirely
ordinary observable–generator pair with $r = 2^{n-1}$. It does not show the
class is large.

## Revision log — 2026-08-09

Full referee pass. Substantive changes:

- **Normalisation defect fixed (was blocking).** The old ansatz
  $\exp_p(\theta A)$ with $\lVert A\rVert \le p^{-1}$ forced
  $\Gamma = \widetilde{[\widehat M, A]} = 0$ identically, so the old Theorem
  5.5 was vacuous — the gradient had $v_p \ge 1$ with probability 1. The $p$
  now sits in the ansatz, $\exp_p(p\theta A)$ with $A$ integral, and results
  are stated for $\nabla\mathcal{L} = p^{-1}\partial_\theta\mathcal{L}$.
  Remark 2.4 proves the scale is forced rather than chosen.
- **Theorem strengthened** to arbitrary fixed $U_B$ (only $U_A$ Haar-random),
  which also makes its hypothesis checkable on a concrete pair.
- **Witness example added** (Ex. 5.8).
- **Two false claims corrected.** The old Remark 4.8 claimed $m_{\max}=O(1)$;
  pigeonhole gives $m_{\max}\ge\lceil N/p\rceil$, so a simple spectrum is
  impossible once $N>p$. The old §6.1 claimed the bad set has measure zero; it
  is a union of congruence classes with positive measure $O(p^{-1})$.
- **Proofs made self-contained** for Lemmas 4.1 and 4.2, replacing appeals to
  smoothness of the unitary group scheme.
- **Citations.** `HCT19` (Havlíček et al., quantum kernels) was wrongly cited
  for the parameter-shift rule; replaced by `Cr19` and `Wi22`. The open problem
  O13 is now cited to the survey (`Gi26survey`) instead of dangling.
- Notation collisions removed ($N$ vs. the field norm, now `\Nm`; $\psi$ vs.
  the additive character, now $\chi$).

## Building

```bash
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

`ceurart.cls`, `elsarticle-num-names.bst` and `cc-by.pdf` are vendored from the
official CEUR template (<https://ceur-ws.org/Vol-XXX/CEURART.zip>).

Local builds additionally need `ccicons.sty` (pulled in by `doclicense`), which
is not in a minimal TeX Live. On Overleaf and on the CEUR production system it
is present. The preamble disables `microtype` font expansion so the build
succeeds without the Libertinus fonts; this is harmless where they are
installed.

## Before submitting

- [x] ~~Run the seven references added for this paper through the
      citation-verification workflow.~~ Done 2026-07-28: `Ce21`, `Ho22`,
      `Se73`, `Gr02`, `Ca72`, `PR94`, `DDMS99` all verified against publisher
      records; DOIs and series data added. Verification also caught a false
      claim attributed to `Gr02` — see below.
- [x] ~~Fill in the `\conference{...}` line once AIQxQIA announces the date and
      co-located venue.~~ Done 2026-09-14: co-located with AIxIA 2026, Perugia,
      per the camera-ready instructions.
- [ ] Confirm affiliation and ORCID in the author block.
- [x] ~~Replace the `Gi26survey` placeholder.~~ Done 2026-09-14: the entry is
      removed and the paper no longer cites the survey. The verbatim O13 box
      is replaced by a self-contained Question 1.1, posed over $G_N$ with the
      $p^{-1}$ benchmark, which Theorem 5.6 answers (reviewer R2's point).
- [ ] Optional: add the exact-arithmetic enumeration check of Lemma 5.3 for
      small $n, p$ described in §6.3.
