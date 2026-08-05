---
name: proof-style
description: Conventions for writing mathematical proofs in this workspace — structure, level of detail, use of proof environments, handling of gaps and references. Use whenever drafting, editing, or reviewing a proof in this workspace.
---

# Proof-writing conventions

The audience is a working mathematician in the same field. Write what they
cannot fill in; skip what they can.

## Structure

Every nontrivial proof should have a visible skeleton. For proofs longer than
half a page:

1. **Opening sentence** that names the strategy. "We argue by induction on n."
   "We apply the direct method of the calculus of variations." "The idea is to
   construct an approximating sequence and pass to the limit."
2. **Named intermediate claims** marked explicitly: "We claim that ..." followed
   by the claim in display style, then "Proof of Claim:" in the next paragraph.
3. **Closing sentence** that connects the final estimate back to the statement
   of the theorem. Don't leave the reader to reconstruct why the argument
   concludes.

## Level of detail — the three-tier rule

Classify each step as one of:

- **Routine** — a competent reader fills it in. Say it in one clause:
  "by the Cauchy–Schwarz inequality," "by continuity of f," "integrating by parts".
- **Standard but worth citing** — give a reference: "by [Evans, Thm 3.2]",
  "by the dominated convergence theorem".
- **Nontrivial** — write it out fully. If this step is the heart of the proof,
  it probably deserves its own lemma.

## Proof environments

Use AMS `proof` environment for theorem proofs. For a proof embedded inside
another proof (e.g. proof of a claim), open a nested `proof` with an optional
argument: `\begin{proof}[Proof of Claim 1]`.

End every proof with `\end{proof}` — the `\qed` symbol is inserted automatically.
Don't write "This completes the proof." The `\qed` says so already.

## Handling gaps

Never hide a gap. If a step is not justified, one of:

1. Insert `\todo{justify that the limit commutes with the integral}` inline.
2. State the missing lemma as a "we claim" with a following `\todo` describing
   what to prove.
3. Stop the draft and flag the blocker explicitly to the user.

If you don't know whether a step is true, say "We believe that ... but have not
verified this." Never present unknowns as knowns.

## Quantifiers and dependencies

In analysis, make ε–δ dependencies explicit. If δ depends on ε and on some
norm of the data, say so at the point of choice:

> Given ε > 0, choose δ = δ(ε, ‖f‖∞) > 0 small enough that ...

Uniformity claims should be stated uniformly:

> ... where the constant C is independent of n.

## Constants

If you write "C" and it changes from line to line, say "where C denotes a
constant, possibly changing from line to line, depending only on n and p."
Say this once at the start of the section; don't repeat it.

For arguments where tracking constants matters (quantitative estimates, rates
of convergence), subscript them: C₁, C₂, ...

## Cases

Multi-case arguments get numbered cases:

```latex
\textbf{Case 1: \(x \in \Omega\).}
...

\textbf{Case 2: \(x \in \partial\Omega\).}
...
```

Each case ends with a paragraph break, not with "This concludes Case 1."

## What to cut

When polishing:

- Remove "clearly", "obviously", "it is easy to see" — if it's easy, the reader
  will see it; if not, you're gaslighting them.
- Remove "note that", "observe that", "we point out that" — just state the fact.
- Remove "in order to" → "to"; "due to the fact that" → "because".
- Keep "we" and "note" together at most once per paragraph.

## Figures and diagrams

For arguments that benefit from a picture (geometric, topological, categorical,
commutative diagrams) — include one. Use TikZ, tikz-cd for diagrams, or
asymptote for 3D. Don't describe what a diagram would show in prose; draw it.

## Voice

First-person plural ("we") is standard in mathematical writing. Imperative mood
for instructions to the reader ("Let f be ...", "Assume ..."). Present tense for
argument, past tense only for historical remarks.
