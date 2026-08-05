---
name: lean4-mathlib
description: Idioms and conventions for Lean 4 formalization with Mathlib — naming, namespaces, tactic style, how to search for existing lemmas. Use whenever writing Lean 4 code, translating an informal proof into Lean, or debugging a Lean tactic error.
---

# Lean 4 + Mathlib conventions

Mathlib has strong conventions. Follow them — non-conformant Lean is harder to
read, harder to maintain, and won't be accepted upstream if you ever contribute.

## Project setup

A new formalization project:

```bash
cd formalizations/
lake new <project-name> math     # sets up Mathlib dependency
cd <project-name>
lake update                      # fetches Mathlib
lake exe cache get               # downloads precompiled Mathlib (saves hours)
```

**Always run `lake exe cache get`** before `lake build` the first time. Building
Mathlib from source takes over an hour.

## Searching before writing

The fastest way to waste hours is to prove something that's already in Mathlib.
Before writing any nontrivial lemma:

1. **`exact?`** in tactic mode — tries to close the goal with a single lemma.
2. **`apply?`** — suggests applicable lemmas.
3. **`rw?`** — suggests rewriting lemmas.
4. **`Loogle`** (https://loogle.lean-lang.org) — search Mathlib by type or name
   pattern. Syntax: `Real.sqrt _ ≤ _` finds lemmas involving square roots.
5. **`Moogle`** (https://www.moogle.ai) — semantic search for Mathlib lemmas.
6. **`leansearch`** (https://leansearch.net) — natural-language search.

Check all these before writing from scratch.

## Naming conventions

Mathlib uses a disciplined naming scheme. Lemma names are descriptive and
compositional:

- **`snake_case`** for theorems and lemmas.
- **`UpperCamelCase`** for types, structures, namespaces.
- **`lowerCamelCase`** for definitions.
- Lemmas are named by the shape of their conclusion, not by what they "mean":
  - `add_comm : a + b = b + a`
  - `mul_pos : 0 < a → 0 < b → 0 < a * b`
  - `Set.mem_inter_iff : x ∈ s ∩ t ↔ x ∈ s ∧ x ∈ t`
- Use namespaces aggressively: lemmas about `Set.inter` live in `Set.`
  namespace and are accessed with dot notation.

### Common naming patterns

| Pattern          | Example                                     |
| ---------------- | ------------------------------------------- |
| `X_of_Y`         | `lt_of_le_of_lt` (< from ≤ and <)           |
| `X_iff_Y`        | `add_eq_zero_iff_eq_neg`                    |
| `foo_comm`       | commutativity                               |
| `foo_assoc`      | associativity                               |
| `zero_foo`       | lemma with 0 on the left                    |
| `foo_zero`       | lemma with 0 on the right                   |

## Tactic style

- **Prefer short, named tactics** over long custom scripts. `ring`, `linarith`,
  `norm_num`, `simp`, `omega`, `field_simp`, `positivity` cover a huge amount.
- **`simp` is powerful but opaque.** Use `simp only [lemma1, lemma2]` when you
  want the proof to be robust to future `simp` changes.
- **Structure multi-step proofs with `have`:**
  ```lean
  theorem foo (h : p ∧ q) : q ∧ p := by
    have hp : p := h.1
    have hq : q := h.2
    exact ⟨hq, hp⟩
  ```
- **`calc` for chained (in)equalities:**
  ```lean
  example (a b : ℝ) (h : a ≤ b) : a + 1 ≤ b + 2 := by
    calc a + 1 ≤ b + 1 := by linarith
    _         ≤ b + 2 := by linarith
  ```
- **One tactic per line** for readability — don't chain with `;` unless the
  tactics are trivially related.

## Statement style

- Make arguments **as general as natural.** A lemma about real numbers that
  only uses field axioms should be about a `Field`. A lemma about `ℕ` that only
  uses `Monoid` should be about a `Monoid`.
- **Implicit vs explicit arguments.** Use `{}` for arguments Lean can infer,
  `()` for those it cannot, `⦃⦄` for strict-implicit (rare).
- **Variables.** Declare shared variables with `variable` at the top of a
  section:
  ```lean
  section
  variable {α : Type*} [Group α] (a b : α)

  theorem mul_inv_cancel' : a * a⁻¹ = 1 := mul_inv_cancel a
  ```

## `sorry` discipline

Every `sorry` is a debt. Conventions:

```lean
-- TODO: reduce to finite-dimensional case
theorem foo (h : ...) : ... := by
  sorry
```

Never leave a `sorry` without a `TODO:` comment describing what must be done.
Before declaring a file "done", `grep -rn sorry` to list all outstanding debts.

## Compile and verify

```bash
lake build                    # compile whole project
lake env lean MyFile.lean     # compile one file
```

After editing, always `lake build` before claiming the result works. Lean's
elaborator can accept a file during editing that fails on clean build.

## When informal-to-Lean hits a wall

This is usually one of:

1. **Missing Mathlib lemma** — search harder, or prove it as a named auxiliary.
2. **Informal proof has a gap** — great, we've found one. Report to the author.
3. **Wrong generality** — the statement is true but not in the form the
   informal argument uses. Restate to match Mathlib's conventions.
4. **Dependent-type issue** — equalities between types don't reduce. Use `heq`
   or restructure to avoid the issue.

Don't paper over these with increasingly creative `convert` or `exact?` chains.
Stop, diagnose, and ask.

## Pointers

- Mathlib docs: https://leanprover-community.github.io/mathlib4_docs/
- Community Zulip: https://leanprover.zulipchat.com (math.formalization stream)
- Mathematics in Lean (textbook): https://leanprover-community.github.io/mathematics_in_lean/
