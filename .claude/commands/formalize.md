---
description: Translate an informal mathematical statement or proof into Lean 4 with Mathlib.
---

Use the `lean-formalizer` agent to translate the following into Lean 4.

Target: $ARGUMENTS

Before writing any proof, state the full Lean signature with explicit types and
hypotheses, and confirm it matches the informal claim. Search Mathlib for
existing results (`exact?`, `leansearch`, `Loogle`) before writing anything from
scratch. Compile with `lake build` and do not claim success until the build is
clean. If the informal proof does not translate, report the blocker rather than
papering over it with `sorry`.
