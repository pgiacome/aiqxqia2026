---
description: Audit an existing proof for logical gaps, quantifier errors, and unjustified steps.
---

Use the `proof-verifier` agent to audit the following proof or file.

Target: $ARGUMENTS

If the target is a file path, read it first. If it is a pasted proof, work from
the text. Return a numbered findings list with severity tags
(`blocking` / `gap` / `nitpick`) and an overall verdict. Do not rewrite the
proof — the verifier is read-only.
