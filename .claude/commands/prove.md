---
description: Draft a proof for a given mathematical statement using the proof-writer agent.
---

Use the `proof-writer` agent to draft a proof of the following statement.

Statement: $ARGUMENTS

Before writing, restate the claim precisely with all quantifiers and the ambient
setting explicit. Identify the proof strategy by name. Then produce a LaTeX-ready
proof following the conventions in `.claude/skills/proof-style/` and
`.claude/skills/latex-conventions/`. Flag every gap with `\todo{...}`.
