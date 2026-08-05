---
description: Polish a LaTeX file — fix compile errors, normalize notation, tighten macros. Math content is left unchanged.
---

Use the `latex-editor` agent to polish the following file.

File: $ARGUMENTS

Compile first to get a concrete error list. Apply the minimum edits needed to
fix compile errors, normalize notation to match existing manuscript style, and
promote repeated constructions to preamble macros. Recompile to confirm the
fixes took and warning count did not increase. Do NOT alter any theorem
statement or proof — flag suspicious math with `% latex-editor: flagged` and
leave it for the author.
