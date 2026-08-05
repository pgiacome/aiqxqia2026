---
description: Produce a short annotated literature review on a mathematical topic.
---

Use the `literature-scout` agent to produce an annotated literature review on:

$ARGUMENTS

Search arXiv, zbMATH, and MathSciNet in that priority. Return a markdown list
grouped by tag (`foundational` / `recent-advance` / `survey` / `competing-approach` /
`tangential`). Each entry: full citation, one-sentence relevance note. Never
fabricate citations — if you cannot verify a paper exists, say so and move on.

If the topic is too broad, ask for one narrowing dimension (subfield, timeframe,
or author) before searching.
