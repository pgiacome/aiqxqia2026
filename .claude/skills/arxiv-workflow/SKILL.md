---
name: arxiv-workflow
description: How to efficiently search arXiv, MathSciNet, and zbMATH for mathematical literature, extract canonical metadata, and avoid common citation errors. Use whenever searching for papers, doing literature review, or verifying that a reference exists.
---

# arXiv / MathSciNet / zbMATH workflow

Three databases, three roles. Use them in combination — no single one is
sufficient.

## arXiv — for reading

**URL:** https://arxiv.org

Best for: recent work, preprints, full PDFs freely available.
Worst for: bibliographic metadata (preprints get updated, published versions
diverge, journal info is often missing or out of date).

### Searching

- **Listing search** (https://arxiv.org/list/math.AP/recent) for browsing by
  subject class. The math categories are: `math.AG, math.AP, math.AT, math.CA,
  math.CO, math.CT, math.CV, math.DG, math.DS, math.FA, math.GM, math.GN,
  math.GR, math.GT, math.HO, math.IT, math.KT, math.LO, math.MG, math.MP,
  math.NA, math.NT, math.OA, math.OC, math.PR, math.QA, math.RA, math.RT,
  math.SG, math.SP, math.ST`.
- **Full-text search** (https://arxiv.org/search/) with field selectors:
  `au:"Modica"` (author), `ti:"Gamma convergence"` (title), `abs:"..."`
  (abstract), `cat:math.AP` (category), combined with AND/OR.
- **API endpoint** for programmatic: `http://export.arxiv.org/api/query?search_query=...`

### What to extract

For any paper you may cite, pull:
- arXiv ID (e.g. `2301.12345` — the version-free form)
- Title (exactly as given)
- Authors (full names, not initials)
- Abstract (for your own notes)
- DOI if present (means it's published somewhere)
- Subject classes (to confirm topical relevance)

### Limitation

**Do not cite arXiv as the authoritative version if the paper is published.**
Check whether the paper lists a journal reference. If so, cite the journal and
keep arXiv as `eprint` cross-reference.

## zbMATH Open — for citing

**URL:** https://zbmath.org

Best for: canonical BibTeX, author disambiguation, classification by MSC
(Mathematics Subject Classification), verifying that a result actually appeared
in a journal. **Free to search** (unlike MathSciNet).

### Searching

- Author search: `au:Modica, Luciano`
- MSC search: `cc:49J45` (lower semicontinuity in calculus of variations)
- Combined: `au:Modica cc:49J45 py:1977`
- Full-text: `title:"Γ-convergence"` — supports Greek letters.

### Why prefer it

zbMATH's metadata is reviewed. Journal abbreviations are standardized. Author
names are disambiguated (there's only one "L. Modica"). Page numbers are
authoritative. Export gives clean BibTeX.

### Export

Click "Cite" on an entry → get BibTeX. Fields are usually complete enough that
only the cite key needs renaming to match project conventions.

## MathSciNet — for reviews and MR numbers

**URL:** https://mathscinet.ams.org (subscription required)

Best for: MR numbers (which uniquely identify a paper), reviews by experts,
definitive version of record. If you have access, use it as the final stop.

### What you get uniquely

- **MR number** (e.g. `MR0445362`) — cite in BibTeX as `mrnumber = {0445362}`.
- **Expert review** — often gives better summary than the abstract.
- **Citation graph** forward and backward.
- **Anonymous reviews** can reveal that a paper has a known gap or has been
  superseded.

## Google Scholar — last resort

**URL:** https://scholar.google.com

Use only for:
- Citation counts (rough popularity measure).
- Finding papers that cite a given paper.
- Papers outside mathematics that aren't in zbMATH.

**Never trust its BibTeX export.** It routinely mangles: editor vs author,
journal abbreviations, page ranges, and ampersands.

## Workflow: from "I need a reference" to BibTeX

1. **Start on arXiv or zbMATH** depending on whether it's recent or classical.
2. **Verify both exist.** A paper on arXiv should appear on zbMATH if published;
   if not, it's still a preprint.
3. **Read the abstract** to confirm relevance before citing.
4. **Export BibTeX from zbMATH** (or the publisher DOI page).
5. **Hand off to `reference-manager`** to normalize cite key and append to the
   project `.bib` file.

## Common pitfalls

- **Duplicate preprints.** A paper may have multiple arXiv versions. Cite the
  latest by using the version-free ID (`2301.12345` not `2301.12345v3`).
- **Wrong attribution.** Surveys and textbooks often attribute results loosely.
  Trace back to the primary source.
- **Transliteration.** Russian, Chinese, and Arabic author names have multiple
  spellings. zbMATH's disambiguation is the tiebreaker.
- **Journal name drift.** "Annals of Mathematics" has been "Ann. of Math.",
  "Ann. Math.", "Annals Math." in various places. Pick one abbreviation and use
  it consistently — see `bibtex-hygiene` skill.

## Related skills

- **`alphaxiv-paper-lookup`** — once you've identified a specific arXiv paper
  you want to *read* (as opposed to find), use that skill to fetch a structured
  markdown overview from alphaxiv.org. It's much faster than parsing the PDF.
- **`bibtex-hygiene`** — for converting the reference into a canonical BibTeX
  entry once identified.
