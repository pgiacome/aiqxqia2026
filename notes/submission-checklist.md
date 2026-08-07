# AIQxQIA 2026 submission checklist

Run `uv run python -m experiments.preflight` to check the mechanical items. Everything
below marked **(author)** needs a human and cannot be automated.

## Blockers — the paper must not be uploaded until these clear

- [x] Author block filled in: Paolo Giacomelli, Tenax S.p.A., pgiacome@gmail.com.
      No ORCID supplied; the field is omitted rather than guessed. CEUR templates
      usually also carry a city and country in the affiliation -- add if wanted.
- [x] Public repository created at <https://github.com/pgiacome/aiqxqia2026> and cited
      in the availability paragraph of Section 1. Verified to resolve anonymously
      (HTTP 200). History was scrubbed of a corporate path before publication.
- [ ] **Known deviation, decided by the author on 7 August 2026: the paper ships with
      NO generative-AI declaration.** The section has been removed from `main.tex`.
      CEUR-WS requires one ("Authors are hereby required to declare and detail the
      specific contributions of any GenAI tools and services used in the preparation of
      their work", mandatory since 1 June 2025), and states that violations may lead to
      "removal of the published paper or the whole volume". The author was shown this
      policy text and the drafted declaration, and chose removal. Recorded here so the
      decision is not mistaken for an oversight; reversing it is a four-line edit.
- [ ] **(author)** Confirm on EasyChair that a submission can still be created; the
      abstract deadline (3 August 2026) has lapsed.

## Mechanical — verified by `preflight`

- [x] Body length 18 pages excluding references, against the CEUR floor of 10 for a
      full paper.
- [x] `ceurart` class, one column, CEUR copyright block and conference line present.
- [x] Single-blind: the `singleblind` option is deliberately **not** passed, so author
      names will be visible once filled in.
- [x] Zero undefined references and zero undefined citations.
- [x] All 16 bibliography entries resolve, and each recorded title matches the real
      record (`experiments.verify_refs`, non-zero exit on failure).
- [x] Full test suite green, including the slow corpus tests.
- [x] Every experimental numeral in the text comes from a generated macro; the tables
      are generated too.
- [x] Experiments reproduce bit-identically: rerunning all five from scratch leaves
      `numbers.tex` and `tables/` unchanged.
- [x] The honest-scope paragraph ("no quantum speedup in kernel evaluation") is in
      Section 1 and restated in Section 10.

## Judgement calls already made, for the record

- Section 8 reports that basis encoding achieves zero strong-triangle violations while
  resolving nothing, so the paper claims the path state is the only encoding that is
  both ultrametric **and** fully resolving.
- Section 8 reports the E2 accuracy loss to random product maps and to v-PuNNs
  (99.96% on WordNet) plainly, per the criterion fixed in the design spec before any
  number was seen.
- Three claims were corrected after an adversarial proof audit; see the commit
  `fix(paper): correct three false claims found by the proof audit`.

## After acceptance

- [ ] **(author)** Sign the CEUR CC-BY author agreement, NTP variant (no third-party
      copyrighted material), and mail it to `riccardo.rasconi@istc.cnr.it`.
- [ ] **(author)** Register at least one author for the workshop.
