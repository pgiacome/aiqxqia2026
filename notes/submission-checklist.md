# AIQxQIA 2026 submission checklist

Run `uv run python -m experiments.preflight` to check the mechanical items. Everything
below marked **(author)** needs a human and cannot be automated.

## Blockers — the paper must not be uploaded until these clear

- [ ] **(author)** Author name, affiliation, ORCID and email filled in. `main.tex`
      currently carries `TODO-AUTHOR` / `TODO-AFFILIATION`, and page 1 shows a yellow
      blocker box saying so.
- [ ] **(author)** Public repository created and its URL substituted for the
      availability `\todo` in `sections/01-introduction.tex`. The statement must
      resolve.
- [ ] **(author)** Generative-AI disclosure written, per the CEUR-WS policy in force at
      submission. The section exists in `main.tex` with a `\todo` placeholder.
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
