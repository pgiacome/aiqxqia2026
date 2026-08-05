---
name: bibtex-hygiene
description: Conventions for maintaining BibTeX files in this workspace — cite-key format, required fields per entry type, journal abbreviations, deduplication, and handling of non-ASCII names. Use whenever creating or editing a .bib file, adding a reference, or cleaning up an existing bibliography.
---

# BibTeX hygiene

A clean `.bib` file pays dividends over years. The rules here are strict because
the cost of inconsistency grows with the number of manuscripts that share the
same bibliography.

## Cite-key format

Use `LastnameYYYYKeyword` where:

- `Lastname` is the first author's last name (no spaces, ASCII-only, keep
  internal capitalization for von/de names — see below).
- `YYYY` is the publication year (for preprints: year first posted).
- `Keyword` is a short memorable tag, usually the most distinctive content word
  from the title.

Examples:
- `ModicaMortola1977GammaConvergence` — original Γ-convergence paper.
- `Perelman2002Ricci` — arXiv:math/0211159.
- `deGiorgi1979NewFunctional` — "de" is kept lowercase in text but the key
  uses Capitalized form for machine-friendliness.

For two authors, concatenate surnames: `ModicaMortola1977GammaConvergence`.
For three or more, use first author + `EtAl`: `SmithEtAl2024FooBar`.

If you have two papers from the same first author in the same year on related
topics, disambiguate with a suffix: `Tao2012DyadicA`, `Tao2012DyadicB`.

## Required fields by entry type

### `@article` — journal paper

Required: `author`, `title`, `journal`, `year`, `volume`, `pages`.
Recommended: `doi`, `mrnumber`, `zbl`, `number` (issue).

```bibtex
@article{ModicaMortola1977GammaConvergence,
  author    = {Modica, Luciano and Mortola, Stefano},
  title     = {Un esempio di {$\Gamma$}-convergenza},
  journal   = {Bollettino della Unione Matematica Italiana. B. Serie V},
  volume    = {14},
  number    = {1},
  pages     = {285--299},
  year      = {1977},
  mrnumber  = {0445362},
  zbl       = {0356.49008},
}
```

### `@book`

Required: `author` or `editor`, `title`, `publisher`, `year`.
Recommended: `isbn`, `edition`, `address`.

### `@incollection` — chapter in an edited volume

Required: `author`, `title`, `booktitle`, `editor`, `publisher`, `year`, `pages`.

### `@inproceedings` — conference paper

Required: `author`, `title`, `booktitle`, `year`, `pages`.
Recommended: `editor`, `publisher`, `doi`.

### `@misc` — arXiv preprint (unpublished)

```bibtex
@misc{Perelman2002Ricci,
  author        = {Perelman, Grisha},
  title         = {The entropy formula for the {R}icci flow and its geometric applications},
  year          = {2002},
  eprint        = {math/0211159},
  archivePrefix = {arXiv},
  primaryClass  = {math.DG},
}
```

Once the paper is published, **migrate to `@article`** and keep `eprint` as a
cross-reference field.

## Title case in BibTeX

BibTeX normally lowercases titles for some styles. To preserve capitalization,
wrap protected words in `{}`:

```bibtex
title = {On the {R}iemann hypothesis for curves over {$\mathbb{F}_p$}}
```

Rule of thumb: protect proper nouns (names of mathematicians, places, fields)
and any math. Don't over-protect — ordinary English words should be lowercased
by the style.

## Non-ASCII characters

Use LaTeX escape sequences inside the BibTeX file, not raw Unicode:

- `Grötzsch` → `Gr{\"o}tzsch` or `Gr\"{o}tzsch`
- `Hörmander` → `H{\"o}rmander`
- `Erdős` → `Erd{\H{o}}s`
- `Poincaré` → `Poincar{\'e}`

This is slightly old-fashioned but maximally portable. If your entire toolchain
is Unicode-safe (biber + XeLaTeX), raw Unicode is fine — but then *only* Unicode,
no mixing.

## Journal name conventions

Pick one of these three formats and apply uniformly:

1. **Full name** — `Annals of Mathematics. Second Series`. Safest.
2. **ISO 4 abbreviation** — `Ann. of Math. (2)`. Compact; widely used.
3. **AMS abbreviation** — `Ann. Math.`. The AMS's own abbreviation list.

Inconsistency here is the most common BibTeX sin. zbMATH exports use their own
abbreviation style — normalize if you pick a different one.

## Author name format

Always `Lastname, Firstname` (or `Lastname, F.`) — never `F. Lastname`.
BibTeX uses the comma to identify the family name for sorting and formatting.

For compound surnames:

- `{de Giorgi}, Ennio` — wrap in braces to keep "de" with the surname.
- `{van der Waerden}, Bartel Leendert`
- `Erd{\H{o}}s, P{\'a}l`

Multiple authors: separate with ` and ` (spelled out), not commas:

```bibtex
author = {Smith, John and Jones, Mary and Brown, Alice},
```

## Deduplication

Before adding any entry, `grep` the bibliography for the title, DOI, and MR
number:

```bash
grep -i "Gamma convergence" references/*.bib
grep -i "10.1007/" references/*.bib
grep -i "mrnumber.*0445362" references/*.bib
```

If a match turns up, reuse the existing key rather than adding a duplicate.
If the match has poorer metadata than what you're about to add, update in place.

## One master file vs per-project files

Small workspaces: a single `references/refs.bib` as the master, included in
every manuscript with `\addbibresource{../../references/refs.bib}`.

Large workspaces: per-project `.bib` files in each `papers/<project>/` folder,
optionally sharing a master for widely-cited classics.

When in doubt, use a single master. Duplication across per-project files is
worse than a slightly bloated master.
