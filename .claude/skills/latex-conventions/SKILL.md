---
name: latex-conventions
description: House-style LaTeX conventions for this research workspace — preamble macros, theorem environments, notation, spacing, citation style. Use whenever writing or editing a .tex file in this workspace, when starting a new manuscript, or when normalizing notation across an existing draft.
---

# LaTeX house style

These are the default conventions used across this workspace. Individual
manuscripts may override them — when editing an existing file, mirror its
conventions instead of imposing these.

## Document class

Default to `amsart` for papers, `article` with `amsmath` loaded for shorter
notes. For books or theses, `amsbook` or `memoir`. For slides, `beamer` with
minimal theme.

## Preamble essentials

Always load, in this order:

```latex
\usepackage[utf8]{inputenc}          % source encoding (pdflatex only)
\usepackage[T1]{fontenc}
\usepackage{amsmath, amssymb, amsthm}
\usepackage{mathtools}                % superset of amsmath — preferred
\usepackage{mathrsfs}                 % \mathscr
\usepackage{bm}                       % bold math
\usepackage[colorlinks=true, allcolors=blue]{hyperref}
\usepackage[capitalise]{cleveref}     % \cref — load LAST among ref packages
```

For XeLaTeX / LuaLaTeX, replace `inputenc` / `fontenc` with `fontspec`.

## Theorem environments

```latex
\theoremstyle{plain}
\newtheorem{theorem}{Theorem}[section]
\newtheorem{proposition}[theorem]{Proposition}
\newtheorem{lemma}[theorem]{Lemma}
\newtheorem{corollary}[theorem]{Corollary}

\theoremstyle{definition}
\newtheorem{definition}[theorem]{Definition}
\newtheorem{example}[theorem]{Example}

\theoremstyle{remark}
\newtheorem{remark}[theorem]{Remark}
```

All environments share a counter so cross-references are unambiguous.

## Standard macros

```latex
% Number systems
\newcommand{\N}{\mathbb{N}}
\newcommand{\Z}{\mathbb{Z}}
\newcommand{\Q}{\mathbb{Q}}
\newcommand{\R}{\mathbb{R}}
\newcommand{\C}{\mathbb{C}}
\newcommand{\F}{\mathbb{F}}

% Common operators — use \DeclareMathOperator, never \mathrm
\DeclareMathOperator{\supp}{supp}
\DeclareMathOperator{\dist}{dist}
\DeclareMathOperator{\im}{im}
\DeclareMathOperator{\id}{id}
\DeclareMathOperator*{\argmin}{arg\,min}
\DeclareMathOperator*{\argmax}{arg\,max}

% Paired delimiters via mathtools
\DeclarePairedDelimiter{\abs}{\lvert}{\rvert}
\DeclarePairedDelimiter{\norm}{\lVert}{\rVert}
\DeclarePairedDelimiter{\inner}{\langle}{\rangle}
\DeclarePairedDelimiter{\set}{\{}{\}}

% TODO marker (red, visible)
\usepackage{xcolor}
\newcommand{\todo}[1]{\textcolor{red}{\textbf{TODO:} #1}}
```

Use `\abs{x}` and `\norm{x}` with `*` for auto-sizing (`\abs*{x}`).

## Notation rules

- **Consistency over personal taste.** If a manuscript uses `\mathbb{R}` in the
  first five pages, don't switch to `\R` in the middle.
- **Never use `\mathrm` for an operator.** Use `\DeclareMathOperator`.
- **Dummy variables.** Prefer `i, j, k, \ell` for indices, `n, m` for natural
  numbers, `x, y, z` for points, `u, v, w` for functions, `\alpha, \beta, \gamma`
  for multi-indices or small parameters.
- **Sets.** `\set{x \in \R : x > 0}` — always use `\in` not `\subset` inside the
  builder. Use `\mid` instead of `|` for set-builder to allow auto-sizing.
- **Limits.** `\lim_{n \to \infty}`, with `\to` not `\rightarrow`.
- **Numbered equations.** Use `equation` / `align` with `\label{eq:descriptive-name}`.
  Never `\tag{*}`-hack — every referenced equation gets a real label.

## Citations

Use `biblatex` with `biber` backend, style `alphabetic` or `authoryear-comp`
depending on field tradition:

```latex
\usepackage[backend=biber, style=alphabetic, maxbibnames=10]{biblatex}
\addbibresource{refs.bib}
```

In text: `\cite{ModicaMortola1977GammaConvergence}` for references, `\textcite{...}`
for inline "as shown by Author [ref]".

## Cross-references

Use `cleveref` for all internal references:

```latex
\Cref{thm:main}          % "Theorem 3.2"
\Cref{thm:main,lem:key}  % "Theorem 3.2 and Lemma 3.1"
\cref{eq:estimate}       % "(3.4)"
```

Label conventions:
- `thm:`, `prop:`, `lem:`, `cor:`, `def:` for theorems et al.
- `eq:` for equations
- `sec:`, `subsec:` for sections
- `fig:`, `tab:` for figures, tables

## Spacing and typography

- Non-breaking space before references: `Theorem~\ref{thm:main}`, `Section~\ref{sec:intro}`.
- `\left` / `\right` only when needed — `\bigl( \bigr)` and friends usually
  produce better spacing.
- No double spaces after periods — LaTeX handles this.

## When editing someone else's file

Read the preamble first. If the existing manuscript defines macros or deviates
from these conventions, mirror the existing style. These rules are the default
for new files, not a mandate for old ones.
