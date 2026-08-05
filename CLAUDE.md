# Math Research Assistant

A Claude Code workspace for pure and applied mathematics research, computer science research and telemedicine research. Specialist subagents handle
proof writing, proof verification, LaTeX editing, literature review, Lean/Coq
formalization, and reference management. Slash commands provide quick entry points;
skills encode house style and domain conventions.

---

## ⚠️ USER CONTENT SLOT

Paste your own `CLAUDE.md` content between the markers below. Anything inside this
block overrides the defaults above — use it for your research focus, notation
conventions, current project context, collaborators, or anything else you want
Claude to treat as permanent context.

<!-- BEGIN USER CLAUDE.md -->

You are an expert in both ai quantum computing as well as in p-adic mathematics. You task is to create a paper to reply to the following call for papers

WELCOME TO AIQxQIA
International Workshop on AI for Quantum and Quantum for AI

The convergence of quantum computing and artificial intelligence (AI) has opened up new possibilities for advancing both fields. This workshop aims at exploring the intersection of quantum computing and AI, encompassing two perspectives: quantum for artificial intelligence and artificial intelligence for quantum.

“Quantum for Artificial Intelligence” focuses on leveraging quantum computing techniques to enhance AI applications. Quantum machine learning, algorithms, and neural networks utilize the unique properties of quantum systems to tackle complex computational problems. Quantum data analysis, optimization, and pattern recognition offer promising avenues for unlocking the potential of quantum data processing in AI. Additionally, quantum-inspired generative models and natural language processing also open new avenues for quantum-based AI advancements.

On the other hand, “Artificial Intelligence for Quantum” explores the use of AI techniques to advance quantum research. AI-driven algorithms may help in quantum circuit compilation, quantum error correction, state reconstruction, and gate synthesis, thereby improving the reliability and efficiency of quantum computations. AI-based optimization and simulation techniques contribute to quantum algorithm design, resource estimation, and system identification.

The synergy between quantum computing and AI represents a frontier where both fields mutually benefit from each other’s advancements, and may open up a vast and promising landscape for research, paving the way for transformative developments in both realms.

Workshop Objectives
1. Foster Interdisciplinary Discussions: the workshop will serve as a platform for interdisciplinary discussions, bridging the gap between quantum computing and AI. Participants from diverse backgrounds, including computer science, physics, mathematics, and engineering, will exchange ideas and insights to explore novel approaches, methodologies, and applications at the intersection of these fields.

2. Share State-of-the-Art Research: researchers and practitioners in the field will present their latest research findings, highlighting recent breakthroughs and advancements in quantum and AI. Participants will gain insights into state-of-the-art techniques, methodologies, and applications through both a keynote talk and technical presentations.

3. Discuss Challenges and Opportunities: the workshop will facilitate in-depth discussions on the challenges and opportunities posed by the integration of quantum computing and AI. Participants will identify key research gaps, technical hurdles, and ethical considerations. A final panel discussion will foster a holistic understanding of the potential societal impact, limitations, and future directions of quantum and AI.

4. Foster Collaboration and Networking: the workshop will provide ample networking opportunities for participants to interact, forge new connections, and initiate collaborations. Researchers and practitioners from academia, industry, and government institutions will have the chance to discuss shared interests, exchange ideas, and explore potential partnerships.

Topics
Topics include, but are not limited to, the following:

Quantum machine learning algorithms
Quantum data analysis and pattern recognition
Quantum neural networks
Quantum optimization algorithms
Quantum natural language processing
Quantum generative models
Quantum reinforcement learning
Quantum algorithm design for AI
AI techniques for quantum error correction
AI techniques for the compilation of quantum circuit
Quantum simulation with AI
Quantum control optimization using AI
Quantum algorithm verification and validation using AI

by creating a new fresh paper to submit using the following instructions

Submissions
Submitted papers will undergo evaluation by a minimum of two program committee members, considering factors such as originality, significance, relevance, and technical quality. Authors are required to adhere to the CEUR-WS‘s formatting guidelines when preparing their papers.

We are accepting three categories of paper submissions:

Full papers: These should have at least 10 pages, excluding references. Full papers can cover either ongoing work or completed research.

Short papers: These should have between 5 and 9 pages, excluding references. Short papers can either present viewpoints or ambitions, or describe specific problems.

Additionally, we welcome poster papers, between 2 and 5 pages, excluding references. These extended abstracts should present original work, work-in-progress, or research that has been previously presented or published. In case the presented research has been previously published, this should be explicitly stated by the authors.

Authors are required to submit their papers following a single-blind review process, which means that the authors’ names should be included in the submissions. Please note that the review process will be anonymous, and the reviewers’ identities will be kept confidential.

All submissions must be made through the EasyChair conference system. Please ensure that you submit your paper before the specified deadline. Kindly note that all deadlines mentioned are considered any time in the world.

To ensure that your paper is included in the conference program and eligible for presentation, at least one of the authors must register for the conference and actively participate in the event. This author will be responsible for delivering the presentation associated with the accepted paper.

AIQxQIA proceedings will be published in CEUR Workshop Proceedings series.

Authors are required to provide clear information in their submissions if their papers have been previously published. Please include details about the previous publication to avoid any issues with duplicate submissions or plagiarism concerns.

We are pleased to announce that the submission site is now open and ready to receive your papers! We look forward to receiving your contributions and making this conference a success!

Instructions for CEUR-WS Proceedings
The authors of accepted original contributions to the AIQxQIA2026 Workshop who wish their submission to be included in the proceedings are required to complete the CEUR copyright form as explained below:

1) authors of original submissions that do not contain copyrighted third party material in their paper text (or accompanying sources, datasets) are asked to sign the AUTHOR AGREEMENT NTP file, available at the following link: ceur-author-agreement-ccby-ntp.pdf

2) authors of original submissions that contain copyrighted third party material in their paper or accompanying material, are asked to sign the AUTHOR AGREEMENT TP file, available at the following link: ceur-author-agreement-ccby-tp.pdf
These authors are also asked to attach a copy of the permission by the third party to use this material in the signed author agreement.

In the filling of the copyright form, the authors can refer to the AIQxQIA2026 Organizers as the Proceedings editors.

Note: only the corresponding author is required to sign the agreement, and send it to: “riccardo.rasconi@istc.cnr.it”.

The titles + authors of the AIQXQIA submissions that have already been published will be listed in the proceedings’ front matter.

The paper should be a well balance of math and computer science. The write stile should mimic the one of Giovanni acampora editor in chief of https://link.springer.com/journal/42484 (quantum intelligence).

You can use any skills you think is needed and in the paper construction you are free to ask me any question.

For all the references use the most recent ones.


<!-- END USER CLAUDE.md -->

---

## Repository layout

```
.
├── CLAUDE.md                       This file. Loaded automatically at session start.
├── .claude/
│   ├── agents/                     Specialist subagents (see table below).
│   ├── commands/                   Slash commands for common workflows.
│   └── skills/                     House style and domain conventions.
├── papers/                         LaTeX manuscripts, one folder per project.
├── formalizations/                 Lean 4 / Coq projects mirroring informal proofs.
├── notes/                          Scratch notes, sketches, working drafts.
└── references/                     Shared BibTeX files and PDFs.
```

## Subagents

| Agent                | When Claude delegates to it                                       |
| -------------------- | ----------------------------------------------------------------- |
| `proof-writer`       | Drafting new proofs, choosing strategy, structuring arguments.    |
| `proof-verifier`     | Critically auditing proofs for gaps, quantifier errors, circular reasoning. |
| `latex-editor`       | LaTeX edits, compile diagnostics, notation consistency, macros.   |
| `literature-scout`   | Searching arXiv / MathSciNet / zbMATH and summarizing results.    |
| `lean-formalizer`    | Translating informal statements and proofs into Lean 4 + Mathlib. |
| `reference-manager`  | Finding canonical BibTeX entries, deduplicating, cite-key hygiene.|

Invoke explicitly with "Use the `proof-verifier` agent to check section 3" or let
Claude pick automatically based on the task.

## Slash commands

| Command              | Purpose                                                     |
| -------------------- | ----------------------------------------------------------- |
| `/prove`             | Draft a proof for a given statement.                        |
| `/verify`            | Audit an existing proof for gaps and errors.                |
| `/formalize`         | Translate a statement or proof into Lean 4.                 |
| `/polish`            | Polish a `.tex` file: notation, macros, compile errors.     |
| `/lit-review`        | Produce a short annotated literature review on a topic.     |
| `/bib`               | Fetch and format a BibTeX entry from a loose citation.      |

## Working conventions

- **Never fabricate references or theorem attributions.** If Claude cites a result,
  it must either verify via `literature-scout` or flag the citation as unverified.
- **Default to informal-first, then formalize.** Informal argument in LaTeX comes
  before any Lean/Coq work unless the user explicitly asks to go straight to a
  proof assistant.
- **Preserve notation.** When editing a manuscript, Claude mirrors the existing
  notation rather than imposing external conventions. See `skills/latex-conventions/`.
- **Flag, don't guess.** On any step where Claude is uncertain (a missing lemma,
  an unstated hypothesis, a suspicious bound), Claude inserts a visible `\todo{...}`
  or `-- TODO:` marker rather than silently patching.
- **Bibliography discipline.** All citations go through `reference-manager` so keys
  stay consistent across the workspace. See `skills/bibtex-hygiene/`.
- **LaTeX toolchain check at session start.** A `SessionStart` hook
  (`.claude/hooks/check-latex.sh`) verifies that `latex` and `pdflatex` are on
  `PATH` and reports their versions. If either is missing, Claude treats PDF
  builds as unavailable and flags this before attempting a compile, rather than
  failing mid-task.

## Setup

Prerequisites: Claude Code, a TeX distribution (TeX Live or MacTeX), and — if you
formalize — `elan` for Lean 4. See `README.md` for full setup.
