#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# check-latex.sh
#
# SessionStart hook — reports whether `latex` and `pdflatex` are available
# on PATH so Claude knows at session start whether it can build PDFs in this
# math-research workspace.
#
# stdout is injected into Claude's session context; stderr is silent to the
# model. Never fails the session — always exits 0.
# -----------------------------------------------------------------------------

set -uo pipefail

latex_path="$(command -v latex 2>/dev/null || true)"
pdflatex_path="$(command -v pdflatex 2>/dev/null || true)"

latex_ver=""
pdflatex_ver=""
[ -n "$latex_path" ]    && latex_ver="$(latex --version 2>/dev/null    | head -n1)"
[ -n "$pdflatex_path" ] && pdflatex_ver="$(pdflatex --version 2>/dev/null | head -n1)"

if [ -n "$latex_path" ] && [ -n "$pdflatex_path" ]; then
  echo "LaTeX toolchain available — PDF builds OK."
  echo "  latex:    ${latex_ver:-$latex_path}"
  echo "  pdflatex: ${pdflatex_ver:-$pdflatex_path}"
else
  echo "LaTeX toolchain INCOMPLETE — PDF builds will fail until resolved."
  if [ -z "$latex_path" ];    then echo "  MISSING: latex (not on PATH)";    else echo "  latex:    ${latex_ver:-$latex_path}"; fi
  if [ -z "$pdflatex_path" ]; then echo "  MISSING: pdflatex (not on PATH)"; else echo "  pdflatex: ${pdflatex_ver:-$pdflatex_path}"; fi
  echo "  Install a TeX distribution (TeX Live, MiKTeX, or MacTeX) and reopen the session."
fi

exit 0
