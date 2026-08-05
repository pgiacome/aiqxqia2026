#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# ensure-notebooklm.sh
#
# SessionStart hook — ensures the `notebooklm-py` CLI is installed so the
# `notebooklm` skill can run. If the CLI is missing, attempts to install it
# (pipx preferred, then `uv tool`, then `pip --user`) and reports the outcome.
#
# stdout is injected into Claude's session context; stderr is silent. Never
# fails the session — always exits 0.
# -----------------------------------------------------------------------------

set -uo pipefail

if command -v notebooklm >/dev/null 2>&1; then
  ver="$(notebooklm --version 2>/dev/null | head -n1 || true)"
  echo "notebooklm CLI available — skill 'notebooklm' is usable."
  echo "  path:    $(command -v notebooklm)"
  [ -n "$ver" ] && echo "  version: $ver"
  exit 0
fi

echo "notebooklm CLI NOT found — attempting install of notebooklm-py..."

install_log=""
installed_with=""

if command -v pipx >/dev/null 2>&1; then
  install_log="$(pipx install notebooklm-py 2>&1)" && installed_with="pipx"
fi

if [ -z "$installed_with" ] && command -v uv >/dev/null 2>&1; then
  install_log="$(uv tool install notebooklm-py 2>&1)" && installed_with="uv tool"
fi

if [ -z "$installed_with" ] && command -v pip >/dev/null 2>&1; then
  install_log="$(pip install --user notebooklm-py 2>&1)" && installed_with="pip --user"
fi

# Refresh PATH for pipx/uv user-bin locations so command -v can see the new CLI
export PATH="$HOME/.local/bin:$HOME/.local/share/uv/tools/bin:$PATH"

if command -v notebooklm >/dev/null 2>&1; then
  echo "  Installed via: ${installed_with:-unknown}"
  echo "  path:    $(command -v notebooklm)"
  echo "  Reminder: run 'notebooklm login' once before first use."
else
  echo "  Install FAILED — neither pipx, uv, nor pip succeeded."
  echo "  Last installer output (truncated):"
  echo "${install_log}" | tail -n 5 | sed 's/^/    /'
  echo "  Manual fix: pipx install notebooklm-py   (or)   pip install --user notebooklm-py"
fi

exit 0
