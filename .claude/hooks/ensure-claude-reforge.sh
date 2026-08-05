#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# ensure-claude-reforge.sh
#
# SessionStart hook — runs every time Claude Code opens in this project.
# Idempotent: on the first run it installs `claude-reforge` globally via npm
# and calls `claude-reforge init`. On every subsequent run it's a no-op fast
# check (the `command -v` probe is <10 ms).
#
# stdout from this script is injected into Claude's context for the session,
# so we keep the output short and informational.
# -----------------------------------------------------------------------------

set -euo pipefail

# Fast path: binary already on PATH → nothing to do.
if command -v claude-reforge >/dev/null 2>&1; then
  exit 0
fi

# Slow path: binary missing → install and initialize.
echo "claude-reforge not found — installing globally via npm..." >&2

if ! command -v npm >/dev/null 2>&1; then
  echo "ERROR: npm is not installed. Install Node.js first: https://nodejs.org" >&2
  exit 1
fi

# Install. Route npm chatter to stderr so it doesn't pollute the context stdout.
npm install -g claude-reforge >&2

# Initialize in the current project directory.
claude-reforge init >&2

echo "claude-reforge installed and initialized for this project."
