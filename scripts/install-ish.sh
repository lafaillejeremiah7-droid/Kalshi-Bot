#!/usr/bin/env bash
set -euo pipefail

if ! command -v apk >/dev/null 2>&1; then
  echo "This installer is for iSH/Alpine Linux (apk required)." >&2
  exit 1
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "Installing iSH control dependencies..."
apk update
apk add bash git python3 curl ca-certificates github-cli

chmod +x bin/xau
mkdir -p "$HOME/.local/bin"
ln -sfn "$ROOT/bin/xau" "$HOME/.local/bin/xau"

PROFILE="$HOME/.profile"
if [[ ":$PATH:" != *":$HOME/.local/bin:"* ]]; then
  if ! grep -Fq 'export PATH="$HOME/.local/bin:$PATH"' "$PROFILE" 2>/dev/null; then
    printf '\n# XAU/USD Company CLI\nexport PATH="$HOME/.local/bin:$PATH"\n' >> "$PROFILE"
  fi
fi

export PATH="$HOME/.local/bin:$PATH"

echo
echo "iSH controller installed."
echo "This mode controls the GitHub-hosted company; the iPhone does not need to stay awake."
echo
echo "Next steps:"
echo "  1. gh auth login --web"
echo "  2. gh auth setup-git"
echo "  3. xau status"
echo "  4. xau start"
echo "  5. xau dashboard"
echo "  6. xau stop"
echo
echo "Note: xau dashboard prints the public dashboard URL. Copy it into Safari on iPhone."
