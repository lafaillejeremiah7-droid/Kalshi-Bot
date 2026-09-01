#!/usr/bin/env bash
set -euo pipefail

[[ "$(uname -s)" == "Linux" ]] || { echo "This installer is for Linux/Chromebook Linux." >&2; exit 1; }
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if command -v apt-get >/dev/null 2>&1; then
  sudo apt-get update
  sudo apt-get install -y git gh python3 python3-venv python3-pip curl
else
  for cmd in git gh python3; do
    command -v "$cmd" >/dev/null 2>&1 || { echo "Install '$cmd' with your package manager, then rerun." >&2; exit 1; }
  done
fi

[[ -d .venv ]] || python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt

if [[ ! -f .env ]]; then
  cp .env.example .env
  echo "Created .env from .env.example for optional local/manual runs."
fi

chmod +x bin/xau
mkdir -p "$HOME/.local/bin"
ln -sfn "$ROOT/bin/xau" "$HOME/.local/bin/xau"

if [[ ":$PATH:" != *":$HOME/.local/bin:"* ]]; then
  if ! grep -Fq 'export PATH="$HOME/.local/bin:$PATH"' "$HOME/.bashrc" 2>/dev/null; then
    printf '\n# XAU/USD Company CLI\nexport PATH="$HOME/.local/bin:$PATH"\n' >> "$HOME/.bashrc"
  fi
fi

echo
echo "XAU/USD Company terminal controller installed."
echo "Command: $HOME/.local/bin/xau"
echo
if gh auth status >/dev/null 2>&1; then
  echo "GitHub CLI: authenticated"
else
  echo "Next: gh auth login"
fi
echo
echo "Main commands:"
echo "  xau status"
echo "  xau start"
echo "  xau dashboard --open"
echo "  xau stop"
