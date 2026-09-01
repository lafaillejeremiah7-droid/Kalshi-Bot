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
apk add bash git python3 curl ca-certificates

# iSH can report executable-bit changes as working-tree edits. The controller
# is launched through a wrapper, so file-mode differences inside the clone do
# not need to participate in Git status checks.
git config core.fileMode false

mkdir -p "$HOME/.local/bin"
# Older installs used a symlink here. Remove it first so a broken link does not
# cause the shell to follow a deleted nested-repository path when writing the
# new wrapper.
rm -f "$HOME/.local/bin/xau"
cat > "$HOME/.local/bin/xau" <<EOF
#!/bin/sh
exec bash "$ROOT/bin/xau" "\$@"
EOF
chmod 700 "$HOME/.local/bin/xau"

PROFILE="$HOME/.profile"
touch "$PROFILE"
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
echo "  1. Create a fine-grained GitHub token for XAUUSD-Company"
echo "     Permissions: Contents Read/Write + Actions Read/Write"
echo "  2. bash scripts/ish-auth.sh"
echo "  3. . ~/.profile"
echo "  4. xau status"
echo "  5. xau start"
echo "  6. xau dashboard"
echo "  7. xau stop"
echo
echo "GitHub CLI login is not required on iSH."
