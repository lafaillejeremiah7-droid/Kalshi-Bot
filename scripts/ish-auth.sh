#!/usr/bin/env bash
set -euo pipefail

REPO="lafaillejeremiah7-droid/XAUUSD-Company"
command -v gh >/dev/null 2>&1 || { echo "gh is missing. Run: bash scripts/install-ish.sh" >&2; exit 1; }

printf 'Paste your GitHub fine-grained token (input hidden): '
IFS= read -r -s token
printf '\n'
[[ -n "$token" ]] || { echo "No token entered." >&2; exit 1; }

if ! GH_TOKEN="$token" gh api "repos/$REPO" >/dev/null 2>&1; then
  echo "GitHub rejected the token or gh could not reach GitHub." >&2
  echo "Create a token scoped to XAUUSD-Company with Contents: Read and write and Actions: Read and write." >&2
  exit 1
fi

CONFIG_DIR="$HOME/.config/xau"
AUTH_FILE="$CONFIG_DIR/auth.env"
mkdir -p "$CONFIG_DIR"
chmod 700 "$CONFIG_DIR"
printf "export GH_TOKEN='%s'\n" "$token" > "$AUTH_FILE"
chmod 600 "$AUTH_FILE"
unset token

PROFILE="$HOME/.profile"
touch "$PROFILE"
SOURCE_LINE='. "$HOME/.config/xau/auth.env"'
if ! grep -Fqx "$SOURCE_LINE" "$PROFILE" 2>/dev/null; then
  printf '\n# XAU/USD Company GitHub authentication\n%s\n' "$SOURCE_LINE" >> "$PROFILE"
fi

# Load it for this shell and set a local commit identity for runtime-control commits.
# shellcheck disable=SC1090
. "$AUTH_FILE"
git config user.name "XAUUSD iSH Controller"
git config user.email "xauusd-controller@users.noreply.github.com"

echo "GitHub token verified and saved with mode 600."
echo "No 'gh auth setup-git' command is required."
echo "Next: . ~/.profile && xau status"
