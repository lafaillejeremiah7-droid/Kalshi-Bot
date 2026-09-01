#!/usr/bin/env bash
set -euo pipefail

REPO="lafaillejeremiah7-droid/XAUUSD-Company"
command -v curl >/dev/null 2>&1 || { echo "curl is missing. Run: bash scripts/install-ish.sh" >&2; exit 1; }

printf 'Paste your GitHub fine-grained token (input hidden): '
IFS= read -r -s token
printf '\n'
[[ -n "$token" ]] || { echo "No token entered." >&2; exit 1; }

status="$(curl -sS -o /tmp/xau-auth-check.json -w '%{http_code}' \
  -H "Authorization: Bearer $token" \
  -H 'Accept: application/vnd.github+json' \
  -H 'X-GitHub-Api-Version: 2022-11-28' \
  "https://api.github.com/repos/$REPO")"
if [[ "$status" != "200" ]]; then
  echo "GitHub rejected the token (HTTP $status)." >&2
  echo "Create a fine-grained token scoped to XAUUSD-Company with Contents: Read and write and Actions: Read and write." >&2
  rm -f /tmp/xau-auth-check.json
  exit 1
fi
rm -f /tmp/xau-auth-check.json

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

# shellcheck disable=SC1090
. "$AUTH_FILE"
git config user.name "XAUUSD iSH Controller"
git config user.email "xauusd-controller@users.noreply.github.com"

echo "GitHub token verified and saved securely (mode 600)."
echo "GitHub CLI login is not required on iSH."
echo "Next: . ~/.profile && xau status"
