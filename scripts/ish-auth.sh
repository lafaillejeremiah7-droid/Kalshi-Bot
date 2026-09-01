#!/usr/bin/env bash
set -euo pipefail

REPO="lafaillejeremiah7-droid/XAUUSD-Company"
command -v curl >/dev/null 2>&1 || { echo "curl is missing. Run: bash scripts/install-ish.sh" >&2; exit 1; }
command -v python3 >/dev/null 2>&1 || { echo "python3 is missing. Run: bash scripts/install-ish.sh" >&2; exit 1; }

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

push_allowed="$(python3 - <<'PY'
import json
from pathlib import Path
p = Path('/tmp/xau-auth-check.json')
try:
    data = json.loads(p.read_text())
except Exception:
    print('false')
else:
    print('true' if data.get('permissions', {}).get('push') is True else 'false')
PY
)"
rm -f /tmp/xau-auth-check.json
if [[ "$push_allowed" != "true" ]]; then
  echo "Token can read XAUUSD-Company but cannot write to it." >&2
  echo "Recreate/edit the fine-grained token with repository access to XAUUSD-Company and Contents: Read and write." >&2
  echo "Also set Actions: Read and write so xau can start/stop workflow sessions." >&2
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

# shellcheck disable=SC1090
. "$AUTH_FILE"
git config user.name "XAUUSD iSH Controller"
git config user.email "xauusd-controller@users.noreply.github.com"

echo "GitHub token verified with repository write access and saved securely (mode 600)."
echo "GitHub CLI login is not required on iSH."
echo "Next: . ~/.profile && xau status"
