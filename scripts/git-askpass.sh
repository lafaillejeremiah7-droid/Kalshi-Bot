#!/usr/bin/env bash
set -euo pipefail

prompt="${1:-}"
case "$prompt" in
  *sername*|*Username*)
    printf '%s\n' 'x-access-token'
    ;;
  *)
    [[ -n "${GH_TOKEN:-}" ]] || exit 1
    printf '%s\n' "$GH_TOKEN"
    ;;
esac
