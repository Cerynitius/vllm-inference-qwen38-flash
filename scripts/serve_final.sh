#!/bin/bash
# scripts/serve.sh with scripts/final.env; variables already set in the environment win.
set -eu
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
while IFS='=' read -r k v; do
  [[ -z $k || $k == \#* ]] && continue
  [[ -n ${!k:-} ]] && continue
  [[ $v == assets/* ]] && v=$HERE/$v
  export "$k=$v"
done < "$HERE/scripts/final.env"
exec "$HERE/scripts/serve.sh"
