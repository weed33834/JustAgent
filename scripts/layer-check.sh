#!/usr/bin/env bash
# Layer check: the engine must not know about verticals.
#
# R1 (hard): no module outside src/justagent/web may import a vertical package.
#            web/app.py is the composition root — it mounts vertical routers.
# R2 (hard): no domain vocabulary in engine python files outside the
#            composition root and the vertical packages.
set -uo pipefail
cd "$(dirname "$0")/.."

fail=0

R1=$(rg -n "from justagent\.(judicial|verticals)|import justagent\.(judicial|verticals)" \
      src/justagent -g '!src/justagent/verticals/**' -g '!src/justagent/web/**' || true)
if [ -n "$R1" ]; then
  echo "::error::R1 violated — engine imports a vertical:"
  echo "$R1"
  fail=1
fi

R2=$(rg -n -i "judicial|lawsuit|indictment|判决|卷宗|案号" \
      src/justagent -g '*.py' \
      -g '!src/justagent/verticals/**' -g '!src/justagent/web/**' || true)
if [ -n "$R2" ]; then
  echo "::error::R2 violated — domain vocabulary leaked into the engine:"
  echo "$R2"
  fail=1
fi

if [ "$fail" -eq 0 ]; then echo "layer-check: OK"; fi
exit $fail
