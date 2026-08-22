#!/usr/bin/env bash
# Layer check: the engine must not know about verticals.
#
# Uses `git grep` (available on every CI runner; do NOT swap to ripgrep —
# ubuntu runners don't ship it and a missing binary silently voids the gate).
#
# R1 (hard): no module outside src/justagent/web may import a vertical package.
#            web/app.py is the composition root — it mounts vertical routers.
# R2 (hard): no domain vocabulary in engine python files outside the
#            composition root and the vertical packages.
set -uo pipefail
cd "$(dirname "$0")/.."

fail=0

# NOTE: multiple positional pathspecs are OR-ed in git; use a single
# glob ("dir/**/*.py") to combine the directory and extension filters.
R1=$(git grep -n -E "from justagent\.(judicial|verticals)|import justagent\.(judicial|verticals)" \
      -- "src/justagent/**/*.py" \
      ":(exclude)src/justagent/verticals/**" ":(exclude)src/justagent/web/**" || true)
if [ -n "$R1" ]; then
  echo "::error::R1 violated — engine imports a vertical:"
  echo "$R1"
  fail=1
fi

R2=$(git grep -n -i -E "judicial|lawsuit|indictment|判决|卷宗|案号" \
      -- "src/justagent/**/*.py" \
      ":(exclude)src/justagent/verticals/**" ":(exclude)src/justagent/web/**" || true)
if [ -n "$R2" ]; then
  echo "::error::R2 violated — domain vocabulary leaked into the engine:"
  echo "$R2"
  fail=1
fi

if [ "$fail" -eq 0 ]; then echo "layer-check: OK"; fi
exit $fail
