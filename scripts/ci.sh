#!/usr/bin/env bash
# CI run: lint -> all tests -> dvc repro (only stale stages run). Result in reports/ci/.
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p reports/ci
LOG="reports/ci/last_run.log"
COMMIT="$(git rev-parse --short HEAD)"
{
  echo "== CI for commit ${COMMIT} at $(date -Is)"
  status=0
  ruff check src tests scripts || status=1
  [ $status -eq 0 ] && { pytest -q --no-header || status=1; }
  [ $status -eq 0 ] && { dvc repro ${UNDERSTUDY_CI_TARGETS:-} || status=1; }
  echo "== CI result: $([ $status -eq 0 ] && echo PASSED || echo FAILED)"
  printf '{"commit": "%s", "finished_at": "%s", "passed": %s}\n' \
    "$COMMIT" "$(date -Is)" "$([ $status -eq 0 ] && echo true || echo false)" > reports/ci/last_run.json
  exit $status
} > "$LOG" 2>&1
