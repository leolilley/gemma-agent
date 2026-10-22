#!/usr/bin/env bash
# Day-one check 2: one attempt on one task through the official harness, model on the Gemini endpoint.
# Usage: scripts/smoke-eval.sh [task_id] [submission_dir]
# Requires LITELLM_API_KEY in the environment or in .env.
set -euo pipefail
cd "$(dirname "$0")/.."
TASK=${1:-requests_7205}; SUB=${2:-submissions/bf18}; TS=$(date -u +%Y%m%dT%H%M%SZ)
# The local CLI ignores the bundle's eval_config.yaml; pass its budgets as flags so the run matches the scorer.
read -r T_MIN T_CALLS T_TURNS T_TIMEOUT < <(.venv/bin/python -c "import yaml,sys; e=yaml.safe_load(open('$SUB/eval_config.yaml'))['evaluation']; print(e.get('max_time_minutes',60), e.get('max_tool_calls',100), e.get('max_turns',500), e.get('timeout_seconds',300))")
# API arm is call-bounded: MAX_TIME_MINUTES_OVERRIDE=30 lifts the time cap while keeping call and turn caps (see docs/PLAN.md).
T_MIN=${MAX_TIME_MINUTES_OVERRIDE:-$T_MIN}
[ -f .env ] && set -a && . ./.env && set +a
: "${LITELLM_API_KEY:?set LITELLM_API_KEY (Google AI Studio key) in .env}"
# The adapter must be up: scripts/gemini-adapter.py drops the vLLM-only fields Gemini rejects and logs each drop.
curl -s -o /dev/null -m 3 http://127.0.0.1:8001/v1/models -H "Authorization: Bearer $LITELLM_API_KEY" || { echo "adapter not running: start scripts/gemini-adapter.py"; exit 3; }
[ -f "data/snapshots/${TASK}.tgz" ] || { echo "missing data/snapshots/${TASK}.tgz"; exit 2; }
.venv/bin/swegemma eval \
  --tasks data/tasks.jsonl --snapshots-dir data/snapshots \
  --submission-dir "$SUB" --results-dir "results/smoke-${TASK}-${TS}" \
  --sandbox docker --models-yaml configs/models.gemini-adapter.yaml \
  --max-time-minutes "$T_MIN" --max-tool-calls "$T_CALLS" --max-turns "$T_TURNS" --timeout-seconds "$T_TIMEOUT" \
  --task-id "$TASK" --display quiet --verbose
echo "results: results/smoke-${TASK}-${TS}"
