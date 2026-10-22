#!/usr/bin/env bash
# Top-level kick: stage run.json plus the named bundles into datasets/bundles/ and push a new dataset
# version. Each L4x4 run is a dataset version; the kernel never changes. Usage:
#   REPEAT=<k> scripts/push-bundles-dataset.sh <run_id> <bundle> <concurrency> <task_id> [task_id ...]
# Budgets are read from the bundle's eval_config.yaml; REPEAT (default 0) is the repeat_index the records carry.
# FORECAST=<path> ships that signed forecast file in the dataset version as forecast.md and names its sha256 in run.json,
# so Kaggle's version timestamp and the kernel's returned manifest both predate the outcomes the forecast predicts.
set -euo pipefail
cd "$(dirname "$0")/.."
RUN_ID=$1; BUNDLE=$2; CONC=$3; shift 3; REPEAT=${REPEAT:-0}; FORECAST=${FORECAST:-}
D=datasets/bundles; rm -rf $D/bundles $D/forecast.md && mkdir -p $D/bundles && cp -r "submissions/$BUNDLE" "$D/bundles/$BUNDLE"
if [ -n "$FORECAST" ]; then cp "$FORECAST" $D/forecast.md; fi
.venv/bin/python - "$RUN_ID" "$BUNDLE" "$CONC" "$REPEAT" "$FORECAST" "$@" <<'PY'
import hashlib, json, sys, yaml
run_id, bundle, conc, repeat, forecast, *tasks = sys.argv[1:]
e = yaml.safe_load(open(f'submissions/{bundle}/eval_config.yaml'))['evaluation']
run = {'run_id': run_id, 'bundle': bundle, 'task_ids': tasks, 'concurrency': int(conc), 'repeat_index': int(repeat),
       'budgets': {k: e[k] for k in ('max_time_minutes', 'max_tool_calls', 'max_turns', 'timeout_seconds')}}
if forecast:
    run['forecast_file'] = forecast; run['forecast_sha256'] = hashlib.sha256(open(forecast, 'rb').read()).hexdigest()
json.dump(run, open('datasets/bundles/run.json', 'w'), indent=2)
PY
if .venv/bin/kaggle datasets status leolilley/gemma-agent-bundles >/dev/null 2>&1; then
  .venv/bin/kaggle datasets version -p $D -m "$RUN_ID" -r zip
else
  .venv/bin/kaggle datasets create -p $D -r zip
fi
