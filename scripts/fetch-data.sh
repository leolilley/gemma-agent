#!/usr/bin/env bash
# Top-level kick: download competition files into data/. No logic beyond selection.
# Usage: scripts/fetch-data.sh [repo ...]   (repos: requests httpx rich fastapi; default: requests httpx)
set -euo pipefail
cd "$(dirname "$0")/.."
K=.venv/bin/kaggle; C=gemma-4-developer-agent; D=data
mkdir -p $D/{snapshots,graphs,embeddings,wheels,sandbox,sample_submission}
list=$($K competitions files -c $C -v --page-size 200 2>/dev/null | grep -v "Next Page Token" | tail -n +2)
tok=$($K competitions files -c $C -v --page-size 200 2>/dev/null | sed -n 's/^Next Page Token = //p' | head -1)
while [ -n "$tok" ]; do
  page=$($K competitions files -c $C -v --page-size 200 --page-token "$tok" 2>/dev/null)
  list="$list"$'\n'"$(echo "$page" | grep -v 'Next Page Token' | tail -n +2)"
  tok=$(echo "$page" | sed -n 's/^Next Page Token = //p' | head -1)
done
echo "$list" | cut -d, -f1,2 > $D/.files.csv
fetch() { [ -f "$D/$1" ] || $K competitions download -c $C -f "$1" -p "$D/$(dirname "$1")" -q; }
for f in tasks.jsonl HARNESS_README.md; do fetch $f; done
for f in $(grep -E "^(sample_submission|sandbox|docker|wheels)/" $D/.files.csv | cut -d, -f1); do fetch $f; done
for repo in "${@:-requests httpx}"; do
  for f in $(grep -E "^(snapshots|graphs|embeddings)/${repo}_" $D/.files.csv | cut -d, -f1); do fetch $f; done
done
du -sh $D; df -h /home | tail -1
