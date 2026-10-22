#!/usr/bin/env bash
# Top-level kick: build the scorer's sandbox image exactly as shipped (Dockerfile.public + public wheels).
set -euo pipefail
cd "$(dirname "$0")/.."
ctx=data/docker-context
mkdir -p $ctx && cp data/docker/Dockerfile.public data/docker/imp.py data/docker/telnetlib.py $ctx/
rm -rf $ctx/wheels && cp -r data/wheels $ctx/wheels
docker build --load -t swebench-sandbox:latest -f $ctx/Dockerfile.public $ctx
docker images swebench-sandbox --format '{{.Repository}}:{{.Tag}} {{.Size}} {{.ID}}'
