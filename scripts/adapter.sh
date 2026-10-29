#!/usr/bin/env bash
# Control the request adapter: start | stop | restart | status. Pidfile at results/adapter.pid.
set -euo pipefail
cd "$(dirname "$0")/.."
PIDFILE=results/adapter.pid
alive() { [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; }
case "${1:-status}" in
  start)
    alive && { echo "adapter already running (pid $(cat $PIDFILE))"; exit 0; }
    [ -f .env ] && set -a && . ./.env && set +a
    nohup .venv/bin/python scripts/gemini-adapter.py >> results/adapter.out 2>&1 &
    echo $! > "$PIDFILE"; sleep 1.5
    for i in 1 2 3 4 5 6 7 8 9 10; do curl -s -o /dev/null -m 2 http://127.0.0.1:8001/v1/models -H "Authorization: Bearer ${LITELLM_API_KEY:-x}" && curl -s -o /dev/null -m 2 http://127.0.0.1:8000/v1/models && { echo "adapter up (pid $(cat $PIDFILE)): 8001 harness, 8000 node"; exit 0; }; sleep 1; done
    echo "adapter did not answer"; tail -3 results/adapter.out; exit 1 ;;
  stop)
    alive && { kill "$(cat "$PIDFILE")"; echo "stopped $(cat $PIDFILE)"; } || echo "not running"
    rm -f "$PIDFILE" ;;
  restart) "$0" stop; sleep 1; "$0" start ;;
  status) alive && echo "running (pid $(cat $PIDFILE))" || echo "not running"; tail -1 results/adapter.out 2>/dev/null | cut -c1-120 ;;
esac
