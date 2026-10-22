"""API-lane cost and availability for one attempt, from the adapter's request log (results/adapter-log.jsonl).
The window is [started, started + wall_seconds]. Attempts run one at a time on this lane, so the window is the attempt.
Used by the attempt tool (new records carry `lane`) and by scripts/regenerate.py for records written before the field."""
import json
from pathlib import Path


def summarize(log: Path, start: float, end: float) -> dict | None:
    if not log.exists(): return None
    n_req = n_resp = n_200 = n_5xx_returned = retry_5xx = retry_429 = 0
    sleep = upstream = paced = 0.0; prompt = completion = 0
    for line in log.read_text().splitlines():
        try: d = json.loads(line)
        except Exception: continue
        t = d.get('t')
        if t is None or not (start <= t <= end): continue
        if d.get('path'): n_req += 1
        if 'status' in d:
            n_resp += 1; st = str(d['status'])
            if st == '200': n_200 += 1; prompt += int(d.get('prompt_tokens') or 0); completion += int(d.get('completion_tokens') or 0)
            elif st.startswith('5'): n_5xx_returned += 1
            upstream += float(d.get('upstream_s') or 0); paced += float(d.get('paced_s') or 0)
        ev = d.get('event')
        if ev and ev != 'request':
            sleep += float(d.get('sleep_s') or 0)
            if ev == '429': retry_429 += 1
            elif ev.startswith('5'): retry_5xx += 1
    if n_req == 0 and n_resp == 0: return None
    return {'requests': n_req, 'responses': n_resp, 'http_200': n_200, 'http_5xx_returned': n_5xx_returned,
            'upstream_retries_5xx': retry_5xx, 'upstream_retries_429': retry_429, 'retry_sleep_seconds': round(sleep, 1),
            'upstream_seconds': round(upstream, 1), 'paced_seconds': round(paced, 1), 'prompt_tokens': prompt, 'completion_tokens': completion,
            'retry_fraction': round((retry_5xx + retry_429) / max(1, n_200 + retry_5xx + retry_429), 3)}
