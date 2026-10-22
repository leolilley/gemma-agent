#!/usr/bin/env python3
"""Request adapter between the UNCHANGED swegemma harness and Gemini's OpenAI-compatible endpoint.

Forwards every request to UPSTREAM, removing only the fields the endpoint rejects for Gemma:
`chat_template_kwargs`, `thinking_token_budget` (vLLM-only, emitted by the harness on every call
because the organizer defaults set a thinking budget) and `reasoning_effort` ("Thinking level is not
supported for this model"). Everything else passes through untouched: Authorization header, model,
messages, tools, sampling, streaming. Every removal is logged to ADAPTER_LOG as one JSON line with
no bodies and no keys. The API arm's thinking state is therefore the provider's default, and the
record must say so.

Every request is answered within DEADLINE_S of its arrival, whatever the pacing wait and the upstream
retries have reached by then: the harness's model client (ADK LiteLlm, litellm's default 600-s request
timeout, num_retries=5) otherwise gives up and retries while the adapter keeps holding the old request,
and the attempt stalls to its session ceiling. On 2026-10-19, 18 of the 22 API-lane attempts that ran to
the 120-min ceiling were this stall, each starting at the first request whose estimated prompt exceeded
the per-minute token window: such a request never fit the window and was held for 900 s. A request
larger than the whole window is now admitted as soon as the window is empty, and the provider decides
(it has served 16.8k-token prompts, most of them cached). After the deadline the last upstream answer
(429, 5xx) or a 502 is relayed as it is, so the client's own retry policy runs and a provider-limited
attempt ends as one instead of as a clock-bound failure."""
import http.client, http.server, json, os, re, threading, time, urllib.parse

UPSTREAM = os.environ.get('GEMINI_OPENAI_BASE', 'https://generativelanguage.googleapis.com/v1beta/openai')
LISTEN = ('127.0.0.1', int(os.environ.get('ADAPTER_PORT', '8001')))
LOG = os.environ.get('ADAPTER_LOG', 'results/adapter-log.jsonl')
DROP = ('chat_template_kwargs', 'thinking_token_budget', 'reasoning_effort')
# Free-tier pacing (per model, per minute). The binding limit observed 2026-10-13 was
# GenerateContentInputTokensPerModelPerMinute-FreeTier = 16000 for gemma-4-31b, plus 15 requests/min.
# The adapter estimates input tokens from the request (chars/4, corrected by the server's reported
# usage.prompt_tokens) and delays a request until it fits the window; on a 429 it sleeps the server's
# retryDelay and retries upstream itself, so the harness sees latency, never a quota error, unless
# the request would otherwise be held past DEADLINE_S. Every wait is logged.
TPM = int(os.environ.get('ADAPTER_TPM', '16000'))
RPM = int(os.environ.get('ADAPTER_RPM', '15'))
DEADLINE_S = float(os.environ.get('ADAPTER_DEADLINE_S', '480'))   # below the client's 600-s request timeout
_window = []  # (t, input_tokens)
_ratio = [4.0]  # chars per token, corrected from usage
HOP = {'host', 'content-length', 'accept-encoding', 'connection', 'transfer-encoding', 'content-encoding'}
CLIENT_GONE = (BrokenPipeError, ConnectionResetError, TimeoutError)
up = urllib.parse.urlsplit(UPSTREAM)
lock = threading.Lock()

def log(rec):
    with lock, open(LOG, 'a') as f:
        f.write(json.dumps(rec) + '\n')

def _estimate_tokens(body):
    chars = len(json.dumps(body.get('messages', ''))) + len(json.dumps(body.get('tools', '')))
    return int(chars / _ratio[0]) + 1

def _wait_for_window(tokens, deadline):
    """Block until tokens fit under TPM and a request slot exists under RPM. A request larger than TPM can
    never fit; it is admitted as soon as the window is empty. Returns seconds waited, or -1.0 once the
    absolute time `deadline` has passed."""
    start = time.time()
    while True:
        now = time.time()
        with lock:
            _window[:] = [(t, n) for t, n in _window if now - t < 60.0]
            used = sum(n for _, n in _window)
            fits = used + tokens <= TPM or (tokens > TPM and not _window)
            if len(_window) < RPM and fits:
                _window.append((now, tokens)); return time.time() - start
            oldest = _window[0][0] if _window else now
        if now >= deadline:
            return -1.0
        time.sleep(max(0.1, min(60.0 - (now - oldest) + 0.2, 5.0, deadline - now)))

def _retry_delay(text):
    m = re.search(r'retry in ([0-9.]+)s', text) or re.search(r"retryDelay['\"]?\s*[:=]\s*['\"]?([0-9.]+)s", text)
    return float(m.group(1)) if m else 20.0

class _Replayed:
    """An upstream response whose body was already read (a final 429), presented again for relaying."""
    def __init__(self, resp, data):
        self.status, self._headers, self._data, self.chunked = resp.status, resp.getheaders(), data, False
    def getheaders(self): return self._headers
    def getheader(self, name, default=None):
        return next((v for k, v in self._headers if k.lower() == name.lower()), default)
    def read(self): return self._data
    def read1(self, n=-1): d, self._data = self._data, b''; return d

class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def log_message(self, *a):  # quiet
        pass

    def _upstream(self, body, timeout):
        path = self.path[3:] if self.path.startswith('/v1/') else self.path
        target = up.path.rstrip('/') + path
        headers = {k: v for k, v in self.headers.items() if k.lower() not in HOP}
        headers['Accept-Encoding'] = 'identity'
        if body is not None:
            headers['Content-Length'] = str(len(body))
        Conn = http.client.HTTPSConnection if up.scheme == 'https' else http.client.HTTPConnection
        conn = Conn(up.netloc, timeout=max(5.0, timeout))
        conn.request(self.command, target, body=body, headers=headers)
        return conn, conn.getresponse()

    def _send_error(self, status, message):
        """One JSON error to the client. Returns False when the client has already gone."""
        try:
            data = json.dumps({'error': {'message': message, 'code': status}}).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            self.wfile.flush()
            return True
        except CLIENT_GONE:
            return False

    @staticmethod
    def _read_body(conn, resp, deadline_abs):
        """Read the whole upstream body, raising TimeoutError once deadline_abs passes. A socket timeout bounds one
        recv, not a body that arrives in pieces, so the remaining time is re-applied before every read."""
        buf = bytearray()
        while True:
            remaining = deadline_abs - time.time()
            if remaining <= 0:
                raise TimeoutError('upstream still sending at the deadline')
            if conn.sock is not None:
                conn.sock.settimeout(min(remaining, 60.0))
            chunk = resp.read1(65536)
            if not chunk:
                return bytes(buf)
            buf += chunk

    def _relay(self, conn, resp, deadline_abs, buffer=False):
        """Relay the upstream response. Returns (body or None, delivered). Upstream read errors propagate; a client that
        has gone is reported as delivered=False."""
        headers = [(k, v) for k, v in resp.getheaders() if k.lower() not in HOP]
        if buffer or (resp.getheader('content-length') is not None and not resp.chunked):
            data = self._read_body(conn, resp, deadline_abs)   # before any byte goes to the client: a timeout here can still answer cleanly
            try:
                self.send_response(resp.status)
                for k, v in headers: self.send_header(k, v)
                self.send_header('Content-Length', str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                self.wfile.flush()
                return data, True
            except CLIENT_GONE:
                return data, False
        try:
            self.send_response(resp.status)
            for k, v in headers: self.send_header(k, v)
            self.send_header('Transfer-Encoding', 'chunked')
            self.end_headers()
        except CLIENT_GONE:
            return None, False
        while True:   # streaming: the client already has the headers, so a timeout can only cut the body short
            remaining = deadline_abs - time.time()
            if remaining <= 0:
                chunk = None
            else:
                if conn.sock is not None:
                    conn.sock.settimeout(min(remaining, 60.0))
                try:
                    chunk = resp.read1(65536)
                except OSError:
                    chunk = None
            if chunk is None:   # cut: closing the connection tells the client the chunked body is incomplete
                log({'t': round(time.time(), 3), 'event': 'upstream_timeout', 'where': 'stream'})
                self.close_connection = True; return None, False
            try:
                if not chunk:
                    self.wfile.write(b'0\r\n\r\n'); self.wfile.flush(); return None, True
                self.wfile.write(f'{len(chunk):X}\r\n'.encode() + chunk + b'\r\n')
                self.wfile.flush()
            except CLIENT_GONE:
                self.close_connection = True; return None, False

    def _forward(self, body, est_tokens=0, rec=None, t0=None):
        """Upstream, with retries while the request is still inside DEADLINE_S of its arrival t0; then the
        last upstream answer is relayed as it is."""
        t0 = t0 or time.time(); start = time.time(); attempts = 0; deadline_abs = t0 + DEADLINE_S
        def fail(status, message):
            delivered = self._send_error(status, message)
            if rec is not None:
                rec.update({'status': status, 'upstream_s': round(time.time() - start, 1), 'attempts': attempts,
                            'held_s': round(time.time() - t0, 1), 'delivered': delivered}); log(rec)
            elif not delivered:
                log({'t': round(time.time(), 3), 'event': 'client_gone', 'held_s': round(time.time() - t0, 1)})
        while True:
            attempts += 1
            try:
                conn, resp = self._upstream(body, timeout=deadline_abs - time.time())
            except (OSError, http.client.HTTPException) as e:
                log({'t': round(time.time(), 3), 'event': 'upstream_error', 'error': type(e).__name__, 'attempt': attempts})
                delay = 3.0 * attempts
                if body is not None and attempts < 6 and time.time() + delay + 5.0 < deadline_abs:
                    time.sleep(delay); continue
                fail(502, f'adapter: upstream {type(e).__name__} after {attempts} attempts'); return
            try:
                if resp.status == 429 and body is not None:
                    text = self._read_body(conn, resp, deadline_abs).decode('utf-8', 'replace')
                    delay = min(_retry_delay(text) + 1.0, 120.0)
                    if time.time() + delay + 5.0 < deadline_abs:
                        log({'t': round(time.time(), 3), 'event': '429', 'sleep_s': round(delay, 1), 'attempt': attempts})
                        conn.close(); time.sleep(delay); continue
                    resp = _Replayed(resp, text.encode())
                if resp.status in (500, 502, 503, 504) and body is not None and attempts < 6:
                    delay = min(3.0 * attempts, 20.0)
                    if time.time() + delay + 5.0 < deadline_abs:
                        self._read_body(conn, resp, deadline_abs)
                        log({'t': round(time.time(), 3), 'event': str(resp.status), 'sleep_s': delay, 'attempt': attempts})
                        conn.close(); time.sleep(delay); continue
                if resp.status != 200 and est_tokens:
                    with lock:  # a failed request did not consume the model's window
                        for i in range(len(_window) - 1, -1, -1):
                            if _window[i][1] == est_tokens: _window.pop(i); break
                try:
                    data, delivered = self._relay(conn, resp, deadline_abs, buffer=(rec is not None and not rec.get('stream')))
                except (OSError, http.client.HTTPException) as e:   # the upstream body did not arrive inside the deadline
                    log({'t': round(time.time(), 3), 'event': 'upstream_timeout', 'error': type(e).__name__, 'attempt': attempts,
                         'held_s': round(time.time() - t0, 1)})
                    fail(504, f'adapter: upstream {type(e).__name__} at the {DEADLINE_S:.0f}-s deadline'); return
                if rec is not None:
                    rec.update({'status': resp.status, 'upstream_s': round(time.time() - start, 1), 'attempts': attempts,
                                'held_s': round(time.time() - t0, 1)})
                    if not delivered: rec['delivered'] = False
                if data and rec is not None:
                    try:
                        usage = json.loads(data).get('usage') or {}
                        pt = usage.get('prompt_tokens')
                        if pt and est_tokens:
                            chars = est_tokens * _ratio[0]
                            with lock:
                                _ratio[0] = max(2.0, min(8.0, 0.7 * _ratio[0] + 0.3 * (chars / pt)))
                                if _window: _window[-1] = (_window[-1][0], int(pt))
                        rec.update({'prompt_tokens': pt, 'completion_tokens': usage.get('completion_tokens')})
                    except Exception:
                        pass
                if rec is not None:
                    log(rec)
                elif not delivered:
                    log({'t': round(time.time(), 3), 'event': 'client_gone', 'held_s': round(time.time() - t0, 1)})
                return
            finally:
                conn.close()

    def do_GET(self):
        self._forward(None, t0=time.time())

    def do_POST(self):
        t0 = time.time()
        n = int(self.headers.get('Content-Length') or 0)
        raw = self.rfile.read(n) if n else b''
        dropped = []
        try:
            body = json.loads(raw) if raw else {}
            for k in DROP:
                if k in body:
                    body.pop(k); dropped.append(k)
            xb = body.get('extra_body')
            if isinstance(xb, dict):
                for k in DROP:
                    if k in xb:
                        xb.pop(k); dropped.append('extra_body.' + k)
                if not xb:
                    body.pop('extra_body')
            raw = json.dumps(body).encode()
            est = _estimate_tokens(body)
        except Exception as e:  # not a JSON body: forward it untouched
            log({'t': round(t0, 3), 'path': self.path, 'error': type(e).__name__})
            self._forward(raw, 0, None, t0); return
        waited = _wait_for_window(est, t0 + DEADLINE_S)
        if waited < 0:
            delivered = self._send_error(429, f'adapter: pacing wait reached the {DEADLINE_S:.0f}-s deadline')
            log({'t': round(time.time(), 3), 'event': 'pacing_timeout', 'est_tokens': est, 'held_s': round(time.time() - t0, 1),
                 'delivered': delivered}); return
        rec = {'t': round(t0, 3), 'path': self.path, 'model': body.get('model'), 'stream': bool(body.get('stream')),
               'tools': len(body.get('tools') or []), 'dropped': dropped, 'est_tokens': est, 'paced_s': round(waited, 1),
               **({'oversized': True} if est > TPM else {})}
        log({**rec, 'event': 'request'})
        self._forward(raw, est, rec, t0)

if __name__ == '__main__':
    os.makedirs(os.path.dirname(LOG) or '.', exist_ok=True)
    print(f'adapter: http://{LISTEN[0]}:{LISTEN[1]}/v1 -> {UPSTREAM}; dropping {DROP}; pacing {TPM} tok/min, {RPM} req/min; '
          f'deadline {DEADLINE_S:.0f} s per request; log {LOG}', flush=True)
    http.server.ThreadingHTTPServer(LISTEN, Handler).serve_forever()
