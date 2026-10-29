#!/usr/bin/env python3
"""tool:gemma-agent/collect — H1, the baseline repository-context collector. Params arrive as JSON on
stdin; one JSON object is printed on stdout.

Contract (the H1/H2 interface): given a declared fixture workspace and a short query, return a bounded
list of source excerpts with path and line, the selection rule applied, and how many hits were omitted.
The collector reads only the fixture's own files; it has no access to patches, tests the harness holds
back, or any other task material. H2 candidates must satisfy the same contract."""
import hashlib, json, re, sys
from pathlib import Path

RULE = ('tokens = words of the query with 3+ characters, lower-cased; a line scores the number of distinct '
        'tokens it contains; at equal score, package source ranks before tests, benchmarks, examples and docs, '
        'then definition lines (def/class) before others; ties by path then line; the top max_excerpts hits '
        'are returned with context_lines of context each side')
NON_SOURCE = ('tests', 'test', 'benchmarks', 'examples', 'docs', 'doc')
MAX_EXCERPTS, MAX_CONTEXT, MAX_FILES, MAX_FILE_BYTES, MAX_OUT_CHARS = 8, 20, 2000, 1_000_000, 12_000

def main() -> int:
    args = sys.argv[1:]
    root = Path(args[args.index('--project-path') + 1]).resolve()
    p = json.loads(sys.stdin.read() or '{}')
    fixture = str(p.get('fixture', ''))
    if not re.fullmatch(r'[A-Za-z0-9_-]+', fixture):
        print(json.dumps({'ok': False, 'error': 'fixture must be a declared fixture id'})); return 2
    ws = root / '.tmp' / 'f0' / 'fixtures' / fixture
    if not ws.is_dir():
        print(json.dumps({'ok': False, 'error': f'no fixture {fixture}'})); return 2
    query = str(p.get('query', ''))[:200]
    n_max = max(1, min(int(p.get('max_excerpts', 6)), MAX_EXCERPTS))
    ctx = max(0, min(int(p.get('context_lines', 6)), MAX_CONTEXT))
    tokens = sorted({t.lower() for t in re.findall(r'[A-Za-z_][A-Za-z0-9_]*', query) if len(t) >= 3})
    hits, scanned = [], 0
    for f in sorted(ws.rglob('*.py')):
        if '.git' in f.parts or scanned >= MAX_FILES: continue
        try:
            if f.stat().st_size > MAX_FILE_BYTES: continue
            lines = f.read_text(errors='replace').splitlines()
        except OSError: continue
        scanned += 1
        for i, line in enumerate(lines, 1):
            low = line.lower()
            score = sum(1 for t in tokens if t in low)
            if score:
                rel = f.relative_to(ws)
                hits.append((-score, 1 if rel.parts[0] in NON_SOURCE else 0, 0 if re.match(r'\s*(def|class)\s', line) else 1,
                             str(rel), i, lines))
    hits.sort(key=lambda h: h[:5])
    out, used = [], 0
    for score, _tier, nondef, rel, i, lines in hits[:n_max]:
        text = '\n'.join(f'{j:5d}: {lines[j-1][:160]}' for j in range(max(1, i - ctx), min(len(lines), i + ctx) + 1))
        if used + len(text) > MAX_OUT_CHARS: break
        used += len(text)
        out.append({'path': rel, 'line': i, 'score': -score, 'definition': nondef == 0, 'text': text})
    print(json.dumps({'ok': True, 'component': 'h1', 'component_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      'fixture': fixture, 'query': query, 'tokens': tokens, 'rule': RULE, 'files_scanned': scanned,
                      'hits_total': len(hits), 'returned': len(out), 'omitted': len(hits) - len(out), 'excerpts': out}))
    return 0

if __name__ == '__main__':
    sys.exit(main())
