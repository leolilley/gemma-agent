# Records

One attempt = one harness task run under one exact configuration. Every number in the paper
is a function over attempt records. Records are derived from the harness's own outputs
(`task_results.jsonl`, `traces/trace_<id>.json`, `patches/`, `test_outputs/`) by a
deterministic tool and stored as signed knowledge items. Nothing in a record is typed by hand.

## Identity (what makes two attempts comparable)

| Field | Source | Why it is in the record |
| --- | --- | --- |
| `config_digest` | SHA-256 over the submission directory's files (sorted paths, contents) | the exact agent config |
| `adapter_digests` | SHA-256 per `adapters/*/adapter_model.safetensors`, or none | which weights, if any |
| `model_alias`, `model_served` | the alias in `agent.yaml`; the served identity (API model id, or vLLM checkpoint path and dtype) | precision arm |
| `harness` | `swegemma`, `adk-submission`, `adk-eval-core`, `google-adk` versions; sandbox image digest or `subprocess` | the instrument |
| `task_id`, `repo`, `base_commit`, `snapshot_digest` | `tasks.jsonl` and the snapshot file | the subject |
| `budget` | `max_time_minutes`, `max_tool_calls`, `max_turns`, `timeout_seconds` as resolved | the slack |
| `site` | `local-docker-api`, `kaggle-l4x4`, `kaggle-scorer` | where it ran; cohorts.yaml uses the same names |
| `repeat_index`, `seed` | campaign-assigned | repeats are first-class |
| `started_at`, `duration_seconds` | `task_results.jsonl` | wall-clock, excluding setup |

## Outcome

| Field | Source |
| --- | --- |
| `resolved`, `test_exit_code`, `agent_patch_size`, `error` | `task_results.jsonl` |
| `tool_calls`, `total_llm_calls`, `turns` | `task_results.jsonl`, trace `final_metrics` |
| `tool_sequence` | trace steps: tool name per call, in order |
| `first_edit_call_index` | first `edit_file` or `write_file` in the sequence, or none |
| `format_errors` | count of tool errors whose type indicates a malformed or hallucinated call |
| `repetition_runs` | count of identical consecutive tool invocations (name plus arguments) |
| `submit_called`, `nudges` | trace |
| `patch_applied`, `protected_files_touched` | `test_outputs/<id>.log`, patch |
| `tokens_in`, `tokens_out`, `thinking_tokens` | trace token usage where the server reports it |
| `patch_digest` | SHA-256 of `patches/<id>.patch` |

## As written today (schema `gemma-agent.attempt/v1`, 2026-10-13)

The tables above are the design. The fields the tools actually write, from `lib/record.py` (shared by the API-lane
attempt tool and the GPU-lane ingest tool): `schema`, `attempt_id`, `task_id`, `bundle`, `config_digest`,
`model_alias`, `model_served`, `harness`, `sandbox`, `snapshot_digest`, `budget`, `site`, `repeat_index`,
`started_at`, `wall_seconds`, `outcome` (`resolved`, `test_exit_code`, `agent_patch_size`, `duration_seconds`,
`error`, `tool_calls`, `total_llm_calls`), `tool_sequence`, `first_edit_call_index`, `repetition_runs`,
`patch_digest`, `results_dir`, `tokens` (`prompt`, `completion`, `cached` totals from the trace's `final_metrics`; added
2026-10-14, derived from the trace by the regeneration script for older records), `infrastructure_failure`, `validity` (`infrastructure_invalid` /
`provider_interrupted` / `valid`, by rule in `lib/validity.py`), and on the API lane `lane` (requests, responses,
HTTP 200s, 5xx returned, upstream 5xx and 429 retries, retry sleep, upstream and paced seconds, prompt and
completion tokens, `retry_fraction`; from the adapter log over the attempt window). GPU-lane records take
`started_at` from the archive's `task_results.jsonl` mtime and `wall_seconds` from `duration_seconds`.

Not yet written, from the design: `adapter_digests`, `repo`, `base_commit`, `seed`, `turns`, `format_errors`,
`submit_called`, `nudges`, `patch_applied`, `protected_files_touched`. Recovery and loop counts wait for Leo's decision. Each is added
to `lib/record.py` when a table needs it; the regeneration script never reads a field the records do not carry.

## Taxonomy label (derived, then optionally judged)

Deterministic rule first, in this order; a Codex worker label is only consulted for class 4 vs 5 ambiguity:

1. `budget_exhausted_no_edit`: `first_edit_call_index` is none and the session ended on budget or nudges.
2. `format_error`: `format_errors > 0` and no edit landed.
3. `repetition_loop`: `repetition_runs >= 3`.
4. `edit_tests_failed`: an edit landed, patch applied, `resolved == false`.
5. `patch_rejected`: `patch_applied == false` or `protected_files_touched`.

Resolved attempts carry no failure class.

## Soundness

An attempt on a task marked unsound in the measurement notebook's audit (15 of 129) is recorded
but excluded from every aggregate, and the exclusion is stated in the paper.

## Budgets as data

`config/budgets.yaml` (signed): daily API call cap, weekly L4x4 hours, per-site concurrency, the
cut order of arms when quota slips. The dispatch tool refuses to launch beyond a cap and records
the refusal.


## Provenance chain (added 2026-10-15)

Three kinds of signed file, all with the same header on line 1,
`ryeos:signed:<time>:<sha256 of the body>:<ed25519 signature over that hex digest>:<signer fingerprint>`, signed by the
operator key whose public half is `docs/results/signing-public-key.pem` (fingerprint = sha256 of the 32 raw key bytes):

1. `.ai/config/gemma-agent/cohorts.yaml` and `budgets.yaml`: the arms and the budgets, signed before the data they
   govern. Every signed version is kept verbatim in `docs/results/config-history/<name>-v<version>-<signed-at>.yaml`.
2. `.ai/knowledge/gemma-agent/ledger-<UTC>.md`: one entry per `scripts/ledger.py` run, never edited. It names the
   config signatures in force and lists every record not covered by an earlier entry with the sha256 of the record
   file, the bundle and its directory digest, the site and `started_at`. Run after each GPU ingest and at least daily
   for the API lane (a graph step would need the sign capability on the schedule; not done).
3. The records themselves are not signed; their digests in a signed ledger entry are what fixes them.

`scripts/verify.py` (stdlib plus `cryptography`, no RyeOS) recomputes every body hash, verifies every signature and
fingerprint, checks every ledger row against the record file, its `config_digest` and the bundle on disk, checks that
each attempt started before its ledger entry was signed, and reports records not yet in any entry. What the chain
proves: the arm definitions existed, signed, before the records that cite them; the records have not changed since
they were listed; the tables regenerate from those records (`scripts/regenerate.py`). What it does not prove: that no
record was deleted before a ledger entry (the intent-to-run table and DECISIONS.md carry that), or that the clock on
the signing machine was right (the Kaggle dataset versions and kernel runs give independent timestamps for the GPU
lane).
