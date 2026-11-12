# LoRA cycle record (E4B): cycles 1 and 2

Working record of the first post-training cycle: the campaign's own test-verified 31B trajectories, selected and
rendered by rule, train a LoRA on the competition's E4B QAT checkpoint; one evaluation arm each for the base and the
adapter bundle on a held-out split decides adoption under a rule fixed in advance. Times UTC. Smoke that proved the
route: `docs/lora-smoke-record.md`.

## 1. Pieces and where they live

| Step | Mechanism | Output |
|---|---|---|
| Select | `tool:gemma-agent/lora_select` (signed): resolved by hidden tests, valid, 31B, Kaggle L4x4 lane, task outside the evaluation split, trace sealed; one example per model call | `results/lora/datasets/20261029T031307Z/` (1,436 examples, 78 attempts, 31 tasks; 16 evaluation tasks) |
| Render | kernel `gemma-agent-lora-render` (CPU, free): the model's own chat template, checked against served token counts | `rendered.jsonl` (sha256 242632951c2d16c2…), `render-report.json`, the template and config files of both model mounts |
| Train | kernel `gemma-agent-lora-train` (L4x4) driven by `datasets/lora-run/train.json`; two replicas, gradients averaged; QAT weights decompressed to the served bf16 values | adapter, bundle, `train-result.json`, `identity.json` |
| Register | `tool:gemma-agent/lora_register` (signed): training record with every digest; installs the adapter bundle under `submissions/` | `results/lora/cycles/c1/train.json`, `submissions/bf18-e4b-lora-c1/` |
| Evaluate | `scripts/push-run.py e4b-cell` / `e4b-lora-c1-cell` on the 16 tasks, the eval-4bit kernel (unchanged instrument), `tool:gemma-agent/ingest` | records `<task>-<bundle>-r0-<run_id>.json` |
| Accept | `tool:gemma-agent/lora_accept` (signed): adopt only if the adapter arm resolves at least 2 more of the 16 than the base | `results/lora/cycles/c1/accept.json`, `results/lora/selected.json` |

Acceptance rule written in DECISIONS (2026-10-29T22:23Z) before any evaluation existed.

## 2. Rendering: what the harness actually sends

Established 2026-10-29 by rendering the recorded messages with the Gemma 4 chat template and comparing token counts
with the prompt and completion counts vLLM reported for every call (1,436 calls), then confirmed on Kaggle with the
model mounts' own `chat_template.jinja` files (render kernel v2).

- The harness's request differs from the recorded trace in one way: ADK's identity processor appends
  `You are an agent. Your internal name is "swe_agent".` to the system instruction (after it, with a blank line; the
  instruction keeps the prompt file's final newline; the message reaches the template as a text part, after which the
  template emits a space). The same text appears in the served token ids of the Gluzdov kit's native trajectories.
- With that line, thinking enabled, every earlier turn's reasoning kept (ADK sends `reasoning_content` back and the
  template renders it for tool-call turns) and tool-call arguments rendered as mappings (Gemma's call syntax), the
  prompts of the 4-bit arms match the served counts exactly on 460 of the 534 non-compacted calls, and the
  completions match to within one token (the model ends its reasoning with a variable amount of whitespace before
  `<channel|>`; the template writes one newline).
- The served generation prompt ends at the tool response; the model opens its own thought channel. The template
  vintage in the Kaggle model mounts (version 2) does not append the opener; the kit's trajectories were served by a
  vintage that did.
- The bf16-release arms (`transformers/gemma-4-31b-it`, 628 examples) were served by a different template: history
  without reasoning and a longer tools block, so their prompts cannot be matched against the served counts. They are
  rendered with the serving template of the adapter's base and only their completions are checked (502 pass).
- ADK event compaction (`EventsCompactionConfig`, 15-event interval, 14,336-token threshold) replaced the history with
  a summary the trace does not keep in 318 calls (274 of the 4-bit tier): those served prompts are thousands of tokens
  shorter than the history and are excluded.
- 74 further 4-bit calls differ by a constant +3 to +43 tokens from one call onward within an attempt: tool outputs
  whose JSON escaping differs between the trace and the request. Excluded, not chased.
- The E4B and 31B QAT templates differ only in a branch taken when thinking is off; the E4B template renders the
  training data.
- Usable: 943 of 1,436 (441 exact-match 4-bit, 502 bf16). Targets by kind among the usable: tool-call turns with
  reasoning dominate; final-answer turns with reasoning (`RC-`) fail the completion check (the served completion is
  11 to 57 tokens longer than the parsed reasoning plus content) and are out.

## 3. Runs

| When | What | Result |
|---|---|---|
| 2026-10-29 22:11 | `grid-e4b-cell-lora-eval-r0-20261029`: base arm, bundle `bf18-e4b`, 16 tasks, cell, concurrency 1 (eval-4bit v11) | 2/16 resolved (fastapi_9425, rich_3470); 1 h 20 min |
| 2026-10-29 22:26 | render v1 (CPU, 217 s) | templates recorded; prompts one token short everywhere (system message form) |
| 2026-10-29 22:37 | render v2 (CPU, 212 s) | 943 usable; `rendered.jsonl` sha256 242632951c2d16c2… |
| 2026-10-29 22:42 | `lora-c1-e4b-r16-20261030`: training (lora-train v1): r=16, alpha=16, lr 1e-4 cosine, accumulate 4 per replica, 2 replicas, 1 epoch, both tiers, caps 16,384/4,096, 6% of attempts held out | v1–v5 failed before training (T4x2 twice; weights all on one GPU twice; shared-KV dict copied across devices); v6 trained: 423 examples per replica, 0 skipped, 105 updates, held-out loss 0.4315 → 0.3099, adapter sha256 8b98e73b…, 3 h 20 min |
| 2026-10-30 03:05 | `lora_register` c1 | `results/lora/cycles/c1/train.json`; `submissions/bf18-e4b-lora-c1/` (sha256 482d5f85…) |
| 2026-10-30 03:08 | `grid-e4b-lora-c1-cell-lora-eval-r0-20261030`: adapter arm, bundle `bf18-e4b-lora-c1`, same 16 tasks, same cell (eval-4bit v12) | 0/16; every attempt ran to the clock or the 80-turn budget; 341 of 564 agent steps repeat the previous message verbatim, 332 the previous tool call; 2 h 4 min |
| 2026-10-30 05:16 | `lora_accept` c1 (base 2, adapter 0, margin 2) | REJECTED; `results/lora/selected.json` keeps `bf18-e4b` |
| 2026-10-30 09:18 | lora-probe v1: base + c1 adapter at scale 1.0/0.5/0.25 under the scorer's flags, greedy, 16 held-out and training prompts in served form (13 min) | no loop on teacher histories at any scale; full scale closest to the teacher (0.67 vs base 0.41); the loop needs the agent's own turns |
| 2026-10-30 09:39 | `grid-e4b-lora-c1-s050-cell-loop-probe-r0-20261030`: c1 adapter at half scale, eval kernel unchanged, 4 both-lane-sound tasks outside the split (58 min) | loop mostly gone: 7 of 118 steps repeat the previous message (cycle 1: 341 of 564), 8 at the cap (71); 1 of 4 resolved (training tasks, so not a score) |
| 2026-10-30 10:41 | `grid-e4b-lora-c1-s025-cell-loop-probe-r0-20261030`: quarter scale, same 4 tasks (eval-4bit v14, 22 min) | no loop (0 of 104 steps repeat a message, 6 a call), 0 of 4 resolved: base-like |
| 2026-10-30 21:30 | `lora-c2-e4b-r16a8-qat1ep`: cycle 2 training (lora-train v7), amended plan: 4-bit exact tier only, alpha 8, 1 epoch, from base | 440 usable, 435 trained, 1 val attempt (5 examples), 54 updates per replica, 0 skipped; loss 0.43 → 0.27, held-out 0.80 → 0.60; adapter sha256 c58ed672…; about 1 h 50 min |
| 2026-10-30 23:4x | `lora_register` c2 | `results/lora/cycles/c2/train.json`; `submissions/bf18-e4b-lora-c2/` (sha256 99e6d767…) |
| 2026-10-30 23:56 | `grid-e4b-lora-c2-cell-lora-eval-r0-20261030`: cycle 2 adapter arm, 16 tasks, cell (eval-4bit v15) | 1/16 (rich_3518; base's two not resolved); 431 of 801 steps repeat the previous tool call, 48 at the cap, 14 of 16 tasks loop; 1 h 37 min |
| 2026-10-31 03:3x | `lora_accept` c2 (base 2, adapter 1, margin 2) | REJECTED; `selected.json` keeps `bf18-e4b` |
| 2026-10-31 03:4x | adapter strength on CPU, sum over 258 modules of ‖(alpha/r)·B·A‖ | c1 full 2.13, c1 half 1.06, c2 0.89: c2 is weaker than half-scale c1 and still loops, so magnitude alone does not explain the loop |

For reference on the 16 evaluation tasks at the cell, first attempts: 31B 4-bit resolves 3 of 16, 31B bf16 3 of 16, 12B 4-bit 4 of the 9 rich tasks it ran; some arm in the records (any lane or budget) resolves 10. (Corrected 2026-11-02: earlier text said 31B 4-bit 4 and gave 12B as 4 without its 9-task denominator.)

## 4. Cycles 3 and 4 (added 2026-11-05)

| When | What | Result |
|---|---|---|
| 2026-11-03 | c2 rolled out on the 12 training tasks (none resolved); 31B 4-bit labels one greedy next call at each of 280 states, 277 kept (`results/dagger/`) | the expert repeats the learner's previous call in 88 of 277 states: 87 of 89 where the last three or more calls are identical, 1 of 188 elsewhere |
| 2026-11-03/04 | c3 (`lora-c3-e4b-r16a8-dagger1`, c2 rows + 277 labels, Modal 2× A100) and c3-onset (without the 89 loop-following rows, post hoc), c2's configuration | c3 held-out 0.422 → 0.317; adapters sha256 76aa58f3… / (c3o) |
| 2026-11-04 | Gate on the swap v1 prefixes (base, c2, c3, c3o; `results/dagger/gate/swap-20261103T235727Z/`) | G1 repeat c3 − c2 on looping −0.09 [−0.17, −0.02] pass; G2 edit c3 − base on clean −0.23 fail (floor −0.15); c3o the same |
| 2026-11-04 04:29 | `grid-e4b-lora-c3-cell-lora-eval-r0-20261104`: exploratory arm (gate failed) | 1/16; 482 of 814 calls repeat the previous call; 14 attempts loop; REJECTED |
| 2026-11-04 06:34–09:55 | Collection: `bf18-e4b-explore` (bf18-e4b at temperature 0.7, 30-minute clock), 52 training tasks, 6 passes, concurrency 4 | 34 resolved of 312 attempts on 12 distinct tasks; loop rate about 1% |
| 2026-11-04 23:57 | `self_select` (signed) → dataset 20261104T235718Z; render v3 (CPU, 67 s) | 305 examples; 261 prompt-exact, 229 usable; rendered sha256 6e4cff9a… |
| 2026-11-05 00:00 | `lora-c4-e4b-r16a8-self1-2ep` (lora-train v8, L4x4, 1 h 29 min): from base, r16 alpha 8, 2 epochs | 227 within caps, 204 trained, 23 val (3 attempts), 50 updates per replica; loss 0.31 → 0.26, held-out 0.308 → 0.271; adapter sha256 75a9b5f5…; `submissions/bf18-e4b-lora-c4/` (2ec2e00f…) |
| 2026-11-05 02:05 | Gate (swap kernel v3, L4x4, 27 min; base, c2, c4; `results/swap/runs/c4-gate-kaggle-v3-20261105/`) | G1 repeat c4 − base on clean −0.001 [−0.011, +0.011] pass; G2 edit +0.012 [−0.042, +0.064] pass; repeat on looping base 0.052, c2 0.296, c4 0.017 |
| 2026-11-05 02:43 | `grid-e4b-lora-c4-cell-lora-eval-r0-20261105` (eval-4bit v24, 32 min) | 1/16 (fastapi_9425); median 10 calls, max 20, every attempt edits, none loops; `lora_accept` REJECTED (needs 4); `selected.json` keeps `bf18-e4b` |
| 2026-11-05 03:19 | Base repeat r1 (`grid-e4b-cell-lora-eval-r1-20261105`, 31 min) | 2/16 (fastapi_9425, rich_3905) |
| 2026-11-05 03:53 | Base repeat r2 (`grid-e4b-cell-lora-eval-r2-20261105`, 26 min) | 2/16 (fastapi_9425, rich_3905) |

Reading across the cycles: adapters trained on histories from another model (c1, c2 whole; c3 as labels on the learner's own
states) all loop on their own rollouts; the adapter trained on the learner's own resolved histories (c4) behaves like the base on
every measure and moves the score nowhere. Swap kernel v2 (01:37Z) ran against the previous swap-data version (lora_c1 + lora_c2)
and is kept at `results/swap/runs/swap-kaggle-v2-stale-c1c2-20261105/` as a replicate of base and c2, not as a gate.
