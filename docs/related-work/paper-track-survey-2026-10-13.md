# Paper-track field survey, 2026-10-13

Notes from reading the 16 most-voted public notebooks attached to the Gemma 4 Developer Agent Paper Track
(pulled read-only with `kaggle kernels pull -m`; none contained saved outputs, so every number below comes from
markdown prose or hard-coded constants, not executed cells). Written by a reader agent, checked against the
notebook list by the assistant; treat as working notes for PAPER.md's positioning table, not as citations. Cite the
notebooks themselves.

## One row per notebook

| slug | author | topic | type | data size (as stated) | key claim |
|---|---|---|---|---|---|
| budgetgraph-reproducible-retrieval-audit | hvanphucs112 | Import-graph PageRank diffusion fused with BM25 for issue-to-file localization at a fixed file budget; frozen protocol 2026-10-09 | measurement (retrieval only; no LLM) + resource (lzma payload with SHA256, rebuild code, tests) | 143 SWE-bench Lite tasks, 12 repos; 4 methods × 5 graph conditions × 5 seeds; paired bootstrap 2000 | Graph diffusion hurts: Hit@5 BM25 0.676 vs fusion 0.603; pure PPR 0.162. No Gemma performance claimed. |
| cortex-gemma-swe-paper | breycode | Poincaré embeddings, Fiedler slicing, Hebbian co-activation, ternary VQ, glyph CoT compaction, analyzer sub-agent, LoRA r=16 | essay/proposal with tasks.jsonl EDA | 129 dev tasks | 70.5% of gold patches touch 1 file, 89.1% ≤ 2 files. No resolution rate. |
| found-is-not-fit-cpu-context-budget-audit | eeshanvaghjiani2004 | Retrieval finds a file that does not fit the context budget | resource (CPU replay; no inference) | 57 issues, 12 repos | No numbers in notebook. |
| gemma-4-a-16384-output-cap-refuses-long-prompts | dariushafshar | Host sample's `max_output_tokens: 16384` + `max_model_len 32768` refuses any prompt ≥ 16,385 tokens; compaction (14,336) one call late | measurement (real per-call traces, 4×L4, 7 tasks) + resource (CSV + SHA256) | 7 tasks, 105 calls | 1/7 refused; model calls 98.7–100% of agent time; 34 tok/s; 90% of generated text is thinking the next call never sees; compaction 24–33 s; 0/7 resolved at 4 min; same config 0.06 public. |
| gemma-4-agent-research-artifact | jek1wantaufik | Bundle validator + experiment card manifest | resource | none | No results. |
| gemma-4-ast-fault-localization | muhzadit | Asserts W4A16 "attention dilution"; AST skeletonization; Localizer/Synthesizer/Verifier | proposal with asserted numbers, no data | "60-instance ablation" asserted | Pass 48.3% → 85.0% asserted; no bf16 arm, no traces. |
| gemma-4-dev-agent-paper | jefferyjeromeharris | Two-agent (coder + analyzer) experience report | essay; taxonomy asserted | ~120 hidden tasks | Single agent thinking-off ~6%, two-agent ~12%; failure buckets 40/25/15/10/5/5 without data; "thinking OFF better within the 4-minute budget". |
| gemma-4-graph-tools-audit | kmussa77 | "Silent Failures in Code-Graph Tools": graph audit, agent behaviour from per-call tables, run-to-run noise, 20 hand-inspected tasks | measurement with released per-call tables; raw traces withheld as competition data | 127 graphs; 3 runs; 60 tasks | swegemma 0.2.7 `APIRouter` returned ok with 0 results (fixed 0.2.11); how-to paragraph produced zero graph calls, so two runs serve as a noise estimate (exact McNemar). |
| graphfault-portable-gemma-localization-audit | winvist | Graph perturbation (swap50 vs clean) effect on E2B/E4B file localization | resource + measurement (localization only) | SWE-bench Lite + Verified | E2B swap50 − clean MRR@5 interval crosses zero. |
| kaggle-gemma4-papertrack | alexthomaskott | E4B bf16 on a T4, baseline vs AST-grounded prompt, first tool-call decision only | small single-step measurement | N = 20 | Numbers at runtime only. |
| selective-restarts-analysis-registration-v3 | hemantkumarbk | Preregistration: how much of a monitor+restart gain a non-selective interruption would produce; retry correlation and remaining budget | preregistration with exploratory priors | WTAS 71 tasks × 4 attempts; own 70 tasks × 5 attempts | μ 0.35–0.37, beta-binomial ρ 0.705–0.73, outcome ICC 0.74, one-retry headroom 0.061–0.069. "vLLM batching makes 'seed' a label, not a replication." Budget curves 0.5–2.0 × cap over replayed attempts. |
| tfd-agent-autonomous-swe-bench-resolution-with-ge | rajrajak99 | Analyzer + coder, BM25/AST localization, pre-patch failing-test assertion | essay + trajectories dataset | leaderboard | Rank 35/1,182; ladder 0.00 → 0.08 → 0.13 → "0.17+ projected". |
| thoughtkeep-analysis-registration-v1 | hemantkumarbk | Preregistration: cost and benefit of retaining prior-turn thoughts across tool calls on 31B W4A16; arms KEEP / STRIP / LAST-1 | preregistration with pilot | planned 70 tasks × 3 seeds × 3 arms; pilot 3 tasks | Prompt-growth per reply KEEP 1.07, STRIP 0.15, LAST-1 0.82; MDE ~5.4 pp; budgets 40 calls / 30 min / 100 turns, concurrency 4, temp 0.2. |
| token-insulated-compound-agent-graphs-for-gemma-4 | manderson240 | Token-insulated analyzer sub-agent, thinking_budget 1024, bytecode/diff pre-flight | essay/proposal | ladder only | 10.0% vs sample 5.0% vs monolithic 0.0%; tool turns 14.8 vs 21.2 vs 28.4; banning `search_similar_code` "+34% localization". |
| tracegraph-experiments | utsabdahal03 | Symbol overlap and commit provenance between graphs and embeddings; grep fallback | partly mock (Tables 1–4 hard-coded) | 4 repos; 7 fastapi tasks | Graph/embedding commit mismatch in 3/4 repos; Hit@5 fastapi 0/7 with graph vs 7/7 grep. |
| yaonem-gemma4-paper-harness-pareto-analysis | netaro | "Coupled-timeout wall-clock trap": `timeout_seconds` governs both the agent watchdog and `grade_task`; 540-minute container limit; proposes 2 min / 270 s / 4096 output / drop `search_similar_code` / 12-call budget | analysis with invented constants for the LB table | 129 dev tasks EDA | N_pub = 58 from LB quantum 1/58; median patch lines fastapi 18, rich 9, requests 4. |

## Clusters

1. Code-graph / retrieval / localization tooling (8 of 16). Three independent notebooks find the organisers' graph and
   embedding artefacts unhelpful or broken (budgetgraph, tracegraph, graph-tools-audit).
2. Two-agent analyzer + coder essays (7 of 16), ladder numbers 0.08–0.14.
3. Harness mechanics and budget arithmetic (4): 16384-cap (measured), yaonem, dev-agent-paper, token-insulated. The
   32,768 − max_output_tokens prompt ceiling, the 14,336 compaction trigger and the container limit are common knowledge.
4. Preregistered designs on the official harness (3): selective-restarts, thoughtkeep, budgetgraph. Paired per-task
   tests, task bootstrap, MDE, Holm, published before the run. This is the methodological bar.
5. Reproducibility packaging (6): embedded payloads with SHA256, hash-checked archives, per-call tables, replay scripts.

Quality is bimodal: about six rigorous entries, about ten essays or proposals with asserted, simulated or mock numbers.

## Overlap with our design

- Noise floor with repeats: hemantkumarbk already has it at one configuration (ICC 0.74, ρ ≈ 0.7, 70–71 tasks × 4–5
  attempts) and a preregistered 3-seed paired design. graph-tools-audit uses two behaviourally identical runs as a
  noise estimate. 16384-cap reports 1/7 with an exact interval and refuses to read it as a rate.
- bf16 vs 4-bit paired in the loop: none. ast-fault-localization names quantization as the cause with no control.
- Lifted-budget reruns of the same failures: none. selective-restarts treats budget as an estimand multiplier over
  replayed attempts (budget curves), which is not a rerun with more budget.
- Failure taxonomy from traces: several single-condition ones (graph-tools-audit loops / malformed / timeouts;
  thoughtkeep overflow / compaction / identical chains; selective-restarts eight monitor rules). None attributed to
  precision or size.
- Task soundness: graph-side audits exist (graph-tools-audit, tracegraph, budgetgraph). Nobody audits test-side
  soundness; the measurement notebook's audit we use (114 of 129) is the only one of that kind we know of.
- Regenerable records: standard among the rigorous entries. graph-tools-audit withheld raw traces as competition
  data; our release needs the same per-call-table shape or a stated licence basis.
- API-vs-GPU proxy fidelity: none. Size axis in the loop: none (graphfault varies E2B vs E4B for localization only).

## Gaps nobody covers

1. Paired bf16 vs 4-bit of the same agent on the same tasks in the loop.
2. A size axis (12B vs 31B) in the repair loop.
3. Lifted-budget reruns of the same failed tasks: slack measured directly.
4. API-vs-GPU proxy fidelity.
5. Attribution of named failure modes to precision or size.
6. Noise floor as a function of condition.
7. Test-side task soundness.

## Where our paper would not be novel

- "The hard budget binds and failures show up in the score" is the field's consensus and is already measured (model
  calls 98.7–100% of agent time; 90% of generated text is unseen thinking; thinking-off beats thinking-on at 4 min).
  The novel sentence is only that the budget unmasks precision and size damage, which needs the bf16 and 12B arms.
- Run-to-run noise with repeats, preregistration, task bootstrap, hashes and replay scripts are table stakes. They
  raise verifiability, not novelty. The noise floor is novel only when reported per condition and used to gate the
  contrasts.
- A failure taxonomy per se exists in several forms. Showing which modes move with precision or size, and which
  vanish under a lifted budget, would be new.
- "Quantization hurts the agent" has been asserted without a control (ast-fault-localization). Ours would be the
  first measurement; judges may be primed to ask for the control, which the bf16 arm is.
- Context-overflow and compaction mechanics are fully worked by 16384-cap and thoughtkeep; cite, do not re-derive.
- Task-set EDA is background.

## Facts to carry into our own setup

- bf18's `max_output_tokens: 8192` already avoids the 16,384 refusal trap (prompt ceiling 24,576).
- Kernel session limits: 52 tasks at up to 8 minutes plus grading will not fit one session; shard the grid runs.
- Concurrency in the rigorous entries: thoughtkeep ran at 4. The scorer's is still unknown.
