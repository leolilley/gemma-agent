# Paper outline

Target: Gemma 4 Developer Agent Paper Track, 3,000 words, Kaggle Writeup, due 2026-11-12 23:59 UTC.
Awards: $15k overall, $10k best new resource, $10k best new application. We target overall first, resource second.
Criteria, each 0 to 5, averaged: novelty, quality (generality beyond the competition), relevance,
verifiability, clarity.

The analysis plan that governs every number here is the "Analysis plan" section of
[PLAN.md](PLAN.md). This outline and that plan freeze together on 2026-10-22 (moved from 2026-10-20 on 2026-10-18,
when the paper was reframed around the clock prediction; DECISIONS 2026-10-18).

## Working title

The Clock Is the Budget: how a per-task clock converts decode speed into pass rate for a Gemma 4 coding agent

"Turns, Not Weights" (2026-10-18) was retired the same day on review: it asserts that the weights do not matter, which
one checkpoint pair and an uncertain matched-calls estimate cannot carry. It may survive as a section heading if H4's
interval is tight and includes zero. The 2026-10-15 title "No Room to Recover" is retired: it promised a recovery
result the arms do not isolate. Branch 2 has no preset title; it names the finding plainly ("A forecast that failed:
...").

## Thesis

Under a fixed wall-clock budget, a coding agent's pass rate is set by how many tool calls it can afford. On these two
releases decode speed moved the pass rate before anything the weights did; whether the weights matter at equal calls is
the confirmatory question H4, answered with an interval and not assumed. We measured this on one Gemma 4 31B agent, unchanged,
in its two released forms (bf16 and the quantization-aware-trained W4A16 checkpoint) on the same tasks, harness and
4×L4 hardware. The 4-bit release takes about 3.5 seconds per call against 15 to 17 for bf16 and reaches a median 26 of
28 calls in 8 minutes against 12. The same budget binds differently on the two: the 4-bit agent runs into the 28-call
cap, the bf16 agent into the clock. Lifting the clock for bf16 exposes its calls-to-solve curve, and that curve
predicts the clocked outcome task by task. A leaderboard number at this cell is a throughput number for bf16 and a
call-count number for 4-bit, and the exchange rate between seconds, calls and passes is the quantity a developer on
everyday hardware can act on. The soundness audit of the local evaluation lane is the second, named contribution.

## Claim and evidence matrix

This is the paper's claim budget. A claim appears at the strength its evidence supports, and
missing or contrary evidence stays visible in the result and the conclusion. Criteria are this competition's five.

| ID | Candidate claim | Required evidence | What narrows or kills it | Criteria |
| --- | --- | --- | --- | --- |
| H1 (headline) | Under the clock, the pass rate of a configuration is predicted by its calls-to-solve curve and its seconds per call: the lifted-clock run predicts the clocked outcome per task. | A3: retrodiction on 39 rich tasks (predicted vs observed count, per-task agreement, disagreements named); the signed forecast on 29 fastapi tasks and the cell run after it; `docs/results/curve.csv`. | Forecast outside its count interval, task-level agreement below the criterion, or agreement not above both baselines (PLAN.md A3): branch 2. | novelty, relevance, quality, verifiability |
| H2 | The same budget binds differently by precision: 4-bit at the call cap (S(k) rising to k = 28, P(C) flat after 5 min, 20 of 68 at the cap), bf16 at the clock (29 of 39 at the clock, S(k) flat beyond 14 with the clock lifted). | A2 per arm: terminal reasons, S(k), P(C). | A 4-bit S(k) that is flat before 28, or a bf16 lifted S(k) still rising at 28. | relevance, clarity |
| H3 | The exchange rate: seconds per call by precision, decode tokens per second, calls reached at 8 minutes; model calls are 96 to 98% of agent time, so call cost is the whole budget. | A1 from the traces; the latency fit. | Nothing kills it; it is a measurement. Its generality is limited to this hardware and stack and is said so. | relevance, verifiability |
| H4 (confirmatory) | At matched calls the W4A16 release and bf16 resolve about the same tasks; the as-scored gap is the clock's. | A4: D, the matched-calls contrast and the as-scored contrast with 90% intervals on the 68 tasks. | Interval excludes zero either way: reported as such; the headline does not depend on it. | quality |
| H5 | A smaller model at the same clock gets more calls; whether the extra calls become passes is the size point. | 12B 4-bit cell on 39 rich tasks: seconds per call, calls reached, pass rate, terminal reasons, beside 31B 4-bit. | 12B gets no more calls. Reported as a point, never as "smaller is better". | relevance |
| H6 | 4-bit gains from calls beyond 28: S(k) continues past the cap. | The 4-bit lifted subset (20 cell failures, 40 calls / 16 min): passes, solve calls, loop calls vs recovery episodes. | No passes beyond call 28, or the extra calls mostly loops. | quality |
| H7 | Run-to-run variance at the cell is concentrated at the clock: the API-lane noise floor is near-deterministic, and the clock-bound GPU attempts are where reruns flip. | Noise floor (8 tasks × 5 repeats, API lane); the clock-capped 4-bit reruns; the GPU repeats on the 10-task subset. | GPU repeats flip tasks that finished well inside the clock. | quality, verifiability |
| H8 (resource) | The harness's subprocess sandbox, as every local Kaggle notebook runs it, grades the image's preinstalled package for src-layout repositories and installs no test dependencies, so 58 of 129 public tasks cannot be measured on that lane; 68 are sound on both lanes. | The two-lane audit (`docs/results/lane-soundness.csv`): `verify_task` with empty-patch and gold-patch controls on all 129 tasks, both lanes; the import-source check. | A task we call unsound that passes the gold control on a clean Kaggle session. | relevance, verifiability, quality |
| H9 | Every number regenerates from the released records by one script; the forecast was signed, and shipped in the Kaggle dataset version the cell run read, before the data it predicts existed; one verify script and the public key are shipped. | Records, `regenerate.py`, `curve.py`, `verify.py`, the signed forecast, the public key. | Any figure that cannot be rebuilt; any signature that fails. | verifiability |
| H10 | The submitted bundle is a cell of the grid, not a separate artifact. | Submitted zip digest equals the bundle digest in the records; submission id in T1. | A submission whose config is not in the records. | relevance |

### Decision rule for the paper's centre

Fixed 2026-10-18 before any fastapi bf16 data, any 12B data and any 4-bit lifted data existed (what had been observed:
PLAN.md, "What had been observed when the primary analysis was added"). Not revisited after.

1. Forecast supported on all three criteria of PLAN.md A3 (count inside the signed interval; task-level agreement at
   or above the criterion; agreement above both baselines): headline "The Clock Is the Budget"; H1 to H3 are the body,
   H4 one paragraph, H5 and H6 the size and cap points.
2. Forecast fails any criterion: the paper reports a failed forecast, in the abstract, with the per-task
   disagreements and both baselines. A miss establishes that the lifted-clock run did not predict the clocked run on
   these tasks; it does not say why. The GPU repeats (H7) and the disagreements' trace reasons are then used to ask
   whether run-to-run variance at the clock accounts for it, and the answer, or its absence, is reported as such.
   H8 and the exchange rate (H3) carry the paper.
2b. **Amendment adopted 2026-10-19 on the fourth review, Leo may veto before the freeze; moot for the signed fastapi forecast, which has six late solvers (2026-10-21)** (a case the rule did not
   cover, found by testing the forecast script on 10 rich tasks before any fastapi data existed): the fastapi set has
   no late solver, so the clock rule and the lifted-only predictor are the same function and the lifted-only
   comparison (c2) can be neither met nor failed. The count criterion (a), the task-level criterion (b) and the
   all-fail comparison (c1) still apply and must all pass. Then (c2) is reported as untestable on that set, the
   headline becomes "the lifted run predicts the clocked outcome", the clock rule's own increment is read from the
   rich late solvers only, and the abstract says so. A failed applicable criterion is branch 2.
   **Amendment adopted 2026-10-20 on the fifth review (Leo's go):** the branch is decided on the preregistered rule (solve time = last
   edit to any file). The fastapi forecast also carries the source rule (last edit under the repository's source directory), declared
   after the rich retrodiction showed rich_3043's miss was a config-file edit (.pre-commit-config.yaml) after a 116-s fix, with its own count, interval and
   criteria; it is reported beside the primary in every branch and decides nothing. Its fastapi source directories were amended on
   2026-10-20 (docs_src/ added) after inspection of the lifted fastapi outcomes and before any fastapi cell outcome; the paper says
   so where it reports the source rule, and the forecast file carries the original mapping's one differing prediction (PLAN.md A3).
3. The cell fastapi run could not be made (quota): the retrodiction on 39 rich tasks is reported as a retrodiction,
   the word "forecast" does not appear in the title or abstract, and the paper says no forecast was tested.

In every branch the counts, the intervals and the disagreements are printed. The endpoint is never chosen after
looking.

## Claims to avoid

- "Quantization breaks coding agents." One checkpoint pair, four repositories, one harness.
- "Quantization" as such. The 4-bit arm is Google's quantization-aware-trained W4A16 release, different weights from
  the bf16 release, not the bf16 model rounded. The contrast is between the two artifacts a developer can deploy.
- "The clock explains everything." The clock takes almost nothing from 4-bit here (5 of 68 at the clock); 4-bit is
  bound by the call cap. The claim is that the budget binds differently, and that for the slower configuration the
  clock is what binds.
- "Faster is better." Turns are necessary, not sufficient; the 12B point is the test of whether extra turns become
  passes, and it is one point.
- The exchange rate generalises beyond 4×L4 and this serving stack. The method generalises; the numbers do not.
- "RyeOS makes this verifiable." The records, the scripts and the signed forecast do; another stack could have
  produced them.
- Anything about the hidden set, or where a configuration would land on the private leaderboard.
- "First to measure noise in agentic evaluations." Randomness (ICLR 2026), ICC (Dec 2025) and Measure Before You
  Tune (Oct 2026) did.
- "Beats the reference" from local numbers alone. Only scorer results count, with the ±2-task movement beside them.
- "Repeats are replications." Repeats of one identity estimate variance; they do not replicate the finding.
- "The API lane is the full-precision arm" without the proxy-fidelity caveat.
- "More budget means more recovery." Extra calls are split into loop calls and recovery episodes before any claim.
- "Local Kaggle-notebook scores are comparable to the scorer." Only the two-lane-sound tasks are comparable.
- A per-task prediction described as "correct" when it agrees by chance. Agreement is reported beside the base
  rates (most tasks fail under both readings).

## Positioning against prior work (cite, then extend)

| Prior | Finding | What we add |
| --- | --- | --- |
| On Randomness in Agentic Evals, ICLR 2026 | 60k SWE-bench trajectories; single-run pass@1 varies 2 to 6 points; SD above 1.5 at temperature 0 | method, not headline: repeats and intervals on every comparison, in this regime |
| Stochasticity in Agentic Evaluations (ICC), Dec 2025 | ICC decomposition; 8 to 32 trials to stabilize | the variance located: at the clock, not spread over tasks |
| Flat Score, Amplified Failures (Jang, Yang, Lim, Park), Jul 2026 | τ²-bench, 16/8/4-bit: task score flat because the benchmark allows ten errors; quantization amplifies existing failures; shrinking the error budget re-exposes a gap. Step, turn and time budgets and model size not varied. | a hard clock and call cap on real repository fixes; the finding that at a clock the precision sets the budget before it sets the answer; the W4A16 release rather than post-training rounding |
| Budget-pressure studies, 2026 | exploration declines under budget; failure invisible until ~75% of budget | the budget measured as calls affordable, with the calls-to-solve curve that says what a given number of calls is worth |
| Measure Before You Tune (Kaggle, Oct 2026) | 692 outcomes, 8 configs, 114/129 soundness audit, 0.06 spread on identical archives | the second lane's audit with test-side controls; the paired precision arm; the prediction test |
| Kaggle paper-track entries, Oct 2026 (survey in docs/related-work) | repeated-attempt panels at one configuration (ICC 0.74, 70 tasks × 4–5 attempts); preregistered paired designs; a per-call trace study of the 16,384 output cap; code-graph tool audits; hashes and replay scripts as the norm | the first paired bf16 vs W4A16 arm in the loop; the lifted-clock design that turns the clock into a measured quantity; a forecast signed before its data |
| Kaggle agent-track notebooks, Oct 2026 | the 0.17–0.18 recipe: 8 min / 28 calls, single agent, thinking on, graph tools unused, edit by call 12; attempts finishing by 14 calls resolve 52%, those dragging to 15–20 calls 5%; 100-call budgets fed loops | the same facts read through S(k): where on the curve the recipe sits, and why "edit by call 12" is a statement about calls affordable, not about agents |

### What is not new, and is not claimed as new

- That the hard budget binds and failures show up in the score: measured by several entries (model calls take 98 to
  100% of agent time; 90% of generated text is thinking the next call never sees; the 50 → 25 call ablation).
- Run-to-run noise with repeats, preregistration, task bootstraps, hashes and replay scripts: table stakes among the
  rigorous entries. They earn verifiability points, not novelty.
- A failure taxonomy as such, and the context-overflow and compaction mechanics (16,384 cap, 14,336 trigger).
- "Quantization hurts the agent" as a sentence: asserted in the field without a control. Ours is the control, and the
  control says the clock did it.

### Decided before the freeze (Leo; DECISIONS lines)

1. 2026-10-15: D is the confirmatory precision endpoint; the bf16 lifted cell is dropped (clock-bound at 16 min);
   size is 12B 4-bit at the cell only; loop calls and recovery episodes are counted by rules fixed before the data.
2. 2026-10-18: the paper's centre is the clock prediction (A1 to A3); the forecast on fastapi is signed before the
   bf16 cell fastapi run; 12B is promoted above the GPU repeats; the freeze moves to 2026-10-22.

## Sections and word budget

| # | Section | Words | Figure / table |
| --- | --- | --- | --- |
| 0 | Abstract | 150 | |
| 1 | Question: every developer running an open model locally runs it under a clock; what does the clock buy and cost, and can it be predicted | 250 | |
| 2 | Related work (table above) | 200 | |
| 3 | Setup and method: unchanged harness as instrument; the two releases; the grid (cell, calls-only); the 68 tasks; definitions (solve call, solve time, S(k), P(C)); the predeclared analyses and validity rule; records | 300 | T1 exact identities |
| 3.1 | Resource, named: instrument soundness. 58 of 129 tasks unmeasurable on the Kaggle lane, 68 sound on both; controls; the per-task table and the fix; the discussion post | 200 | T3 two-lane soundness |
| 3.2 | Verification and the forecast certificate: configs and records signed, the forecast signed and shipped in the cell run's dataset version before its data (Kaggle's version timestamp is the external anchor; a signature header's time is the signer's claim), the launch guard that refuses the forecast run without a verified covering forecast and records the dataset version and timestamp, the verify script and the public key; RyeOS named once as the producer | 100 | |
| 4.1 | The exchange rate and what binds: seconds per call, calls reached, terminal reasons, S(k) and P(C) per precision | 400 | F1 P(C), F2 S(k) |
| 4.2 | The prediction: retrodiction on rich, forecast on fastapi, per-task agreement, the disagreements | 300 | F3 predicted vs observed |
| 4.3 | Precision at matched calls, read against the noise floor: D and both contrasts; where reruns flip | 250 | F4 contrasts with intervals |
| 4.4 | Size and the cap: 12B at the clock; 4-bit beyond 28 calls, net of loops | 150 | T2 |
| 5 | Implications: how to spend a clock (output length, decode speed, model size, what a tool call must return to pay for itself; the code-graph results read as call cost) | 300 | |
| 6 | Limitations: four repos, one checkpoint pair, one hardware, one repeat on most arms, public tasks the model may have seen, retrodiction vs forecast | 150 | |
| 7 | Resource and reproducibility: the record set, the curve data, one regeneration script, how to verify | 200 | |

## Figures

- F1 pass rate versus clock, P(C), for 4-bit (cell, 68 tasks) and bf16 (clock lifted, 68 tasks), with the 8-minute
  line and the observed cell pass rates marked.
- F2 S(k) for both precisions with the 28-call line; the 4-bit lifted subset extends the 4-bit curve past 28.
- F3 predicted versus observed per task: rich retrodiction (39) and fastapi forecast (29), four cells each.
- F4 the three contrasts (as scored, matched calls, D) with 90% intervals; the noise-floor bracket beside them.
- T1 exact identities of every artifact used, including the submitted bundle and submission id.
- T2 size and cap: 12B vs 31B at the cell; 4-bit lifted subset passes beyond call 28, loop calls vs recovery episodes.
- T3 two-lane soundness per task (docs/results/lane-soundness.csv).
- T4 cost vector per arm.

One diagram only, the grid. No component diagram of anything.

## Rules for the writing

- Every number comes from a record; the regeneration script is the appendix, and each figure
  names the query that produced it.
- Lead with the finding. The abstract's first sentence is the thesis, the second is the exchange rate, the third is
  the prediction result.
- RyeOS appears in one place, section 3.2, as what produced and signed the records and the forecast, with the verify
  recipe and the sentence that another stack could have produced the same records. Nowhere else.
- Graph tools: one setup sentence (unregistered, as in every top bundle; zero graph calls in the traces) and one
  guidance paragraph in section 5. No graph arm.
- No claim about the hidden set. Public tasks and public scores are separate evidence.
- The retrodiction is called a retrodiction and the forecast a forecast, every time.
- Repeats of one identity are never called replications.
- Freeze this outline on 2026-10-22. Changes after that need a line in DECISIONS.md stating
  what had already been observed at the time.
