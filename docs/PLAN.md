# Plan and gates

Dates are UTC. Paper deadline 2026-11-12 23:59. Agent entry 2026-11-25, final 2026-12-02.

| Week | Ends | Deliverable | Gate (miss it and the stated fallback applies) |
| --- | --- | --- | --- |
| 1 | 2026-10-20 | Reference bundle submitted and scored. Harness runs locally on `requests` tasks via the API. Attempt tool writes a record. Noise-floor cohort running: one config, same tasks, 5 repeats. 4-bit and bf16 kernels pushed on the free L4x4 notebook. Paper angle, claim matrix and analysis plan frozen. | A record from a resolved attempt exists by 2026-10-17 or the campaign stops and the cause is written down. (Met 2026-10-13: `requests_7309` resolved through the daemon-run tool.) |
| 2 | 2026-10-27 | Reframe done (2026-10-18): the clock-prediction analysis is the centre, PLAN and PAPER rewritten, freeze 2026-10-22. bf16 calls-only rich chunks complete; the rich retrodiction computed on all 39. Agent track: one leaderboard submission as capacity allows. | The retrodiction on 39 rich tasks exists by 2026-10-27 or the headline falls back to the exchange rate and what binds (A1, A2) with the soundness audit. |
| 3 | 2026-11-03 | bf16 calls-only fastapi chunks complete; forecast signed; bf16 cell fastapi run landed and the forecast evaluated unchanged; the 4-bit calls-only rerun of fastapi_15661 (D complete on 68); the GPU repeats; then 12B 4-bit cell rich and the 4-bit lifted subset as quota allows (order reconciled 2026-10-21 on the seventh review). | The forecast is signed before the cell run is pushed, or the paper reports the retrodiction only and says no forecast was tested. |
| 4 | 2026-11-12 | Paper, 3,000 words, every number regenerated from records by one script. Submit by 2026-11-10. | Submitted. |

## Cells

| Cell | max_time_minutes | max_tool_calls | max_turns | command timeout |
| --- | --- | --- | --- | --- |
| competition (the scorer's, the public 0.18 reference) | 8 | 28 | 80 | 240 s |
| lifted | 16 | 40 | 80 | 240 s |
| calls-only (added 2026-10-14) | 30 | 28 | 80 | 240 s |

At the competition cell the binding caps are calls and time; 80 turns is rarely reached. The
lifted cell lifts exactly those two. Earlier notes said "40 turns"; they meant 40 tool calls.
The calls-only cell keeps 28 calls and lifts only the clock, so that the call cap binds for both precisions: on L4x4
bf16 decodes at about 15 tok/s against 35 for 4-bit, and at 8 minutes bf16 gets a median 12 calls against 26
(DECISIONS 2026-10-14 13:50). It is the matched-calls precision contrast; the cell contrast is a throughput result.
On the API lane time is meant to be non-binding (120-minute ceiling since 2026-10-13 15:26 UTC; the 30-minute ceiling
bound at 14 calls and the 90-minute one at 21) and the cell is call-bounded; the
would-have-been-cut-at-8-minutes outcome is a derived column from per-call timestamps.

## Measurement arms (revised 2026-10-13 after the API lane was measured; reframed 2026-10-18)

Task set for the grid and size arms (revised 2026-10-14): the **68 tasks sound on both lanes**, rich 39 and fastapi 29.
A task is in the set if the measurement notebook's Docker-lane audit (114 of 129) and our Kaggle-lane audit
(`results/kaggle/lane-audit-2`, the harness's own `verify_task` with empty and gold controls under
`sandbox='subprocess'`) both pass it. requests (0 of 13) and httpx (0 of 1) fail the Kaggle lane because its
subprocess sandbox never installs the repository and grades the image's preinstalled package; they stay in the
Docker-only noise floor. fastapi loses 33 tasks on the Kaggle lane to missing test dependencies (`inline_snapshot`,
`dirty_equals`, pydantic version) that the image lacks and subprocess mode never installs. Table:
`docs/results/lane-soundness.csv`. Kernel runs are sharded by repo.

| Arm | Attempts | Where, cost | State |
| --- | --- | --- | --- |
| Noise floor: bf18 on the API lane (provider-default thinking), call-bounded cell, 8 sound requests/httpx tasks × 5 repeats | 40 | API, about 4 days | complete 2026-10-16: 35 valid 120-min attempts, 12 resolved, 0.35 [0.10, 0.62]; 1 task of 8 with mixed outcomes |
| 31B bf16, cell, rich | 39 | L4x4; bf16 runs to the 8-min clock on most tasks: about 5 h | r0 done 2026-10-14: 10/39, 29 at the time cap |
| 31B bf16, cell, fastapi: **the out-of-sample prediction run** | 29 | L4x4, 3.6 h | landed 2026-10-21 (kernel v20 on dataset version 25, 18 h after the forecast's signature; v18 and v19 errored at model load without attempting a task): 9/29, 20 at the clock; forecast evaluated unchanged, branch 1 (predicted 7, observed 9 in [6, 11]; agreement 27/29; baselines 20 and 25). Was "dropped 2026-10-14"; reinstated 2026-10-18 as the test |
| 31B bf16, lifted cell (16 min, 40 calls) | ≤ 68 | | dropped 2026-10-15: at 16 min bf16 is still clock-bound; replaced by the calls-only arm |
| 31B 4-bit, cell | 68 | L4x4, about 5 h per repeat | r0 done 2026-10-14: rich 12/39, fastapi 15/29; 20 of 68 attempts ended at the call cap, 5 at the clock |
| 31B bf16, calls-only cell (30 min, 28 calls), all 68 tasks, 1 repeat | 68 | L4x4, 1.5 to 2 GPU h per 5-task chunk, 14 chunks, rich first | complete 2026-10-20: rich 16/39 (8 chunks), fastapi 13/29 (6 chunks); 6 attempts ran to the 30-min clock. Priorities 1 and 2 |
| 31B 4-bit, calls-only cell, the 5 tasks whose cell attempt hit the 8-min clock (rich 3180, 3468, 3506, 3930; fastapi 15661) | 5 | L4x4, about 40 min | rich 4 done 2026-10-15 (2/4, both inside 8 min); fastapi_15661 was to ride with the fastapi chunks and did not; pushed on its own 2026-10-21 01:40 UTC (seventh review) |
| 12B 4-bit, cell, rich, 1 repeat | 39 | L4x4, about 2 h | post-freeze, priority 5 (was 4; the repeats moved ahead of it 2026-10-20 on the fifth review, Leo's go): the size point on the exchange rate  **Landed 2026-10-26 (kernel eval-4bit v9, 3.5 h): 11 of 39; C3 in tables.md; DECISIONS 22:25 UTC.** |
| 31B 4-bit, lifted cell (16 min, 40 calls), seeded subset of 20 of the 41 cell failures (seed 20261015, DECISIONS 2026-10-15) | 20 | L4x4, about 2 h | post-freeze, priority 6 (was 5, 2026-10-20): reads S(k) beyond 28 for 4-bit, whose cell attempts end at the call cap **Landed 2026-10-27 00:09 UTC (kernel eval-4bit v10, 2.1 h): 4 of 20 against a cell-rerun baseline of 2 of 6; C4 in tables.md; DECISIONS 00:15 UTC.** |
| GPU-lane repeats: both cells once more on a seeded subset of 10 rich tasks (seed 20261019, drawn when pushed: rich_2943, 3052, 3067, 3454, 3470, 3518, 3521, 3675, 3935, 4077; one late solver, no inside-boundary miss; DECISIONS 2026-10-21 01:45) | 20 | L4x4, about 2.5 h | pushed 2026-10-21 23:56 UTC (both cells, repeat 1); priority 4 (was 6; moved ahead of the 12B arm 2026-10-20 on the fifth review, Leo's go: whether the late-solver separation survives run-to-run variation is the paper's central uncertainty, and the repeats let forecast misses be read against ordinary repeat disagreements): same-lane run-to-run variance at the clock |
| Proxy fidelity: bf18 on the API lane, same tasks, 1 repeat | 68, or a subset | API, 12 attempts a day | rich snapshots fetched and the cohort running since 2026-10-19; complete 2026-10-22 02:30 UTC: 39 of 39 rich tasks attempted, 12 resolved, per-task agreement with the first 4-bit cell attempt 33 of 39 (C5); under the adapter deadline every attempt completes or reaches the 120-min ceiling, and the free tier's rate limit binds (21 of the 39 ceiling-bound). fastapi not started (snapshots not on disk) |

The bf16 arm is the precision arm. The API lane carries the provider's default thinking and its
own latency, so it is the noise-floor lane, the proxy-fidelity lane and the cheap configuration
lane, never the precision contrast.

## Cohort discipline

Three cohorts, in order, then only repeats:

1. `noise_floor`: bf18 on the API lane, 8 sound requests and httpx tasks × 5 repeats. Complete.
2. `grid`: precision (bf16, 4-bit) × budget (cell, calls-only) on the 68 both-lane tasks, on L4x4; the bf16 cell
   fastapi run as the forecast test; the 4-bit lifted cell on the seeded subset of 20 cell failures.
3. `size`: 12B 4-bit at the cell on the 39 rich tasks. E4B, 12B bf16 and 12B with raised calls are future work.

After the freeze (2026-10-22 23:59 UTC; moved from 2026-10-20 on 2026-10-18 for the reframe) a new cohort may only
add repeats, lifted-cell reruns, or the same arms on more tasks. A new axis or mechanism needs a DECISIONS.md line
and an explicit go from Leo. Nothing in the campaign may require a RyeOS capability that was not installed on
2026-10-13; if a step seems to, the step is wrong, not the node.

Why: the ARC paper campaign answered each null with a new protocol instead of more repeats,
reached fifteen protocols and a scratch tree of 934 files, and its primary comparison stayed null.
Here the hypotheses are fixed and only N grows.

## Analysis plan (freezes 2026-10-22 with the paper angle)

**Experimental unit.** One attempt: task × bundle × site × repeat. Attempts on one task are
correlated, so intervals come from a cluster bootstrap over tasks with repeats pooled within task.
Repeats of one identity are same-identity samples: they estimate variance and are never called
replications.

**Definitions (fixed 2026-10-18; implemented in `scripts/curve.py`, computed by `scripts/regenerate.py`).**
*Solve call*: the index of the last `edit_file` or `write_file` call of an attempt, the call after which the graded
patch was on disk. *Solve time*: wall seconds from the first model call to the completion of that edit (the tool
observation's timestamp), so it excludes the decode of the following call. *S(k)*: over one attempt per task, the
share of tasks resolved with solve call ≤ k; S(28) is the pass rate at the call cap. *P(C)*: the share resolved with
solve time ≤ C seconds; for a set of attempts the clock did not cut, P(480) is what the same configuration would have
scored under the 8-minute clock. *Seconds per call*: the median gap between consecutive model responses in an attempt.
*Calls reached*: tool calls completed when the attempt ended.

**Primary analysis (added 2026-10-18; replaces D as the paper's centre; D stays as the confirmatory endpoint A4).**
The thesis is that under the competition's clock a coding agent's pass rate is set by the tool calls it can afford,
with decode speed first-order and weight precision second-order. Four analyses, in order:

- **A1, the exchange rate.** Per precision: seconds per call, decode tokens per second (the latency fit of
  `scripts/latency-fit.py`: call seconds against prompt and output tokens), output tokens per call, calls reached at
  8 minutes. Reported as medians with the task-cluster interval.
- **A2, what binds.** Per arm: terminal reasons (call cap, time cap, resolved, tests failed, no edit), S(k) and P(C).
  The claim "4-bit is bound by the call cap and bf16 by the clock" is read from three numbers each: the share of
  attempts ending at each cap; whether P(C) is flat over the last three minutes of the clock (4-bit); whether S(k)
  is still rising over the last eight calls of the cap (4-bit) or flat (bf16 with the clock lifted).
- **A3, the clock prediction, the headline test.** Per task, the bf16 calls-only attempt (clock lifted, same 28-call
  cap) predicts the bf16 cell outcome: a pass if and only if it resolved with solve time ≤ 480 s. Two readings:
  (i) *retrodiction* on the 39 rich tasks, where the cell outcomes were known (2026-10-14) before the lifted runs
  landed: predicted count against observed, per-task agreement, and the disagreements named with their trace
  reason; (ii) *forecast* on the 29 fastapi tasks, where no bf16 cell data exist: after the bf16 calls-only fastapi
  chunks land and before the bf16 cell fastapi run is pushed, the per-task predictions, the predicted count, and the
  acceptance interval are written to a knowledge item `knowledge:gemma-agent/forecast-bf16-cell-fastapi`, signed
  with the operator key; the cell run is pushed only after that signature exists. The signature covers the body, not
  the time in its header, so the chronology is anchored outside our key: the signed forecast file is shipped inside the
  Kaggle dataset version the cell run reads (`scripts/push-run.py --forecast`), its sha256 is written into that
  version's run.json and travels into every record of the run at ingest; `scripts/verify.py` requires each covered
  record to carry that digest and to have started after the signature time. Kaggle's version timestamp is the
  third-party record of when the forecast existed. The launcher refuses the bf16 cell run on any fastapi task unless
  it is given a forecast under `.ai/knowledge/gemma-agent/` whose signature verifies against the shipped public key,
  whose signer is the operator key and whose task list covers the run (`scripts/forecast_check.py`); it then writes
  the dataset version, Kaggle's timestamp and the kernel version to `docs/results/forecast-runs/<run_id>.json`, and
  the kernel returns the forecast copy with the outcomes. The bundles dataset is private during the campaign and is
  made public at release so that timestamp can be checked. An attempt's `started_at` is its first model call from the
  harness trace; GPU records written before 2026-10-19 carried the run's finishing time read in the wrong timezone,
  corrected by a signed correction item that the verifier applies (records named in ledger entries are never edited).
  Acceptance, three criteria,
  all written into the signed forecast before the cell run: (a) count: the observed pass count lies inside the 90%
  interval of the predicted count under the rich retrodiction's per-task disagreement rate (each task's prediction
  wrong independently with that probability); (b) task level: per-task agreement on the 29 tasks is at least the rich
  retrodiction's agreement minus 0.10; (c) baselines: that agreement exceeds both the all-fail predictor (every task
  fails) and the lifted-only predictor (the lifted outcome with no clock rule). The clock rule and the lifted-only
  predictor differ only on the late solvers, the tasks the lifted attempt resolved after 480 s, so (c) is decided on
  that set and the forecast names it: how many late solvers, and how many went each way. (c) is two comparisons, (c1)
  against the all-fail predictor and (c2) against the lifted-only predictor. **Adopted 2026-10-19 on the fourth review
  (forwarded by Leo; Leo may veto before the freeze; moot for the fastapi forecast, which has six late solvers, so 2b cannot apply to it, recorded 2026-10-21):** when the target set has no late solver the two predictors
  coincide and only (c2) becomes untestable; (a), (b) and (c1) still apply and must pass. Then the forecast reports
  (c2) as untestable on that set, and if the applicable criteria pass the finding is that the lifted run predicted the
  clocked outcomes while the clock rule's increment stays untested on fastapi and is read from rich only (PAPER.md
  decision rule, branch 2b). A failed applicable criterion is a failed forecast as before. Found when the forecast
  script was tested on 10 rich tasks with no late solver: agreement 9 of 10, all-fail 8 of 10, lifted-only 9 of 10;
  that test passes the narrower rule as branch 2b. The writer refuses to write a forecast for any task whose cell
  outcome already exists, and has a separate retrospective mode for tests on known data that cannot be signed. A forecast that misses any
  criterion is a failed forecast; what failed it is a further question, asked of the GPU repeats and of the
  disagreements' trace reasons, never asserted from the miss. A disagreement where the cell attempt passed inside the
  clock but the lifted attempt solved late (or the reverse) is run-to-run variance by definition, counted and named.
  **Adopted 2026-10-20 on the fifth review (Leo's go):** solve time in the rule above counts the last successful edit to any file,
  scratch scripts included; that is the preregistered rule and stays PRIMARY. The rich retrodiction showed that its one wrong
  late-solver call, rich_3043 (lifted solve 1,012 s), was an edit to `.pre-commit-config.yaml` after a source fix at 116 s (first
  described here as a scratch-script edit; corrected 2026-10-20 on the replay); the other four
  late solvers made their first repository edit after 480 s. A SOURCE rule, solve time taken over edits under the repository's
  source directories only (`curve.REPO_SOURCE`: rich/ for rich; fastapi/ and docs_src/ for fastapi), is carried in the fastapi
  forecast beside the primary with its own predicted count, interval, criteria and late solvers, declared before any fastapi
  cell data exist. It is evaluated on the same criteria and reported beside the primary; it never decides the branch. On rich,
  after the fact, it gives 37/39 with four late solvers all called right; the paper reports that as a sensitivity reading, not
  as the result.

  **Amended 2026-10-20 on the sixth review (Leo's go):** docs_src/ added to fastapi's source directories. Chronology: the
  mapping was declared as fastapi/ only; after the 20 lifted fastapi outcomes then available were inspected, fastapi_14492 was
  found to have passed by editing only `docs_src/metadata/tutorial001_1_py39.py`, the file the maintainers' reference patch
  changes and whose OpenAPI output the test checks, so docs_src/ is code under evaluation in this repository. The amendment is
  prospective with respect to the fastapi cell outcomes, none of which existed, but it was not chosen before seeing fastapi
  data. The original mapping (`curve.REPO_SOURCE_DECLARED`) and its one changed prediction (fastapi_14492: source-rule fail
  under fastapi/ only, pass at 81 s under the amendment; fastapi_14487 gains a source-edit time and stays a predicted fail)
  are preserved and printed in the forecast file; the amended mapping applies uniformly to all 29 fastapi tasks and is frozen
  there. No primary-rule prediction changes; acceptance is decided on the primary rule as before.
- **A4, the confirmatory precision endpoint (fixed 2026-10-15, unchanged).** On the 68 both-lane tasks:

      D = (P_bf16,calls - P_4bit,calls) - (P_bf16,cell - P_4bit,cell)

  The first bracket is the precision contrast at matched calls (numerics); the second is the contrast as scored
  (numerics plus decode speed); D is what the clock removes. P_4bit,calls for a task is its 4-bit cell attempt when
  that attempt did not end at the clock, and the calls-only rerun when it did. Reported with the 90% cluster-bootstrap
  interval, both brackets beside it, in one paragraph; exploratory language, no p-values. If the fastapi chunks are
  incomplete at the deadline, D is reported on the tasks that have all four arms, the count is stated, and the
  rich-only value is given beside it.

**What had been observed when the primary analysis was added (2026-10-18).** The 4-bit cell r0 on all 68 tasks
(27 resolved; 20 at the call cap, 5 at the clock; S(k) rising from 0.35 at k = 26 to 0.40 at k = 28; P(C) at 0.38 by
5 minutes and 0.40 from 7 minutes; 3.5 s per call); the bf16 cell r0 on the 39 rich tasks (10 resolved, 29 at the
clock, median 12 calls, 17.5 s per call); the bf16 calls-only arm on 15 rich tasks (7 resolved; S(28) 0.47 against
P(480) 0.27; 15.0 s per call); the retrodiction on those 15: predicted 4, observed 5, agreement 14 of 15, the one
disagreement (rich_3043) a cell pass inside the clock against a lifted solve at 1,012 s; D interim +0.067
[-0.133, +0.267]. No fastapi bf16 data of any kind, no 12B data, no 4-bit lifted data. The forecast (A3 ii) is the
part of the headline test that no observation had touched. When the three acceptance criteria were fixed (2026-10-19
00:50 UTC, on review), 20 rich tasks had landed: retrodiction 19 of 20 (predicted 5, observed 6); baselines all-fail
14 of 20 and lifted-only 18 of 20; late solvers 3 (rich_2943 and rich_3278 called correctly, rich_3043 wrongly); still
no fastapi bf16 data.

**Recovery is read against the floor, and net of loops.** A cell failure that passes at the lifted cell may have
passed on a plain rerun at the cell. Recovery is therefore the lifted-cell pass rate among the 20 sampled 4-bit cell
failures minus the cell rerun pass rate among cell failures. The rerun rate comes from the GPU-lane repeats (priority
4 since 2026-10-20) when they exist, otherwise from the API-lane noise floor labelled as a cross-lane proxy. Extra calls in the lifted
attempts are split into recovery episodes and loop calls by the rules below, and recovery is claimed only for the
episodes. Under the reframe the 4-bit lifted subset is first a reading of S(k) beyond the 28-call cap.

**Loop and recovery rules (fixed 2026-10-15, before any calls-only or lifted data; computed from the trace JSON).**
An *edit* is an `edit_file` call, or a `write_file` call to a path inside the repository (not under `/tmp`).
A *loop call* is a tool call whose function name and arguments (canonical JSON) equal those of an earlier call in the
same attempt with no edit between them. Loop calls are counted per attempt; an attempt with three or more is a
repetition-loop attempt (taxonomy class 3). A *recovery episode* is a `run_command` whose result has a non-zero
`exit_code` or an `error` status, followed by at least one edit, followed by the same command string returning
`exit_code` 0. Per arm the tables report: loop calls, recovery episodes, and, for calls after call 12, the share that
are loop calls and the share inside a recovery episode.

**Secondary endpoints.** The noise floor (API lane: pass rate with interval, tasks with mixed outcomes; GPU lane:
agreement between the two cell repeats on the 10-task subset); the size point (12B 4-bit at the cell: seconds per
call, calls reached, pass rate, terminal reasons, beside 31B 4-bit on the same tasks); S(k) beyond 28 from the 4-bit
lifted subset; taxonomy mix per cell; loop calls and recovery episodes per arm; API-lane vs GPU-lane per-task
agreement if the proxy-fidelity arm runs; the cost vector per arm (wall clock, calls, prompt and completion tokens,
GPU hours, API availability as the fraction of calls that needed a retry). Cost is a vector in every table, never
one scalar.

**Validity of an attempt, decided by rule from the record and applied by the regeneration script.**

| Class | Rule on the record | Treatment |
| --- | --- | --- |
| infrastructure_invalid | `outcome.error` set and `total_llm_calls == 0` (sandbox, Docker, adapter down, missing snapshot) | in the intent-to-run table, excluded from valid runs, does not fill its cohort slot; dispatch re-issues the slot |
| provider_interrupted | `outcome.error` set, `total_llm_calls > 0`, error names the provider or transport | in the intent-to-run table, excluded from valid runs, slot re-issued |
| valid | `outcome.error` null, or the terminal reason is the agent's own budget | counted; capped, empty-patch and failed attempts are valid outcomes and are never discarded |
| environment_invalid | the run is listed in `docs/results/excluded-runs.json` with a reason and a DECISIONS line: the verification environment did not test the agent's patch (first case: Kaggle subprocess sandbox without pip, 2026-10-13) | in the intent-to-run table, excluded from valid runs; the run is redone after the environment fix, never silently replaced |

Terminal reasons stay distinct in every table: resolved, tests failed, no edit, call cap, turn cap,
time cap, provider, sandbox, harness. Intent-to-run and valid-run summaries are both reported
wherever the exclusions change a number.

**Missing runs.** Never replaced by a favorable rerun. Outside the lifted-cell arm no failed
attempt is rerun. The lifted-cell arm reruns a seeded random subset fixed before outcomes are inspected; the seed and
the subset go in DECISIONS.md.

**Exposure.** All 129 public tasks are public and the model may have seen the repositories. The
public leaderboard is a sanity check at ±2 tasks, never the instrument. Nothing is claimed about
the hidden set. The tasks the soundness audit excluded are recorded and excluded, and the
exclusion is stated. Precision contrasts are valid only when paired on one serving stack and one
thinking setting.

**Release.** Every attempt record including invalid ones, the regeneration script, `scripts/curve.py` and the
per-attempt `docs/results/curve.csv` behind the curves, the cohort and budget config, the signed forecast, the bundle
digests and the submission ids. Verification is `scripts/verify.py` plus a hand-checked sample of 30 taxonomy
labels with the agreement rate. There is no per-cohort independent review; the records are the audit trail.

**Changes after the freeze.** A line in DECISIONS.md stating what had already been observed at
the time. The endpoint never changes after inspection. The forecast is never edited after signing; if the cell run
is cut by quota, the forecast is published as unmet-by-quota, not withdrawn.

## Cut order and what is never cut

Post-freeze quota (30 free GPU h a week until 2026-10-21, 45 h a week from then with Leo's linked Colab Pro account;
the week's hours ran out at 2026-10-22 01:17 UTC with items (1) to (5) landed and the 12B push refused, the panel
reading 46:01 of 45 h; the quota resets 2026-10-24 00:00 UTC, when (6) is pushed) goes in this order, each item pushed
only when the previous has landed or is in flight: (1) bf16 calls-only rich chunks 4 to 8, done; (2) bf16 calls-only fastapi chunks 1 to 6, done 2026-10-20;
(3) the forecast signed (2026-10-20 07:57 UTC), then the bf16 cell fastapi run (pushed 2026-10-21 01:12 UTC), evaluated
unchanged against the signed criteria; (4) the 4-bit calls-only rerun of fastapi_15661, the one clock-bound 4-bit cell
attempt without one, so D has all four quantities on 68 tasks (pushed 2026-10-21 01:40 UTC); (5) both cells once more on
the seeded 10-task rich subset, about 2.5 h; (6) 12B 4-bit cell rich, about 2 h; (7) the 4-bit lifted subset of 20,
about 2 h. Cut from the end. Reconciled with the arms table 2026-10-21 on the seventh review (this paragraph had kept
the repeats last). If (3) cannot run, the paper reports the rich retrodiction only and says no forecast was tested.

Never cut: the record set and the regeneration script; the noise-floor cohort; the data behind the headline test
(the bf16 calls-only arm and both cells on the tasks it covers); the paired precision contrast at the cell and at
matched calls; raw cost accounting per arm; the submitted bundle being a cell of the grid; the limitations section.

## Failure taxonomy (five classes, fixed)

1. Budget exhausted with no source edit.
2. Tool-call format error (unclosed tool call, malformed arguments, hallucinated tool).
3. Repetition loop (identical tool invocation repeated).
4. Edit made, tests fail.
5. Patch failed to apply or touched protected files.

Deterministic counts come from the trace JSON. Judgment labels, where needed, come
from a Codex worker thread with a signed rubric, with a hand-checked sample and the
agreement rate recorded.

## Known facts from the public record (cite, do not rediscover)

- Public reference: Budget-Fit Single Agent, 0.18 public, 8 min / 28 calls / 80 turns,
  `max_output_tokens` 8192, thinking budget 4096, command timeout 240 s, graph tools
  discouraged. Edit-by-call-12 rule: runs with no edit by call 12 fail 87%.
- Public leaderboard is 58 tasks; identical archives move about 2 tasks; one entrant
  observed a 0.06 spread across identical submissions.
- Our own submission 57127916, the Budget-Fit reference bundle byte-unchanged, scored 0.13 public against its
  author's 0.18: a 3-task movement on 58 for an identical bundle (2026-10-18).
- Hidden set: about 120 tasks from private repositories, split evenly between public
  and private leaderboard; the public tasks come from fastapi (67), rich (48),
  requests (13), httpx (1).
- Scoring runs take 22 to 46 hours; a pending submission blocks the next one.
- The measurement notebook's soundness audit retained 114 of 129 public tasks:
  fastapi 62, rich 44, requests 7, httpx 1.
- L4x4 notebooks consume quota at twice the T4 rate and are restricted to notebooks
  attached to the competition.
- Field size on 2026-10-13: 2,587 teams in the agent track, 201 entrants in the paper track.
- The Kaggle API requires the uploaded archive to be named `submission.zip`; one submission per day.

## API-arm design note (2026-10-13 evening)

Measured on `gemma-4-31b-it` through the free endpoint: about 30 s per call for a trivial prompt,
because the API-served model thinks by default and its thinking cannot be disabled. At the
competition cell (8 min, 28 calls) an API attempt is latency-bound at roughly 15 calls, so the
API arm must be **call- and turn-bounded, not time-bounded**: keep `max_tool_calls 28` and
`max_turns 80`, raise `max_time_minutes` well above the cap (30), and report attempts that would
have been cut at 8 min as a derived column from the per-call timestamps. The 4-bit arm keeps the
competition cell as-is. Latency itself goes into the proxy-fidelity table (T3). Upstream returns
intermittent 500s; the harness retries them, and the adapter log records every request.

### Free-tier limits actually observed (2026-10-13, first attempt)

`GenerateContentInputTokensPerModelPerMinute-FreeTier` = **16,000 input tokens per minute** for
`gemma-4-31b`, with `retryDelay` 35 to 49 s on 429. Published alongside: 15 requests/min, 1,500/day.
The harness's prompts pass 10k tokens within a few turns, so the API arm runs at about one call
per minute once the context has grown, and a 28-call attempt takes 20 to 30 minutes of wall clock.
The harness's own retry layers (LiteLLM x5, ModelRetryPlugin x5) hammer the limit and then fail the
task. Consequences:

- `scripts/gemini-adapter.py` now paces requests to the per-minute token and request windows and
  sleeps the server's retry delay on 429, so the harness sees latency, never a quota error.
- Throughput ceiling: roughly 60 to 80 sequential attempts a day on the 31B in theory; about 25 in
  practice (next note). Enough for the noise floor and one ablation, not for data generation at
  scale. Self-distillation on the API is out.
- A request whose single prompt exceeds 16,000 tokens can never pass. The harness compacts at
  14,336 tokens, so this should be rare; the adapter logs it when it happens.
- **Preferred precision arm:** bf16 `gemma-4-31b-it` on the same L4x4 kernel, if Kaggle Models
  serves it (`google/gemma-4/transformers/gemma-4-31b-it/1` exists, 62 GB). 62 GB of weights fit
  in 96 GB with tensor parallel 4. That makes the precision comparison paired on identical hardware
  and harness, with no API confounds (thinking default, latency, pacing). The API arm is then the
  cheap config-iteration lane and a proxy-fidelity sidebar, not the precision arm.

### Measured on the paced API arm (2026-10-13, requests_7205, bf18)

15 calls in 30 minutes, about 2 minutes per call including pacing and the model's default
thinking; 118,969 prompt tokens, 60,606 cached. Throughput ceiling for the API arm is therefore
about 25 call-bounded attempts a day, sequential. The noise-floor batch (40 attempts) takes two
days on this lane. The L4x4 arm is the primary instrument; the API arm is the slow free lane.

Config-factor noted for the ablation: bf18's prompt does not route scratch scripts to `/tmp`, so
`write_file` reproductions land in the patch (harness gotcha 3). The Rozen prompt has the rule.

### API arm reliability (2026-10-13 evening)

During the first daemon-run attempt, 8 of 10 upstream calls returned 500 or 503 from the Gemini
endpoint within two seconds. The adapter now retries 5xx up to five times with a short backoff and
does not count failed requests against the token window. Expect API-arm attempts to take longer on
bad days; the record carries wall clock and the adapter log carries every retry, so the paper can
state the API arm's effective availability rather than assume it.

### First unattended attempt (2026-10-13 10:05 UTC, requests_6644, bf18): the ceiling bound

14 calls in 28 minutes, then `Agent exceeded session timeout (30.0 min)`: unresolved, 589-byte patch, first edit
at call 9. The adapter log for the window: upstream latency 113 s mean and 278 s max per completed call, 28 HTTP
500s and 3 503s retried by the adapter, 8.2 minutes asleep on retries, no pacing delay at all (prompts were 3k to
7k tokens). The earlier resolved attempt saw 52 s mean upstream latency. So the lane runs at 1 to 2 minutes per
completed call depending on the provider's hour, and a 28-call attempt needs 60 to 90 minutes. Consequences:

- `budgets.api.attempt_wall_clock_minutes` 30 → 90 so the call cap binds, not our ceiling. Throughput is now
  12 to 16 attempts a day, not 25; the noise floor takes three to four days.
- The record stays in the cohort as `valid` with the ceiling visible in `budget.max_time_minutes` and
  `outcome.tool_calls`; the noise-floor pass rate is reported with and without ceiling-bound attempts. It is not
  rerun: an unfavorable outcome is never replaced.
- The graph had ended the pass on this attempt because the attempt tool reported `ok: false` whenever the harness
  set an error. `ok` now means "an attempt happened and its record is on disk"; only `infrastructure_invalid`
  fails the pass. The validity rule is one shared function, `lib/validity.py`, used by dispatch (which slots to
  re-issue) and attempt (what to tell the graph), and later by the regeneration script.

### First full pass (2026-10-13 12:05 UTC, three attempts): the lane measured

`requests_7205` unresolved (19 calls, 50 min, submitted patch, tests failed); `requests_7315` **resolved** (21 calls,
57 min); `requests_7328` unresolved, ceiling-bound at 90 min after 21 calls. Across the three: 2.0 minutes per
completed call in every attempt; upstream latency 115 to 122 s mean, up to 374 s; 40 to 41 HTTP 500s and 24 to 42
429s per attempt retried by the adapter, 21 to 33 minutes asleep on retries; prompts reach 13k to 16k tokens by
the end, so pacing to 16k tokens per minute stays small (0.6 to 2 min). The 7328 attempt lost its last 37 minutes
inside one `run_command` whose script made a live `s.get`, well past the 240-second command timeout: wall clock
lost to hung commands is a cell phenomenon in its own right and is kept as a terminal reason, not cleaned.
Consequences: ceiling 120 minutes, tool timeout 150 minutes; 8 to 14 attempts a day; the noise floor takes about
four days. The graph also ended this pass `failed` on the ceiling-bound attempt because the tool put the
harness's error text under stdout `error`; fixed as `harness_error`.

### First resolved attempt (2026-10-13 04:22 UTC)

`requests_7309`, bf18, API lane, through `tool:gemma-agent/attempt` under the daemon: resolved,
18 tool calls, 20 model calls, first edit at call 5, 6 repetition runs, 30.3 minutes wall clock.
The lane works end to end; it is slow.
