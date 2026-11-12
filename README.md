# gemma-agent

**Paper:** *Where the histories come from*, source [docs/paper-draft-lora.md](docs/paper-draft-lora.md), PDF [docs/paper/paper.pdf](docs/paper/paper.pdf).
**Records:** every attempt record, trace, adapter and signed item is in the Kaggle dataset [leolilley/gemma-agent-records](https://www.kaggle.com/datasets/leolilley/gemma-agent-records); adapter weights and run outputs are there, not in this repository.

A campaign on the installed local RyeOS node for two Kaggle competitions:

| Track | Deadline (UTC) | Prize | Our stance |
| --- | --- | --- | --- |
| [Gemma 4 Developer Agent Paper Track](https://www.kaggle.com/competitions/gemma-4-developer-agent-paper) | **2026-11-12 23:59** | $35k: $15k overall, $10k best new resource, $10k best new application | **Primary.** A measurement paper with every number regenerable from signed records. |
| [Gemma 4 Developer Agent](https://www.kaggle.com/competitions/gemma-4-developer-agent) | entry 2026-11-25, final 2026-12-02 | $65k: $37k / $18k / $10k | **Secondary.** An ADK skill package developed through existing RyeOS workflows, replacing the small-LoRA plan under the dated amendment below. [Protocol and feasibility gates](docs/adk-skill-study.md). |

The project charter is this file. The four-week plan, the frozen analysis plan and the
cohort discipline are in [docs/PLAN.md](docs/PLAN.md). The paper outline and its claim matrix
are [docs/PAPER.md](docs/PAPER.md).
Decisions are logged in [docs/DECISIONS.md](docs/DECISIONS.md). Nothing in `.tmp/`
is authoritative.

The sections below through "What this is not" preserve the original measurement
charter, including its starting budgets. The main-track amendment at the end applies
only to the separate skill study; the frozen paper plan and claims remain independent.

## What we are measuring

**Thesis.** The agent loop normally hides quantization damage by recovering from its
own mistakes. Under the hard per-task budget this competition imposes, that slack
disappears and the damage shows up in the score and in nameable failure modes.
Model size is the third way to buy slack, and the one a developer on everyday
hardware controls.

**Design.** One grid, one failure taxonomy, one record format.

| Axis | Levels |
| --- | --- |
| Precision | bf16 and the 4-bit QAT checkpoint the scorer runs, paired on Kaggle's free L4x4 notebooks; the free Gemini API is a proxy lane, not the precision arm |
| Turn and time budget | the competition cell (8 min, 28 calls, the public 0.18 reference); a lifted cell (16 min, 40 calls) for reruns of every cell failure, read against rerun variance |
| Model size | 31B, 12B, E4B, all 4-bit on Kaggle's free L4x4 notebooks; 12B with turns raised to fill the time cap |

Backbone, not headline: run-to-run variance and intraclass correlation on every
comparison, per the ICLR 2026 randomness paper and the ICC paper. The public
leaderboard is 58 tasks and moves about two tasks between identical submissions
(reported by two independent entrants), so it is a sanity check, never the instrument.

## Rules

These exist because the ARC and Kaggriculture campaigns were lost the same way:
"run one experiment" became "qualify the next piece of substrate first".

1. **This is a project, not substrate work.** It runs on the installed node
   (`ryeos 0.5.102` at `~/.local/share/ryeos`) as it is. RyeOS pieces are Python
   tools, graphs, knowledge items and signed config. Anything needing a Rust change,
   a new provider family, a Render worker, a Modal GPU node or tinygrad is out of
   scope for the duration. If a step requires one, stop and say so.
2. **The harness is the instrument and is never reimplemented.** The agent loop runs
   inside the official `swegemma` harness, installed from the competition wheelhouse,
   with its model routed to an OpenAI-compatible endpoint. RyeOS runs attempts as
   tool invocations, enforces the daily budgets, keeps the records and recovers the
   loop. The campaign's own cognition (failure labeling, variant proposals, analysis)
   runs in directives or Codex worker threads; the subject under test is a black box.
3. **No teacher distillation.** Nothing produced by another provider's model may select or
   shape Gemma's training data. Training trajectories are selected by one rule: the
   hidden tests passed. Failure labels live in a separate knowledge namespace.
4. **Measure before you build.** The first deliverable is a leaderboard number from
   the public 0.18 reference bundle, submitted unchanged. The second is one attempt
   on one task through the harness on this machine with a signed record.
5. **The paper angle is frozen on 2026-10-22** (moved from 10-20 on 2026-10-18 for the reframe; DECISIONS). After that, weeks three and four are
   measurement and writing, not search.
6. **One hypothesis, more repeats.** Three cohorts, then only repeats, lifted-cell reruns, or
   the same arms on more tasks. A new axis needs a decision line and an explicit go. The ARC
   paper answered each null with a new protocol and never got a result; we do not.

## Budgets (all free tiers; money is not the constraint, quota is)

| Resource | Limit | Used for |
| --- | --- | --- |
| Gemini API, `gemma-4-31b-it` | 16,000 input tokens/min, 15 requests/min, 1,500 requests/day per model (measured 2026-10-13); 2 min per completed call plus 20 to 33 min of provider retries per attempt, so 8 to 14 call-bounded attempts/day under a 120-minute ceiling | noise floor, proxy fidelity, configuration iteration |
| Kaggle scoring | one submission at a time; runs take 22 to 46 h | hidden-set score |
| Kaggle L4x4 notebooks | 30 quota-hours/week at 2x consumption, competition-attached only | 4-bit arms, QLoRA |
| Modal starter | $30 credit/month | spillover QLoRA |
| This machine | ~2 GB headroom while in use; 12 GB free on `/home` | sandboxes, tools, records |

## Layout

```
README.md               charter (this file)
docs/                   PLAN.md gates, PAPER.md outline, DECISIONS.md log, measurement notes
data/                   competition files (gitignored): tasks.jsonl, snapshots/ subset,
                        graphs/, embeddings/, wheels/ (sandbox test deps), sandbox/, sample_submission/
wheelhouse/             harness wheels from the Kaggle wheelhouse dataset (gitignored)
submissions/            our agent bundles, one directory per version; bf18/ is the public reference
results/                raw swegemma --results-dir output (gitignored); records are derived from these
.ai/                    RyeOS project: tools/gemma-agent/, graphs/, knowledge/gemma-agent/, config/, directives/
scripts/                top-level kicks only (fetch data, package a bundle); no logic that belongs in a graph
```

## Day-one checks (must pass before anything else is built)

1. `submissions/bf18/` packaged and submitted unchanged; a leaderboard number exists.
2. One `swegemma eval --sandbox subprocess` attempt on one `requests` task on this
   machine, model routed to the Gemini endpoint, results dir written.
3. The Gemini OpenAI-compatible endpoint serves a Gemma 4 model with tool calling,
   or the route is recorded as unavailable and the API arm is redesigned.
4. The vault delivers `GEMINI_API_KEY` to a tool subprocess on the installed node.

## What this is not

Not a RyeOS showcase. The paper gets one method paragraph and a download link for
the records. Not an attempt to win the agent track. Not a reason to touch the
substrate. Not a review process: verification is one regeneration script and a
hand-checked label sample, not an independent review per cohort.

## Main-track amendment — 2026-10-28

The intended main-track direction is an ADK skill package, replacing the small-LoRA
plan. The [skill-study protocol](docs/adk-skill-study.md) owns its rules, the bounded
CPU data check and subsequent gates. The original charter text above is preserved.

1. **Execution.** Gemma writes candidate procedures; existing RyeOS project workflows
   test and retain them. The submitted agent consumes ordinary ADK skills. This work
   uses the installed substrate and the official harness.
2. **Source material.** Valid failed Gemma development attempts may join successful
   attempts under a declared input rule. This is a skill-study change from the old
   passing-attempts-only training rule, not a revision to the measurement campaign.
   Gemma authors the repair knowledge; humans and AI assistants may supply schemas,
   plumbing and evaluation. Reference solutions and evaluation outcomes stay separate.
3. **Evidence.** Compare the reference agent with the selected package at the same
   budget, charging skill overhead. Data feasibility and actual ADK skill execution
   precede a larger study. Local Docker is the independent grading path; Kaggle agent
   workspace compatibility remains a separate requirement.
4. **Priority.** The paper must not depend on this work. Preserve its endpoints,
   forecast and chronology and finish it promptly. The original budget table is
   historical; the Colab Pro change is logged on 2026-10-21, and later GPU work needs
   a current allocation check. The next data check uses CPU only.
