# bf18: public reference bundle

Reproduced byte-for-byte from "Budget-Fit Single Agent" by Wilmer E. Henao (verracodeguacas),
Apache-2.0, https://www.kaggle.com/code/verracodeguacas/gemma-4-agent-budget-fit-single-agent
as republished in "Gemma 4: Measure Before You Tune" (dmitriigluzdov), whose deterministic
zip of these four files has SHA-256 eb3d5ebf21ce39b186d42937de593e3be62bc329024d59e8693e6c5ce1b20c63 and scored 0.18 on the public leaderboard
(submission 56932359). Our rebuilt zip: eb3d5ebf21ce39b186d42937de593e3be62bc329024d59e8693e6c5ce1b20c63.

This bundle is the campaign's unchanged reference. It is submitted as-is for the first
leaderboard number and used as the control arm in every paired comparison.

## bf18-bf16 (campaign variant, 2026-10-13)

bf18 with one change: `agent.yaml` `model: gemma-4-31b-it` (the bf16 checkpoint, served by the eval-bf16
kernel with tensor parallel 4). Prompt, tools, sampling and budgets are byte-identical to bf18. This is the
precision arm's bf16 cell; never submitted.

## bf18-bf16-calls (campaign variant, 2026-10-14)

bf18-bf16 with one change: `eval_config.yaml` `max_time_minutes: 30`, so the 28-call cap binds and the clock does not
(bf16 decodes at about 15 tok/s on 4×L4 against 35 for 4-bit; 28 bf16 calls average about 13 min). Prompt, tools,
sampling, model and every other budget are byte-identical to bf18-bf16. This is the bf16 side of the calls-only precision
contrast (DECISIONS 2026-10-14); never submitted.
