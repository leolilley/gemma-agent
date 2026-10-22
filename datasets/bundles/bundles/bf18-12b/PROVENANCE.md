# bf18: public reference bundle

Reproduced byte-for-byte from "Budget-Fit Single Agent" by Wilmer E. Henao (verracodeguacas),
Apache-2.0, https://www.kaggle.com/code/verracodeguacas/gemma-4-agent-budget-fit-single-agent
as republished in "Gemma 4: Measure Before You Tune" (dmitriigluzdov), whose deterministic
zip of these four files has SHA-256 eb3d5ebf21ce39b186d42937de593e3be62bc329024d59e8693e6c5ce1b20c63 and scored 0.18 on the public leaderboard
(submission 56932359). Our rebuilt zip: eb3d5ebf21ce39b186d42937de593e3be62bc329024d59e8693e6c5ce1b20c63.

This bundle is the campaign's unchanged reference. It is submitted as-is for the first
leaderboard number and used as the control arm in every paired comparison.

## bf18-12b (campaign variant, 2026-10-22)

bf18 with one change: `agent.yaml` `model: gemma-4-12b-it-qat-w4a16-ct` (Google's 12B W4A16 quantization-aware-trained
checkpoint, served by the eval-4bit kernel, which attaches it). Prompt, tools, sampling and budgets are byte-identical to
bf18: the cell budget of 8 minutes and 28 calls. This is the size arm (PLAN.md cohort 3, 12B 4-bit at the cell on the 39
rich tasks); never submitted.
