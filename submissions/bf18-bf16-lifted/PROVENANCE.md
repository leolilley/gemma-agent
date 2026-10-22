# bf18: public reference bundle

Reproduced byte-for-byte from "Budget-Fit Single Agent" by Wilmer E. Henao (verracodeguacas),
Apache-2.0, https://www.kaggle.com/code/verracodeguacas/gemma-4-agent-budget-fit-single-agent
as republished in "Gemma 4: Measure Before You Tune" (dmitriigluzdov), whose deterministic
zip of these four files has SHA-256 eb3d5ebf21ce39b186d42937de593e3be62bc329024d59e8693e6c5ce1b20c63 and scored 0.18 on the public leaderboard
(submission 56932359). Our rebuilt zip: eb3d5ebf21ce39b186d42937de593e3be62bc329024d59e8693e6c5ce1b20c63.

This bundle is the campaign's unchanged reference. It is submitted as-is for the first
leaderboard number and used as the control arm in every paired comparison.

## bf18-bf16-lifted (campaign variant, 2026-10-13)

bf18 with two changes: the bf16 model alias (as bf18-bf16) and the lifted caps (as bf18-lifted). The bf16
lifted cell; never submitted.
