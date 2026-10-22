<!-- ryeos:signed:2026-10-20T07:57:08Z:d766cfb5a09456bf4c3b800870a36bb0e4a350039e07c1612038a4dbab034733:SMy2sL6GJOvZVTmcBH0lzSK6CZZKFA4L3VfOhNlSq/iuyLKuWdyT8jNY61Of+f65OWP3EwH3/JeVC4r1mEKVBQ==:82970d668dcc39d38a33264f59b1bc26760be487512a086be87bfd51e2d88b19 -->
---
kind: knowledge
id: gemma-agent/forecast-bf16-cell-fastapi
version: "1.0.0"
---

# Forecast 2026-10-20T07:57:06+00:00: bf16 cell outcomes on 29 fastapi tasks, from the bf16 calls-only attempts

Predicts: bundle=bf18-bf16 site=kaggle-l4x4 tasks=fastapi_11194,fastapi_13537,fastapi_13920,fastapi_14258,fastapi_14301,fastapi_14306,fastapi_14372,fastapi_14419,fastapi_14448,fastapi_14458,fastapi_14463,fastapi_14479,fastapi_14487,fastapi_14492,fastapi_14616,fastapi_14786,fastapi_14794,fastapi_14873,fastapi_14986,fastapi_15280,fastapi_15588,fastapi_15589,fastapi_15661,fastapi_15800,fastapi_5077,fastapi_5624,fastapi_9425,fastapi_9555,fastapi_9753

Rule (PLAN.md A3, PRIMARY, preregistered): a task is predicted to pass under the 8-minute clock if and only if its clock-lifted attempt (bundle `bf18-bf16-calls`, first valid attempt per task) resolved with solve time <= 480 s, solve time being the completion of the last successfully executed edit to any file, from the harness trace (scripts/curve.py). SOURCE rule (secondary, declared 2026-10-20 after the rich retrodiction and before any `bf18-bf16` fastapi data; decides nothing): the same with solve time taken over edits under the repository's source directory only (curve.REPO_SOURCE). Written from 255 records; latest ledger entry ledger-20261020T075656Z.md. No `bf18-bf16` attempt on any task named above exists at writing time (checked against the records and the run directories); the cell run that produces them is pushed only after this file is signed, with this file shipped in its dataset version.

Predicted passes: 7 of 29

Acceptance interval for the observed count: [6, 11] (90%, under per-task disagreement rate q = 0.077 from the rich retrodiction at writing time: agreement 36 of 39 on rich tasks outside this set; count = N_p - X + Y, X ~ Bin(7, q), Y ~ Bin(22, q)).

Source rule, predicted passes: 7 of 29. Source rule, acceptance interval for the observed count: [6, 10] (90%, q = 0.051 from the rich retrodiction under the source rule: agreement 37 of 39). Source rule, task level: agreement at least 25 of 29. Source rule, late solvers: 6 (fastapi_14419, fastapi_14458, fastapi_14794, fastapi_14873, fastapi_15589, fastapi_5624). The source rule is evaluated on the same criteria and reported beside the primary; it does not decide the branch.

Source directories in force (curve.REPO_SOURCE, applied uniformly to every task here): fastapi: fastapi/, docs_src/. As first declared on 2026-10-20: fastapi: fastapi/. fastapi gained docs_src/ on 2026-10-20 after the lifted fastapi outcomes then available were inspected (fastapi_14492 was fixed by editing only a documentation example that the test suite imports and the maintainers' patch changes) and before any `bf18-bf16` fastapi outcome existed; the amendment is prospective with respect to the cell outcomes but was not chosen before seeing fastapi data. Source-rule predictions that differ under the mapping as declared: fastapi_14492 (fail as declared, pass amended at 81 s). No primary-rule prediction depends on the mapping.

Criteria (PLAN.md A3, PAPER.md decision rule); every applicable criterion must pass:
- (a) count: the observed pass count lies in [6, 11].
- (b) task level: per-task agreement is at least 24 of 29 (rich agreement 0.923 minus 0.10).
- (c1) all-fail baseline: agreement exceeds the all-fail predictor (agreement = number of observed fails).
- (c2) lifted-only baseline: agreement exceeds the lifted-only predictor (the lifted outcome with no clock rule; agreement = tasks where the lifted outcome equals the observed one). The rule and that predictor differ only on the late solvers, tasks the lifted attempt resolved after 480 s: 6 here (fastapi_14419, fastapi_14458, fastapi_14794, fastapi_14873, fastapi_15589, fastapi_5624); the rule predicts each of them to fail. (c2) is decided on that set.
A miss on any applicable criterion is a failed forecast; what failed it is a question for the GPU repeats and the trace reasons, never asserted from the miss.

| task | lifted resolved | lifted solve call | lifted solve time (s) | predicted pass | solve time, source (s) | predicted pass, source |
| --- | --- | --- | --- | --- | --- | --- |
| fastapi_11194 | False | 25 | 992 | False | 664 | False |
| fastapi_13537 | True | 11 | 263 | True | 263 | True |
| fastapi_13920 | False | 23 | 1719 | False | 1719 | False |
| fastapi_14258 | False | 25 | 349 | False | 349 | False |
| fastapi_14301 | True | 14 | 406 | True | 406 | True |
| fastapi_14306 | False | 10 | 220 | False | 220 | False |
| fastapi_14372 | True | 11 | 243 | True | 243 | True |
| fastapi_14419 | True | 23 | 1562 | False | 1562 | False |
| fastapi_14448 | False | 16 | 1635 | False |  | False |
| fastapi_14458 | True | 14 | 869 | False | 815 | False |
| fastapi_14463 | True | 7 | 129 | True | 129 | True |
| fastapi_14479 | False | 7 | 193 | False | 193 | False |
| fastapi_14487 | False | 28 | 1670 | False | 1670 | False |
| fastapi_14492 | True | 7 | 81 | True | 81 | True |
| fastapi_14616 | False | 28 | 1045 | False | 1045 | False |
| fastapi_14786 | True | 4 | 72 | True | 72 | True |
| fastapi_14794 | True | 12 | 678 | False | 631 | False |
| fastapi_14873 | True | 24 | 783 | False | 783 | False |
| fastapi_14986 | False | 13 | 1080 | False | 899 | False |
| fastapi_15280 | False |  |  | False |  | False |
| fastapi_15588 | False | 9 | 637 | False | 637 | False |
| fastapi_15589 | True | 16 | 1306 | False | 1165 | False |
| fastapi_15661 | False | 22 | 1738 | False |  | False |
| fastapi_15800 | False |  |  | False |  | False |
| fastapi_5077 | False | 19 | 477 | False | 446 | False |
| fastapi_5624 | True | 14 | 676 | False | 676 | False |
| fastapi_9425 | False | 21 | 422 | False | 422 | False |
| fastapi_9555 | True | 8 | 312 | True | 312 | True |
| fastapi_9753 | False | 15 | 313 | False | 285 | False |

This file is not edited after signing. If the cell run is cut by quota, the forecast is published as unmet-by-quota, not withdrawn.
