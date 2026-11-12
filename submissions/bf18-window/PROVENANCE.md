# bf18-window: bf18 with one prompt section changed

`bf18` (the submitted reference bundle, PROVENANCE there) with section 1 of `prompts/system.md` replaced: one search, then
`read_file` in 150-line windows around the hit (the harness's read_file returns at most 150 lines; bf18's cap-out attempts
read 30-line windows and truncated whole-file heads and grep about eight times before their first edit, at call 19 against
call 6 on resolved attempts). agent.yaml, configs, eval_config.yaml and the rest of the prompt are byte-identical to bf18.
Pre-registered in docs/DECISIONS.md, 2026-11-05T21:00Z. Data-only until the rule there is met.
