# bf18-e4b-verify: the E4B reference bundle with a verify-before-submit prompt

Byte-identical to `bf18-e4b` (agent.yaml, sampling, tool list, eval_config) except `prompts/system.md`, where three
passages change: the core objective (no "minimum number of tool calls (under 8–10 turns)"; states that E4B typically
uses under two of eight minutes and about half its calls), section 4 (check after every edit, correct on failure, do
not submit a fix whose check just failed), and the budget's post-edit line (up to three check-and-correct rounds;
submit by call 26). Motivation and the rule that decides adoption: docs/DECISIONS.md, 2026-11-02 entry
"bf18-e4b-verify".
