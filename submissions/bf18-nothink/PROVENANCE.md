# bf18-nothink: the bf18 reference with thinking disabled

Identical to `submissions/bf18` except `configs/sampling.yaml` omits `thinking_config`
(bf18 sets `thinking_budget: 4096, include_thoughts: true`). Reason: the Gemini
OpenAI-compatible endpoint rejects the vLLM-only fields the harness emits for thinking
(`extra_body.chat_template_kwargs`, `extra_body.thinking_token_budget`) and rejects
`reasoning_effort` for Gemma ("Thinking level is not supported for this model"), so a
thinking-on config cannot run on the API arm. This variant is the API arm's control and
is also run on the 4-bit arm so thinking on/off is a measured axis, not a confound.
