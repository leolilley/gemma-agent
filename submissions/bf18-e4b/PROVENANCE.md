# bf18-e4b: the reference bundle on the E4B checkpoint

Byte-identical to `bf18` (prompts, sampling, tool list, eval_config) except `agent.yaml` names
`gemma-4-e4b-it-qat-w4a16-ct` instead of `gemma-4-31b-it-qat-w4a16-ct`. It is the base arm of the
LoRA cycle: the adapter arms are this bundle plus `adapters/<name>` and the `adapter:` line.
