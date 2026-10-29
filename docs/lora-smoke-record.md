# LoRA route smoke: train on the QAT weights, serve on the scorer's stack, adapter takes effect

Recorded 2026-10-29T04:59Z (Thu 29 Oct 2026, 18:0x NZDT). Leo's go: "okay yeah lets go" (after "confirm the risk here?").
Kernel `leolilley/gemma-agent-lora-smoke-run` v1–v4, 02:36–04:56 UTC, about 115 min of L4x4 session in all.
Data: 160 training and 21 held-out steps (repository-disjoint) from the LoRA kit's native Gemma 31B trajectories
(Gluzdov, CC BY 4.0), as dataset `leolilley/gemma-agent-lora-smoke`; this smoke trained nothing for submission.

## What the route established (v3 training, reproduced in v4)

| step | result |
|---|---|
| Load `gemma-4-e4b-it-qat-w4a16-ct` for training | decompresses to plain bf16 `Linear` layers (compressed-tensors, `run_compressed=False`; `dequantize()` unimplemented, unneeded). The LoRA trains on the weights the scorer serves, not the separate bf16 release. |
| Attach PEFT LoRA r=16, alpha=8 (scale 0.5) to q/k/v/o/gate/up/down of the language model | 34.9M trainable parameters, 516 tensors, names `base_model.model.model.language_model.layers.N.*`, 139.6 MB |
| Train 160 steps (40 updates, lr 1e-4, loss on target tokens only) | train loss 0.49 → 0.32; held-out loss 0.542 (base; the untrained adapter reproduces it exactly) → 0.441. v4 retrained from scratch and reproduced v3 to three decimals. |
| Speed | 9 s a step with the model split across the four L4s as a pipeline: 1,440 s for 160 steps. Too slow for the full set; data-parallel or a quantized base copy per GPU is the next engineering step. |
| Serve (v4) | `adk_submission.VllmServer` with the eval kernel's flags (`gemma4` parsers, max_model_len 32768, bf16, 0.90 memory, `enable_lora`, TP 4) on the QAT base loaded the adapter: 6 min from training end to healthy. |
| Effect probe | first-token top-20 on a fixed prompt: base-to-base drift 0.0, adapter-to-base max delta 1.50 nats; top-1 unchanged ("3" for `len([1, 2, 3])`). Effect far above drift, model still sane. |
| Behaviour | greedy continuations of four held-out prompts, base vs adapter: adapter answers are shorter and go straight to a tool call (`read_file` of the named file); base ran to the length limit on two of four. 8-gram repetition: adapter 0.12–0.17, base 0.11–0.47. No repetition collapse. |

## Fixes on the way

v1: peft 0.21.2 refuses the image's torchao 0.10 while choosing a layer type (probe disabled; torchao unused).
v2: full-vocabulary logits (262k) OOM on one L4; loss now computed on the target positions via `logits_to_keep`.
v3: vLLM OOM'd capturing CUDA graphs because the kernel lacked `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1` (and the
reasoning parser, max_model_len and dtype the eval kernel passes), then waited the full 40 min startup timeout.
v4: those flags, a 20 min timeout; the uploaded v3 adapter was not found under the dataset mount (nested directory), so
v4 retrained, which doubled as a reproducibility check.

## Not established

Any repair-rate effect; anything about the 31B or 12B; hosted-scorer acceptance of a submission with this adapter
(`check_submission.py` and a scorer run are the tests); training throughput at full data size.
