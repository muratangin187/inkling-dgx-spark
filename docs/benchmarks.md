# Benchmarks on one DGX Spark

Measurements were collected on one NVIDIA GB10 in November 2026. The daily profile is the [pinned SGLang recipe](../bin/serve). Experimental results below describe specific workloads and configurations. Latencies from separate workloads cannot be added.

## Reproduce the baseline

Run the single-session panel against an idle server. On the DGX, use the loopback API directly. From the client, use the SSH tunnel. With a custom client port, change `base_url` in a copy of the benchmark configuration.

```bash
mkdir -p runs/baseline
(
  cd runs/baseline
  python3 ../../benchmarks/vendor/bench.py run ../../benchmarks/baseline.json --skip-cN --reps 6
)
python3 benchmarks/smoke.py --size half
python3 benchmarks/smoke.py --size near-full
```

The baseline panel uses six fixed prompts and the server’s usage counters. Its approximate decode rate includes emitted reasoning tokens. The two cold prefill probes had **4,930 and 38,484 actual input tokens**. For profile changes, warm each server, bracket the candidate with two controls, and run image, tool, and long-context checks alongside timing. The vendored harness is the code from the [pinned inference-optimisation skill](https://github.com/sybil-solutions/inference-optimisation-skill/tree/fda968e7fc1996aa23c67139c755fe541c9d4be0).

## SGLang and client results

| Change | Result | Decision |
| --- | --- | --- |
| DSpark verification width | 38.36 / 45.20 / 38.27 tok/s in the fixed six-prompt panel. Image, tools, and 495,496-input-token retrieval passed. | Lab profile. Whole-agent gains varied by task. |
| Packed BF16 projections | 45.15 / 47.05 / 45.25 tok/s. One workload regressed. | Lab profile. |
| Restricted draft vocabulary | 45.12 / 48.92 / 45.06 tok/s. The window used 65,536 draft IDs. The long OpenCode streaming check did not finish. | Lab profile. |
| Fused Page16 prefill | About 22% more prefill throughput at 38,484 actual input tokens. The 495,496-token multimodal check passed. One ~74K-input OpenCode task produced more tokens and took 116.6 s vs 91.1 s. | Lab profile. |
| Packed target head and MXFP8 draft linears | About 2% gains in independent short panels. | Lab components. |
| Suppress OpenCode title request | First coding output 26.35 s vs 49.52 and 48.96 s. Verified completion 93.75 s vs 160.90 and 125.45 s. | Included in the launcher. |
| OpenCode low reasoning effort | 31.98% median wall-time reduction across three image-to-code tasks relative to each fixture’s mean of two medium controls. All nine runs passed their functional checks. | Opt-in `opencode-low`. |
| Same-chat prefix reuse | First visible output 12.6 → 2.7 s on a ~32K image-and-text archive. About 31.6K cached tokens on the follow-up. | Included in standard serving; the chat must keep its prefix. |

A separate 64K comparison gave **30.35 s** median time to first streamed token on cold vLLM and **22.42 s** on cold SGLang. For a warm shared prefix, those medians were **0.96 s** and **0.25 s**. These measurements include the first reasoning token and used each runtime’s own profile. Adding a new image to a later turn can invalidate SGLang’s multimodal prefix reuse. We also verified a separate ~478K image-and-text request and its cached follow-up on SGLang.

## vLLM FP8 KV cache

This experiment varied only the KV cache datatype, with probabilistic DSpark, one active request, and the same pinned checkpoint. We used fixed seeds, report actual streamed token IDs, exclude the first burst from the decode clock, and compare response wall time separately.

| Metric | BF16 | FP8 | BF16 repeat |
| --- | ---: | ---: | ---: |
| Cold retrieval with 265,824 input tokens | 177.79 s | 187.35 s | 177.46 s |
| Warm coding with 266,093 input tokens | 38.87 tok/s | **47.30 tok/s** | 38.79 tok/s |
| Warm coding response time | 61.41 s | 62.94 s | 61.91 s |
| Completion tokens | 2,142 | 2,658 | 2,142 |
| Executable program passed all checks | Yes | Yes | Yes |
| Reported KV pool capacity | 1,149,845 tokens | 1,913,672 tokens | 1,149,022 tokens |

The pool held about **66% more tokens** with the same ~51 GiB KV budget. The experiment still used a 524,288-token request limit and one scheduled sequence. All three warm coding requests logged the same two Triton compilation warnings before their first token.

FP8 also passed a 495,496-input-token screenshot and retrieval check in **489.05 s**. A follow-up with **495,765 input tokens**, including **495,488 cached tokens**, returned code that passed four fixtures and its self-test. It streamed at **39.99 tok/s** and finished in **56.35 s**. There was no BF16 timing control at this depth.

Two BF16 controls reproduced all four frozen OpenCode response token sequences. FP8 produced different sequences on all four. The finish types matched, with three valid tool calls and one length-limited response in each profile. This was an API replay of OpenCode requests and did not time full agent tasks. Greedy draft sampling also failed to improve the shorter single-session panel and slowed the long coding request. Neither variant is part of the daily launcher.

## Other checks

- Basalt’s separately converted NVFP4/Q8 model pack reached 52–110 tok/s on short tests depending on workload. It passed text, image and tool checks near 495K context. The converted pack and different serving engine stayed experimental.
- Grouped MXFP8 projections improved isolated timings and preserved the tested layer’s numerical output. The full-layer benchmark showed no meaningful gain.
- The lossless KV compression probe reduced storage bytes and increased read latency. A presharded loader failed on a scale tensor shape mismatch.
- A previous concurrency stress test reached about 251–253 aggregate tok/s with 32 short inputs, around 12–13 tok/s per stream. This measurement used a different vLLM profile. It does not establish 200 tok/s for a coding session.
- With the observed FP8 coding workload, the current eight-token verification step averaged roughly 100 ms of request wall time. Even perfect acceptance at that fixed cost would produce about 80 tok/s. This calculation applies to those measurements and that verifier width.

The numerical records in [results.json](../benchmarks/results.json) are drawn from benchmark outputs with an allowlist of measurement fields. They include no conversation archives, generated code, or credentials.

## Experiment recipe and source pins

The SGLang width and kernel experiments build on the [pinned knapcio work](https://github.com/knapcio/Inkling-2-Mini-1x-DGX-Spark/tree/0e465513510fa98cce2be5996398d197e27c6e8d). Use the appropriate upstream setup for those overlays. The included server launcher runs the qualified daily profile.

For the vLLM comparison, use the experimental Inkling image from the [vLLM recipe](https://recipes.vllm.ai/thinkingmachines/Inkling-2-Mini). The image tag at measurement time was `vllm/vllm-openai:inkling-2-mini` and Docker recorded registry digest `sha256:c2dc759f5984f69c1e2418c6af7c26eb756f5ed2a8934c768ae1253645d1449e`. Verify that the local image matches before reproducing the measurements. Use a dedicated maintenance window with the daily engine stopped and enough host memory for the model and cache. Then run the below in order with `KV_DTYPE=bfloat16`, `KV_DTYPE=fp8`, and `KV_DTYPE=bfloat16` again. Stop each run before starting the next.

```bash
KV_DTYPE=bfloat16
docker run --rm --name inkling-kv-lab --runtime nvidia --gpus all --ipc=host \
  -p 127.0.0.1:18139:8000 \
  -v "${HF_HUB_CACHE:-$HOME/.cache/huggingface/hub}:/root/.cache/huggingface/hub:ro" \
  -e HF_HUB_OFFLINE=1 vllm/vllm-openai:inkling-2-mini \
  thinkingmachines/Inkling-2-Mini \
  --revision ed40e7da625f594fc9c4990f5f66be24873c22d7 \
  --served-model-name thinkingmachines/Inkling-2-Mini \
  --max-model-len 524288 --max-num-seqs 1 --gpu-memory-utilization 0.60 \
  --kv-cache-dtype "$KV_DTYPE" --enable-prefix-caching --enable-prompt-tokens-details \
  --reasoning-parser inkling --enable-auto-tool-choice --tool-call-parser inkling \
  --speculative-config '{"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"probabilistic","rejection_sample_method":"standard"}'
```

The published SGLang smoke test exercises the same synthetic archive and screenshot family as the lab. The [numerical record](../benchmarks/results.json) also includes the separate executed coding follow-up and API replay results from the lab.
