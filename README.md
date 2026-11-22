# Inkling 2 Mini on one DGX Spark

Run Thinking Machines Lab’s **Inkling 2 Mini** in OpenCode or Pi from your Mac or Linux workstation. The serving recipe pins SGLang and the model revision, exposes an OpenAI-compatible endpoint over SSH, and supports **524,288 total context tokens**, screenshots, and tool calls on a single DGX Spark (GB10).

## What moved the needle

| Experiment on one DGX Spark | Measured result | Where it runs |
| --- | ---: | --- |
| Reuse a 32K multimodal prefix in the same chat | **12.6 → 2.7 s** to first visible output | Daily setup |
| Skip OpenCode’s automatic title request | **35% less wall time** on one verified ~74K-input coding task | Daily launcher |
| Set OpenCode reasoning effort to low | **32% median task time reduction** across three image-to-code tasks, with all checks passing | Opt-in launcher |
| Tune DSpark verification width | **38.36 / 45.20 / 38.27 tok/s** across baseline, candidate, and repeat | Experimental SGLang |
| Use FP8 KV cache for warm, 266K-input coding | **38.87 / 47.30 / 38.79 tok/s**, with all generated programs passing | Experimental vLLM |

The 266K FP8 response generated more tokens and took **62.9 s**. The two BF16 responses took **61.4 and 61.9 s**. FP8 also passed a separate **495K multimodal retrieval and executable coding check**, with no matched BF16 timing at that depth. The study did not reach 200 tok/s in a single session. See [benchmark details](docs/benchmarks.md) for workload, controls, and why experimental changes stayed out of the daily profile.

## Set up the DGX Spark

You need Docker with the NVIDIA runtime, `flock`, and the Hugging Face CLI (`hf`) on the Spark. This profile is sized for one active request and uses a 60% static memory fraction. Leave enough memory for loading the checkpoint and avoid competing GPU workloads during benchmarking. The first start can take several minutes.

On the **DGX Spark**:

```bash
hf download thinkingmachines/Inkling-2-Mini \
  --revision ed40e7da625f594fc9c4990f5f66be24873c22d7 \
  --cache-dir "${HF_HUB_CACHE:-$HOME/.cache/huggingface/hub}"
```

On the **client machine**, configure a working SSH alias (`dgx` below) and install Python 3, curl, and your agent CLI. This recipe was validated with OpenCode **2.3.2** and Pi **0.84.2**.

```bash
git clone https://github.com/muratangin187/inkling-dgx-spark.git
cd inkling-dgx-spark

export INKLING_SSH_HOST=dgx
ssh "$INKLING_SSH_HOST" 'mkdir -p ~/.local/share/inkling-dgx-spark'
scp bin/serve "$INKLING_SSH_HOST:.local/share/inkling-dgx-spark/serve"

./bin/inkling-dgx start
python3 benchmarks/smoke.py
```

The smoke test calls a tool, returns its result to the model, then checks that the model can read a synthetic screenshot and recover three exact values from the accompanying text. `--size half` and `--size near-full` exercise longer inputs. They can take several minutes.

### Use it from a project

From your project directory, call the launcher using the path to the checkout:

```bash
/path/to/inkling-dgx-spark/bin/inkling-dgx opencode
/path/to/inkling-dgx-spark/bin/inkling-dgx opencode-low
/path/to/inkling-dgx-spark/bin/inkling-dgx pi @/path/to/screenshot.png "Investigate this behavior"
```

OpenCode launches with a process-local provider configuration and the title agent disabled. The standard launcher uses medium reasoning effort. Pi keeps its own provider settings and session data. Both clients reserve up to **24,576 output tokens** from the shared context budget.

For other clients, use `http://127.0.0.1:18121/v1` and model ID `thinkingmachines/Inkling-2-Mini`. `INKLING_LOCAL_PORT` changes the client-side forwarding port. `OPENCODE_BIN` and `PI_BIN` select binaries outside `PATH`.

| Command | Effect |
| --- | --- |
| `bin/inkling-dgx status` | Check the DGX container state |
| `bin/inkling-dgx logs` | Read startup and serving logs |
| `bin/inkling-dgx disconnect` | Close this SSH tunnel |
| `bin/inkling-dgx stop` | Stop this repository’s container and close the tunnel |

The DGX host port binds to loopback. The launcher forwards a loopback port on the client over SSH. Images, messages, system prompts, and tool history share the context budget. Keep a large archive in the first turn and send follow-up questions as new turns in the same chat so the engine can reuse its prefix.

## Reproduce and inspect

- [Benchmark protocol and full results](docs/benchmarks.md)
- [Sanitized measurement data](benchmarks/results.json)
- [Pinned serving flags](bin/serve)
- [OpenCode profile](config/opencode.json) and [Pi profile](config/pi-models.json)

The repository contains launch scripts, client profiles, synthetic fixtures, and numerical results. Download the checkpoint from the [Thinking Machines Lab model card](https://huggingface.co/thinkingmachines/Inkling-2-Mini).

## Credits

The serving recipe builds on [SGLang’s Inkling support](https://www.sglang.io/blog/inkling-2-small-day0-support). The measurement loop uses [Sybil Solutions’ inference-optimisation skill](https://github.com/sybil-solutions/inference-optimisation-skill). The SGLang verification-width and kernel experiments draw on [knapcio’s DGX Spark work](https://github.com/knapcio/Inkling-2-Mini-1x-DGX-Spark). The vendored benchmark retains its [upstream MIT license](benchmarks/vendor/LICENSE). This repository’s original code is [MIT-licensed](LICENSE).
