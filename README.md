# TENSELERATE

**A maintained llama.cpp fork for serving large models on small and unusual NVIDIA cards**
(CMP 170HX, RTX 3060, Turing) as the model provider for agent frameworks such as Hercules.
Tracks upstream `ggml-org/llama.cpp` (last sync: upstream `67672dc5`, 2026-09-07) and adds
streaming virtual memory (SVMI), all-integer CUDA builds, prompt-cache serving flags, and a
measured performance record for the hardware it targets.

[![CI](https://img.shields.io/github/actions/workflow/status/mintoriakamoto/TENSELERATE-/tenselerate-engine.yml?branch=main&label=CI)](https://github.com/mintoriakamoto/TENSELERATE-/actions/workflows/tenselerate-engine.yml)
[![Release](https://img.shields.io/github/actions/workflow/status/mintoriakamoto/TENSELERATE-/release.yml?branch=main&label=Release)](https://github.com/mintoriakamoto/TENSELERATE-/actions/workflows/release.yml)
[![Latest](https://img.shields.io/github/v/release/mintoriakamoto/TENSELERATE-?label=latest&color=brightgreen)](https://github.com/mintoriakamoto/TENSELERATE-/releases/latest)
[![Issues](https://img.shields.io/github/issues/mintoriakamoto/TENSELERATE-)](https://github.com/mintoriakamoto/TENSELERATE-/issues)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Status

| | |
| --- | --- |
| Maintained | Yes. Every native push to `main` is CI-built (CPU + CUDA sm_80/86) and auto-published as a `main-b<N>-<sha>` release with CPU and static-CUDA tarballs. |
| Latest release | [`main-b11044-073223f`](https://github.com/mintoriakamoto/TENSELERATE-/releases/latest) (2026-09-08) |
| Upstream | Merged with `ggml-org/llama.cpp` master `67672dc5` (2026-09-07); weekly sync tracked in [#59](https://github.com/mintoriakamoto/TENSELERATE-/issues/59) |
| Roadmap | [docs/ROADMAP.md](docs/ROADMAP.md) and the [open issues](https://github.com/mintoriakamoto/TENSELERATE-/issues) |
| Changelog | [CHANGELOG.md](CHANGELOG.md) |
| Reference box | CMP 170HX 40 GiB: Qwen3.8-27B Q4_K_M **pp4096 856 tok/s**; live Hermes serve **~737 t/s prefill, ~60 t/s decode** (MTP n-max 4, `MMVQ_MAX=3`, 8×256K unified). See [PERFORMANCE_ANALYSIS.md](PERFORMANCE_ANALYSIS.md). |

## Install from GitHub (CUDA **12.8** required)

Driver 13.x is OK. The **toolkit must be 12.8**. PATH `nvcc` 12.4 or a CUDA 13 toolkit will not do — the preset fails closed.

```bash
git clone https://github.com/mintoriakamoto/TENSELERATE-.git TENSELERATE
cd TENSELERATE

# toolkit: https://developer.nvidia.com/cuda-12-8-0-download-archive
test -x /usr/local/cuda-12.8/bin/nvcc
export PATH=/usr/local/cuda-12.8/bin:$PATH
export CUDAToolkit_ROOT=/usr/local/cuda-12.8

cmake --preset deploy-cmp170hx
cmake --build build-deploy-cmp170hx -j$(nproc) --target llama-server

MODEL=/path/to/Qwen3.8-27B-*.gguf bash scripts/boot-cmp170hx.sh
# → http://127.0.0.1:8083/v1
```

Preset: `FORCE_MMQ=ON`, `FORCE_CUBLAS=OFF`, `DISABLE_DP4A=ON`, `sm_80-real`, nvcc **12.8**, rpath **12.8**.

Boot: `GGML_CUDA_MMVQ_MAX=3`, `-c 262144 -np 8 -kvu`, MTP n-max 4, q8_0, `-b 8192 -ub 2048`. Do not set `NO_MMVQ=1` for one operator.

**Hermes:** `base_url: http://127.0.0.1:8083/v1` — [docs/hermes.md](docs/hermes.md).  
**Hercules:** `NP=8 CTX=262144 PORT=8083 MMVQ_MAX=3 MTP=4 bash scripts/hercules_serve.sh MODEL.gguf` — [HERCULES.md](HERCULES.md).

This fork is **not** stock llama.cpp. If you also run upstream llama.cpp, keep it on another port (we use **:8082**).

Prebuilt CI tarballs: `FLAVOR=cuda scripts/tenselerate-update.sh --binary` (also CUDA 12.8 sm_80/86).

More: [docs/dev-workflow.md](docs/dev-workflow.md), [docs/physics.md](docs/physics.md), [PERFORMANCE_ANALYSIS.md](PERFORMANCE_ANALYSIS.md).

## What this fork adds (SVMI and friends)

This fork implements **SVMI (Streaming Virtual Memory Inference)**: the GPU is treated
as a cache over a host-RAM weight store. The design goal is 70B-class models in **under
20 GB of VRAM with all matrix math on the GPU** and token-identical output. What is
measured on this branch: the pinned store's upload bandwidth and the ancestor's +64%
prefill on an RTX 3060; the streamed-decode floor for a 70B is modeled at ~0.9 tok/s
at batch 1 (PCIe physics) and has not been run here. The 27B on the 170HX, which
fits in VRAM, does not use streaming at all.

What's in this branch (all opt-in, off by default):

| Feature | Flag / env |
| --- | --- |
| Pinned host weight store — mmap'd weights are page-locked so H2D uploads run as real async DMA (~6–7 → ~20+ GB/s) | `GGML_CUDA_REGISTER_HOST=1` |
| Weight streaming — uploads for upcoming layers are enqueued on dedicated queues (one per DMA copy engine) into a staging ring, overlapping PCIe transfers with compute; generalizes MoE expert prefetch to dense models | `--stream-weights N` |
| Streamed decode — keep all matmuls on the GPU at any batch size instead of computing host-resident layers on the CPU | `--stream-decode` |
| Residency planner — split a VRAM budget between KV cache, staging ring, and resident weights; emits ready-to-use flags | `scripts/svmi-plan.py` |
| Token-identity check — greedy diff (and optional perplexity) of streamed vs baseline output | `scripts/svmi-verify.sh` |
| Benchmark harness + compressed-transport feasibility study | `scripts/svmi-bench.sh`, `scripts/svmi-entropy.py` |
| BitSpec feasibility — acceptance rate of a low-bit resident self-draft (novel; see research notes) | `scripts/svmi-bitspec.py` |
| MAVM + CTX-VM fleet planner — how many agents at 131K/256K context fit one GPU: shared weights (O(1) in agents), paged KV (host-resident context, landmark page table + hot window in VRAM), shared-prefix dedup, idle spill | `scripts/svmi-fleet.py` |
| ARBITER routing optimizer — compute follows memory: GPU + CPU verify their own resident shares concurrently, zero weight bytes on PCIe, split solved by bandwidth arbitrage (modeled 5× stock partial offload for 70B) | `scripts/svmi-arbiter.py` |
| **INT8 mixed quant** — `Q4_K_M` body with `q8_0` attention (~5.6 bpw): the resident half of the model runs on integer MMQ kernels, which is all a card with no usable FP16 has | `llama-quantize model.gguf out.gguf INT8` |
| CMP card support — build/quant advice per card, HBM2e unlock detection (170HX: 8→64 GiB, 10→40 GiB), and a probe for the dp4a-vs-tensor-core throttle crossover | `scripts/svmi-gpucheck.py`, `scripts/svmi-cmpbench.sh` |
| **All-integer CUDA build** — every matmul on MMQ, no cuBLAS FP16 GEMM, dp4a emulated via prmt+dp2a; CI-built for sm_70/80/86 | `cmake --preset cmp170hx-int8` (also `cmp90hx-int8`, `cmp100-210-int8`) |
| Update channel — `main` auto-publishes `main-b<N>-<sha>` releases; this is the client that checks and applies them, with or without a git clone | `scripts/tenselerate-update.sh` |
| **GQA-packed vector attention** — quantized-KV decode reads each K/V byte once per KV head instead of once per Q head; auto above 32K KV | `GGML_CUDA_FATTN_VEC_GQA=-1\|0\|1` |
| **MMVQ width cap** — keep multi-slot decode on dp4a MMVQ up to N streams, hand wider batches to MMQ (crossover measured at 3-4 on the 170HX) | `GGML_CUDA_MMVQ_MAX=N`, `GGML_CUDA_NO_MMVQ=1` |
| **K-cache mean centering** — per-(head,channel) bias subtracted before Q4_0 K quantization; softmax-invariant, better fidelity | `--kv-mean-center FILE`, `tools/kv-mean-center` |
| **Bounded attention window** — the 16 attention layers of the GDN hybrid see a sliding window + pinned sinks; the 48 GDN layers carry the rest, so KV per slot is O(window) and sequences are unbounded: 16 slots at 32K or 9 at 64K on the 40 GiB card ([design](docs/bounded-window-serving.md)) | `--attn-window N --attn-sinks S`, `LLAMA_ATTN_WINDOW`, `LLAMA_ATTN_SINKS` |
| **Agent serving flags** — RAM prompt cache, idle-slot caching, LCP slot selection, slot save/restore, `--reasoning-effort`, MTP draft with sampling guards; one launch builder emits them | `tenselerate boot`, `scripts/hercules_serve.sh`, `scripts/hercules_slots.sh` |

```bash
# 70B Q4_K_M on a 20 GB budget:
python3 scripts/svmi-plan.py model-70b-q4_k_m.gguf --vram-budget 19
# ...then run the flags it prints

# or target a specific consumer card (sets VRAM, PCIe bandwidth, and queue count):
python3 scripts/svmi-plan.py model-70b-q4_k_m.gguf --gpu 3060     # also: 2080ti, 2080, 1660ti

# NVIDIA CMP mining cards: check what the card is really doing, then plan it
python3 scripts/svmi-gpucheck.py                                  # flags a fused-down 170HX
llama-quantize model-f16.gguf model-int8.gguf INT8                # q8_0 attention, Q4_K_M body
python3 scripts/svmi-auto.py model-int8.gguf --gpu cmp170hx-64    # unlocked 8 GiB card = 64 GiB

# stay current with this fork
scripts/tenselerate-update.sh --check                             # exit 10 = newer release
```

Tuned for small-VRAM consumer GPUs (GTX 1660 Ti, RTX 2080 / 2080 Ti, RTX 3060): the
planner accounts for PCIe 3.0's lower streaming ceiling on Turing cards and their single
H2D copy engine. See the [consumer-GPU guide](docs/svmi.md#consumer-gpus-612-gb-1660-ti-rtx-2080--2080-ti-rtx-3060).

Full design, research report, and roadmap (offload-aware speculative decoding,
entropy-coded transport, MoE expert paging): **[docs/svmi.md](docs/svmi.md)**.
Novel techniques designed for this fork (BitSpec self-speculation, pipelined streaming
GEMM, stream-once-serve-many, elastic residency, MAVM multi-agent virtual memory,
CTX-VM paged 131K/256K context, ARBITER compute-follows-memory routing, ...) with
bandwidth math and honesty notes: **[docs/svmi-research.md](docs/svmi-research.md)**.

Lineage: supersedes the `fable5/prefetch-experts` patches from
[thecodacus/llama.cpp](https://github.com/thecodacus/llama.cpp) (pinning + MoE expert
prefetch, +64% prefill on an RTX 3060), rebased on current upstream master and
generalized to dense-model streaming with multi-queue uploads.

## Upstream llama.cpp

The rest of this file is the upstream `ggml-org/llama.cpp` README, kept for reference.

![llama](https://raw.githubusercontent.com/ggml-org/llama.brand/refs/heads/master/cover/llama-cpp/cover-llama-cpp-dark.svg)

<div align="center">

## Recent API changes

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Release](https://img.shields.io/github/v/release/ggml-org/llama.cpp?filter=v*&color=brightgreen)](https://github.com/ggml-org/llama.cpp/releases?q=tag:v0)
[![Nightly](https://img.shields.io/github/v/release/ggml-org/llama.cpp?label=nightly&filter=b*&color=orange)](https://github.com/ggml-org/llama.cpp/releases?q=b)
[![Server](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/server.yml?label=Server)](https://github.com/ggml-org/llama.cpp/actions/workflows/server.yml)
[![Docker](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/docker.yml?label=Docker)](https://github.com/ggml-org/llama.cpp/actions/workflows/docker.yml)
[![Winget](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/winget.yml?label=Winget)](https://github.com/ggml-org/llama.cpp/actions/workflows/winget.yml)

[ggml](https://github.com/ggml-org/ggml) / [ops](https://github.com/ggml-org/llama.cpp/blob/master/docs/ops.md) / [maintainer PRs](https://github.com/ggml-org/llama.cpp/issues?q=is%3Apr%20is%3Aopen%20draft%3AFalse%20(author%3Argerganov%20OR%20author%3AKitaitiMakoto%20OR%20author%3Adanbev%20OR%20author%3Aaldehir%20OR%20author%3Amax-krasnyansky%20OR%20author%3ACISC%20OR%20author%3Aggerganov%20OR%20author%3Aam17an%20OR%20author%3Ajhen0409%20OR%20author%3Abartowski1182%20OR%20author%3Anikwen%20OR%20author%3Ahipudding%20OR%20author%3Aravi9%20OR%20author%3AServeurpersoCom%20OR%20author%3Apwilkin%20OR%20author%3Areeselevine%20OR%20author%3Angxson%20OR%20author%3Ajeffbolznv%20OR%20author%3Amarty1885%20OR%20author%3A0cc4m%20OR%20author%3ATitaniumtown%20OR%20author%3Aangt%20OR%20author%3AIMbackK%20OR%20author%3Aarthw%20OR%20author%3AJohannesGaessler%20OR%20author%3AORippler%20OR%20author%3Aruixiang63%20OR%20author%3Axctan%20OR%20author%3Aallozaur%20OR%20author%3Ayomaytk%20OR%20author%3Aaendk%20OR%20author%3Awine99%20OR%20author%3Agaugarg-nv%20OR%20author%3Ataronaeo%20OR%20author%3Aforforever73%20OR%20author%3Alhez%20OR%20author%3Anetrunnereve%20OR%20author%3Afairydreaming)%20sort%3Aupdated-desc) / [dev stats](https://github.com/ggml-org/llama.cpp-dev) / [lib llama API](https://github.com/ggml-org/llama.cpp/issues/9289) / [llama-server REST API](https://github.com/ggml-org/llama.cpp/issues/9291)

</div>

## Quick start

A few options to get `llama.cpp` installed on your machine:

- Visit https://llama.app and follow the instructions
- Run with Docker - see our [Docker documentation](docs/docker.md)
- Download pre-built binaries from the [releases page](https://github.com/ggml-org/llama.cpp/releases)
- Build from source by cloning this repository - check out [our build guide](docs/build.md)

Once installed:

```sh
# Download and run a model directly from Hugging Face
llama cli -hf ggml-org/Qwen3.5-0.8B-GGUF

# Launch OpenAI-compatible API server
llama serve -hf ggml-org/Qwen3.5-0.8B-GGUF
```

<table align="center">
    <tr>
        <td align="center" width=50%>
            <img width="1310" height="888" alt="VLM session with `llama cli`" src="https://github.com/user-attachments/assets/88726b48-1713-48aa-a525-95a02e78afc4" />
            <i>VLM session with <b>llama cli</b></i>
        </td>
        <td align="center">
            <img width="1392" height="958" alt="Built-in web UI against `llama serve` running Qwen 3.6" src="https://github.com/user-attachments/assets/b402f972-2e32-4def-8771-8d849f08cf2e" />
            <i>Built-in web UI against <b>llama serve</b></i>
        </td>
    </tr>
<table>

## Description

The main goal of `llama.cpp` is to enable LLM (and VLM) inference with minimal setup and state-of-the-art performance on
a wide range of hardware - locally and in the cloud.

- Plain C/C++ implementation without any dependencies
- Apple silicon is a first-class citizen - optimized via ARM NEON, Accelerate and Metal frameworks
- AVX, AVX2, AVX512 and AMX support for x86 architectures
- RVV, ZVFH, ZFH, ZICBOP and ZIHINTPAUSE support for RISC-V architectures
- 1.5-bit, 2-bit, 3-bit, 4-bit, 5-bit, 6-bit, and 8-bit integer quantization for faster inference and reduced memory use
- Custom CUDA kernels for running LLMs on NVIDIA GPUs (support for AMD GPUs via HIP and Moore Threads GPUs via MUSA)
- Vulkan and SYCL backend support
- CPU+GPU hybrid inference to partially accelerate models larger than the total VRAM capacity

The `llama.cpp` project is build on top of the [ggml](https://github.com/ggml-org/ggml) library.

## Supported backends

| Backend | Target devices |
| --- | --- |
| [BLAS](docs/build.md#blas-build) | All |
| [BLIS](docs/backend/BLIS.md) | All |
| [CANN](docs/build.md#cann) | Ascend NPU |
| [CUDA](docs/build.md#cuda) | Nvidia GPU |
| [HIP](docs/build.md#hip) | AMD GPU |
| [Hexagon](docs/backend/snapdragon/README.md) | Snapdragon |
| [IBM zDNN](docs/backend/zDNN.md) | IBM Z & LinuxONE |
| [MUSA](docs/build.md#musa) | Moore Threads GPU |
| [Metal](docs/build.md#metal-build) | Apple Silicon |
| [OpenCL](docs/backend/OPENCL.md) | Adreno GPU |
| [OpenVINO [In Progress]](docs/backend/OPENVINO.md) | Intel CPUs, GPUs, and NPUs |
| [RPC](https://github.com/ggml-org/llama.cpp/tree/master/tools/rpc) | All |
| [SYCL](docs/backend/SYCL.md) | Intel GPU |
| [VirtGPU](docs/backend/VirtGPU.md) | VirtGPU APIR |
| [Vulkan](docs/build.md#vulkan) | GPU |
| [WebGPU](docs/build.md#webgpu) | All |
| [ZenDNN](docs/build.md#zendnn) | AMD CPU |

## Obtaining and quantizing models

The [Hugging Face](https://huggingface.co) platform hosts a [number of LLMs](https://huggingface.co/models?library=gguf&sort=trending) compatible with `llama.cpp`:

- [Trending](https://huggingface.co/models?library=gguf&sort=trending)
- [LLaMA](https://huggingface.co/models?sort=trending&search=llama+gguf)

You can either manually download the GGUF file or directly use any `llama.cpp`-compatible models from [Hugging Face](https://huggingface.co/) or other model hosting sites, by using this CLI argument: `-hf <user>/<model>[:quant]`. For example:

```sh
llama-cli -hf ggml-org/gemma-3-1b-it-GGUF
```

By default, the CLI would download from Hugging Face, you can switch to other options with the environment variable `MODEL_ENDPOINT`. The `MODEL_ENDPOINT` must point to a Hugging Face compatible API endpoint.

After downloading a model, use the CLI tools to run it locally - see below.

`llama.cpp` requires the model to be stored in the [GGUF](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md) file format. Models in other data formats can be converted to GGUF using the `convert_*.py` Python scripts in this repo.

The Hugging Face platform provides a variety of online tools for converting, quantizing and hosting models with `llama.cpp`:

- Use the [GGUF-my-repo space](https://huggingface.co/spaces/ggml-org/gguf-my-repo) to convert to GGUF format and quantize model weights to smaller sizes
- Use the [GGUF-my-LoRA space](https://huggingface.co/spaces/ggml-org/gguf-my-lora) to convert LoRA adapters to GGUF format (more info: https://github.com/ggml-org/llama.cpp/discussions/10123)
- Use the [GGUF-editor space](https://huggingface.co/spaces/CISCai/gguf-editor) to edit GGUF meta data in the browser (more info: https://github.com/ggml-org/llama.cpp/discussions/9268)
- Use the [Inference Endpoints](https://ui.endpoints.huggingface.co/) to directly host `llama.cpp` in the cloud (more info: https://github.com/ggml-org/llama.cpp/discussions/9669)

To learn more about model quantization, [read this documentation](tools/quantize/README.md)

This fork adds `INT8` (aliases `Q4_K_M_INT8`, `Q4KM_INT8`), a mixed quant that keeps
`Q4_K_M`'s mixture but stores every attention tensor as `q8_0` - ~18% of the weights,
so the file is ~15% larger at about 5.6 bpw. It exists for GPUs whose FP16 path is
unusable (the NVIDIA CMP mining cards), where the integer MMQ kernels are the only fast
path, and it lines up with the SVMI split: attention stays resident, the `Q4_K_M` body
is what streams. Also in this tree: `Q2_0` (2.25 bpw) and `Q1_0` (1.125 bpw). See
[docs/svmi.md](docs/svmi.md#mixed-int8q4_k_m-weights-int8).

## [`llama-cli`](tools/cli)

#### A CLI tool for accessing and experimenting with most of `llama.cpp`'s functionality.

- <details open>
    <summary>Run in conversation mode</summary>

    Models with a built-in chat template will automatically activate conversation mode. If this doesn't occur, you can manually enable it by adding `-cnv` and specifying a suitable chat template with `--chat-template NAME`

    ```bash
    llama-cli -m model.gguf

    # > hi, who are you?
    # Hi there! I'm your helpful assistant! I'm an AI-powered chatbot designed to assist and provide information to users like you. I'm here to help answer your questions, provide guidance, and offer support on a wide range of topics. I'm a friendly and knowledgeable AI, and I'm always happy to help with anything you need. What's on your mind, and how can I assist you today?
    #
    # > what is 1+1?
    # Easy peasy! The answer to 1+1 is... 2!
    ```

    </details>

- <details>
    <summary>Run in conversation mode with custom chat template</summary>

    ```bash
    # use the "chatml" template (use -h to see the list of supported templates)
    llama-cli -m model.gguf -cnv --chat-template chatml

    # use a custom template
    llama-cli -m model.gguf -cnv --in-prefix 'User: ' --reverse-prompt 'User:'
    ```

    </details>

- <details>
    <summary>Constrain the output with a custom grammar</summary>

    ```bash
    llama-cli -m model.gguf -n 256 --grammar-file grammars/json.gbnf -p 'Request: schedule a call at 8pm; Command:'

    # {"appointmentTime": "8pm", "appointmentDetails": "schedule a a call"}
    ```

    The [grammars/](grammars/) folder contains a handful of sample grammars. To write your own, check out the [GBNF Guide](grammars/README.md).

    For authoring more complex JSON grammars, check out https://grammar.intrinsiclabs.ai/

    </details>

## [`llama-server`](tools/server)

#### A lightweight, [OpenAI API](https://github.com/openai/openai-openapi) compatible, HTTP server for serving LLMs.

- <details open>
    <summary>Start a local HTTP server with default configuration on port 8080</summary>

    ```bash
    llama-server -m model.gguf --port 8080

    # Basic web UI can be accessed via browser: http://localhost:8080
    # Chat completion endpoint: http://localhost:8080/v1/chat/completions
    ```

    </details>

- <details>
    <summary>Support multiple-users and parallel decoding</summary>

    ```bash
    # up to 4 concurrent requests, each with 4096 max context
    llama-server -m model.gguf -c 16384 -np 4
    ```

    </details>

- <details>
    <summary>Enable speculative decoding</summary>

    ```bash
    # the draft.gguf model should be a small variant of the target model.gguf
    llama-server -m model.gguf -md draft.gguf
    ```

    </details>

- <details>
    <summary>Serve an embedding model</summary>

    ```bash
    # use the /embedding endpoint
    llama-server -m model.gguf --embedding --pooling cls -ub 8192
    ```

    </details>

- <details>
    <summary>Serve a reranking model</summary>

    ```bash
    # use the /reranking endpoint
    llama-server -m model.gguf --reranking
    ```

    </details>

- <details>
    <summary>Constrain all outputs with a grammar</summary>

    ```bash
    # custom grammar
    llama-server -m model.gguf --grammar-file grammar.gbnf

    # JSON
    llama-server -m model.gguf --grammar-file grammars/json.gbnf
    ```

    </details>

## [`llama-perplexity`](tools/perplexity)

#### A tool for measuring the [perplexity](tools/perplexity/README.md) [^1] (and other quality metrics) of a model over a given text.

- <details open>
    <summary>Measure the perplexity over a text file</summary>

    ```bash
    llama-perplexity -m model.gguf -f file.txt

    # [1]15.2701,[2]5.4007,[3]5.3073,[4]6.2965,[5]5.8940,[6]5.6096,[7]5.7942,[8]4.9297, ...
    # Final estimate: PPL = 5.4007 +/- 0.67339
    ```

    </details>

- <details>
    <summary>Measure KL divergence</summary>

    ```bash
    # TODO
    ```

    </details>

[^1]: [https://huggingface.co/docs/transformers/perplexity](https://huggingface.co/docs/transformers/perplexity)

## [`llama-bench`](tools/llama-bench)

#### Benchmark the performance of the inference for various parameters.

- <details open>
    <summary>Run default benchmark</summary>

    ```bash
    llama-bench -m model.gguf

    # Output:
    # | model               |       size |     params | backend    | threads |          test |                  t/s |
    # | ------------------- | ---------: | ---------: | ---------- | ------: | ------------: | -------------------: |
    # | qwen2 1.5B Q4_0     | 885.97 MiB |     1.54 B | Metal,BLAS |      16 |         pp512 |      5765.41 ± 20.55 |
    # | qwen2 1.5B Q4_0     | 885.97 MiB |     1.54 B | Metal,BLAS |      16 |         tg128 |        197.71 ± 0.81 |
    #
    # build: 3e0ba0e60 (4229)
    ```

    </details>

## [`llama-simple`](examples/simple)

#### A minimal example for implementing apps with `llama.cpp`. Useful for developers.

- <details>
    <summary>Basic text completion</summary>

    ```bash
    llama-simple -m model.gguf

    # Hello my name is Kaitlyn and I am a 16 year old girl. I am a junior in high school and I am currently taking a class called "The Art of
    ```

    </details>

#### Tools

- [cli](tools/cli/README.md)
- [completion](tools/completion/README.md)
- [server](tools/server/README.md)
- [GBNF grammars](grammars/README.md)

#### Development

- [How to build](docs/build.md)
- [Running on Docker](docs/docker.md)
- [Build on Android](docs/android.md)
- [Multi-GPU usage](docs/multi-gpu.md)
- [Performance troubleshooting](docs/development/token_generation_performance_tips.md)
- [GGML tips & tricks](https://github.com/ggml-org/llama.cpp/wiki/GGML-Tips-&-Tricks)
- [XCFramework](docs/xcframework.md)
- [Completions](docs/completions.md)
- [Models](docs/models.md)
- [Release process](docs/release.md)

## Contributing

- Contributors can open PRs
- Collaborators will be invited based on contributions
- Maintainers can push to branches in the `llama.cpp` repo and merge PRs into the `master` branch
- Any help with managing issues, PRs and projects is very appreciated!
- Read the [CONTRIBUTING.md](CONTRIBUTING.md) for more information

## Acknowledgements

- [yhirose/cpp-httplib](https://github.com/yhirose/cpp-httplib) - Single-header HTTP server, used by `llama-server` - MIT license
- [nothings/stb](https://github.com/nothings/stb) - Single-header image format decoder, used by multimodal subsystem - Public domain
- [nlohmann/json](https://github.com/nlohmann/json) - Single-header JSON library, used by various tools/examples - MIT License
- [mackron/miniaudio](https://github.com/mackron/miniaudio) - Single-header audio format decoder, used by multimodal subsystem - Public domain
- [sheredom/subprocess.h](https://github.com/sheredom/subprocess.h) - Single-header process launching solution for C and C++ - Public domain
