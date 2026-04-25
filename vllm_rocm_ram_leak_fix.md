# vLLM ROCm RAM Leak Investigation & Fix

**Environment:** vLLM 0.20 · ROCm 7.2.2 · 4× RX7900XTX · Tensor Parallelism 4  
**Date:** April 2026

---

## Problem

Each vLLM worker continuously consumed and never freed system RAM:

- ~35 GiB total after server initialization
- Linear growth during inference, even between sessions
- Required periodic server restarts to recover memory

### Breakdown per worker (~9.6 GiB Private Dirty each)

| Region | Size | Type |
|---|---|---|
| Growing anonymous block | ~4.7 GiB | Python/CPU-side buffer (primary leak) |
| 208 × 16 MB pinned huge pages | ~3.3 GiB | ROCm TTM DMA-pinned pages |
| Other pinned regions | ~1.6 GiB | Misc |

> **Note:** Regions with `RSS=0, PD=0` and `pf io dc de dd ms` flags are GPU-managed I/O address space and do **not** consume actual RAM — not the culprit.

---

## Root Cause

### Primary: Unbounded prefix cache CPU growth (vLLM bug #28726)

vLLM's V1 engine prefix caching stores KV block hash maps in CPU RAM with **no eviction limit**. Memory grows indefinitely as more unique prefixes are seen. This is amplified by TP=4 (one cache per worker).

- Tracked in: [vLLM issue #28726](https://github.com/vllm-project/vllm/issues/28726)
- Fixed in: PR #34183 (merged late 2025)
- The fix introduced LRU eviction and configurable caps via environment variables

### Secondary: ROCm TTM pinned pages

ROCm pre-pins system RAM for GPU DMA at startup and caches these pages in a pool. These are largely static after initialization and not freed between requests by design. Growth here is typically an indirect side effect of the prefix cache issue.

---

## Fix

Two environment variables cap the relevant caches with **LRU eviction** (no errors on cap — oldest entries are silently dropped and recomputed on demand):

### `VLLM_CPU_KVCACHE_SPACE`
- Caps the CPU-side KV cache (attention key/value blocks)
- On cap: LRU eviction — oldest/least-used blocks are dropped
- Evicted blocks are recomputed on next access (cache miss, not a crash)

### `VLLM_MM_INPUT_CACHE_GIB`
- Caps the multimodal input processor cache (preprocessed images/audio)
- On cap: LRU eviction — oldest preprocessed inputs are dropped
- Only relevant if the model receives image or audio inputs

---

## Server Start Command (Fixed)

```bash
source ~/.venvs/vllm-rocm/bin/activate

VLLM_CPU_KVCACHE_SPACE=16 VLLM_MM_INPUT_CACHE_GIB=2 HIP_VISIBLE_DEVICES=0,1,2,3 python -m vllm.entrypoints.openai.api_server \
  --model /path/to/model \
  --served-model-name Qwen3.5-27B-GPTQ-4Bit \
  --tensor-parallel-size 4 \
  --quantization gptq \
  --dtype float16 \
  --max-parallel-loading-workers 1 \
  --enable-prefix-caching \
  --mamba-cache-mode align \
  --mamba-block-size 8 \
  --enable-chunked-prefill \
  --max-num-seqs 16 \
  --max-model-len 131072 \
  --gpu-memory-utilization 0.95 \
  --trust-remote-code \
  --compilation-config '{"cudagraph_capture_sizes": [1, 2]}' \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --host 0.0.0.0 \
  --port 8000
```

> `VLLM_CPU_KVCACHE_SPACE=16` gives ~4 GiB per worker across 4 workers.  
> Drop `VLLM_MM_INPUT_CACHE_GIB` if your workload is text-only.

---

## What Did NOT Help

| Suggestion | Why it didn't apply |
|---|---|
| `--swap-space 0` | Flag is unused/no-op in vLLM V1 engine |
| `--no-enable-prefix-caching` | Not viable for agentic workloads (catastrophic recomputation cost) |
| `PYTORCH_HIP_ALLOC_CONF` | Controls GPU VRAM pools, not system RAM |
| `--gpu-memory-utilization` reduction | Irrelevant — problem was CPU RAM, not VRAM |

---

## Result

RAM usage stabilized after applying the two environment variables. No server restarts required.
