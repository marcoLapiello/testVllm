# Multi-GPU vLLM Setup on ROCm (2x RX 7900 XTX)

**Date:** April 2026  
**Hardware:** 2× AMD Radeon RX 7900 XTX (gfx1100, RDNA3), 24 GB VRAM each  
**System RAM:** 64 GB  
**Stack:** ROCm 7.2 + PyTorch 2.11.0+rocm7.2 + vLLM (built from source)  
**Models tested:** Qwen3.5-35B-A3B-GPTQ-Int4

---

## Verification Commands

Before launching, confirm both GPUs are visible:

```bash
rocm-smi                          # should show Device 0 and Device 1
rocminfo | grep -E "Name|gfx"    # should show gfx1100 twice
```

Confirm PyTorch sees both GPUs:

```bash
source ~/.venvs/vllm-rocm/bin/activate
python3 -c "
import torch
print(torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i), torch.cuda.get_device_properties(i).total_memory / 1024**3)
"
```

---

## Model Details

Both models share the same base architecture:

| Property | Value |
|---|---|
| Architecture | `Qwen3_5MoeForConditionalGeneration` (MoE) |
| Layers | 40 total (10 full attention, 30 linear/RWKV) |
| Experts | 256 total, 8 active per token |
| Active params per token | ~3B (A3B) |
| Max context | 262,144 tokens |
| Weights on disk | ~23 GB |

| | GPTQ | AWQ |
|---|---|---|
| Source | `Qwen/Qwen3.5-35B-A3B-GPTQ-Int4` | `cyankiwi/Qwen3.5-35B-A3B-AWQ-4bit` |
| Format | `gptq` | `compressed-tensors` |
| Bits | int4 | int4 |
| Group size | 128 | 32 |
| Attention layers | kept in bf16 | visual blocks excluded only |

---

## Working Launch Command (GPTQ)

```bash
source ~/.venvs/vllm-rocm/bin/activate

HIP_VISIBLE_DEVICES=0,1 python -m vllm.entrypoints.openai.api_server \
  --model /home/marcolap/Schreibtisch/testVllm/models/hub/models--Qwen--Qwen3.5-35B-A3B-GPTQ-Int4/snapshots/33f4e5e615e1f29a7b218906555ea6fe2d09c741 \
  --served-model-name Qwen3.5-35B-GPTQ \
  --tensor-parallel-size 2 \
  --quantization gptq \
  --dtype float16 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.90 \
  --trust-remote-code \
  --compilation-config '{"cudagraph_capture_sizes": [1, 2, 4, 8, 16, 32]}' \
  --host 0.0.0.0 \
  --port 8000
```

**Run this outside VS Code** (in a standalone terminal or tmux) to prevent the process dying if VS Code crashes.

---

## Constraints and Solutions

### 1. System RAM OOM during CUDA graph capture

**Symptom:** Worker process killed mid-startup at step ~23/51 of CUDA graph capture. dmesg shows:
```
oom-kill: ... task=VLLM::Worker_TP, pid=67637
Out of memory: Killed process 67637 (VLLM::Worker_TP)
```

**Root cause:** On ROCm with HMM, GPU VRAM is mapped into each worker process's virtual address space. With TP=2 and 2×24 GB GPUs, each of the two worker processes occupies ~28 GB of system RAM in page tables alone. During CUDA graph capture (default: 51 graphs × sizes up to 512), each capture temporarily allocates full activation tensors in system RAM — overflowing 64 GB.

**Solution:** Reduce the number of captured graph sizes:
```
--compilation-config '{"cudagraph_capture_sizes": [1, 2, 4, 8, 16, 32]}'
```

**Performance impact:**
- Batch sizes in the list run at full graph speed (fast)
- Batch sizes not in the list fall back to eager (slightly slower)
- Single-user inference (batch=1) is completely unaffected
- High concurrency (>32 simultaneous requests) sees some degradation

**Scaling note:** System RAM requirements grow linearly with GPU count due to HMM page tables. This is a ROCm-specific behaviour (CUDA/NVIDIA does not do this).

---

### 2. Default CUDA graph capture count (51 graphs)

vLLM generates the capture list automatically using this formula:
```
[1, 2, 4] + range(8, 256, step=8) + range(256, max+1, step=16)
```
where `max = min(max_num_seqs * 2, 512)`. With defaults this yields 51 sizes.

To extend coverage safely without OOM risk, add sizes up to 128:
```
--compilation-config '{"cudagraph_capture_sizes": [1, 2, 4, 8, 16, 32, 64, 128]}'
```
Monitor peak RAM during startup with `watch -n1 free -h` before committing to larger sizes.

---

### 3. nvtop crashes on modern amdgpu driver

**Symptom:**
```
nvtop: Assertion `!cache_entry_check && "We should not be processing a client id twice per update"' failed.
```

**Cause:** The apt-packaged nvtop is too old for the fdinfo format used by recent amdgpu kernel drivers.

**Solution:** Use `amdgpu_top` instead:
```bash
cargo install amdgpu_top
~/.cargo/bin/amdgpu_top
```

Or for a quick live view without installing anything:
```bash
watch -n1 rocm-smi
```

---

### 4. KV cache memory

- Only 10 of the 40 layers use full attention (MoE hybrid architecture), so KV cache pressure is low
- With `--gpu-memory-utilization 0.90` and ~11.5 GB weights per GPU, each GPU has ~10 GB for KV cache
- At `--max-model-len 32768`: ~218,000 KV cache tokens available (~23× max context)
- KV cache lives entirely in VRAM and does not grow into system RAM during inference

---

## `--enforce-eager` vs CUDA Graphs

| Mode | Startup | Decode speed |
|---|---|---|
| Default (51 graphs) | ~2–3 min | Fast for all batch sizes |
| Reduced capture list | ~15–30 sec | Fast for listed sizes, eager for others |
| `--enforce-eager` | Instant | Always slow (no graphs) |

`--enforce-eager` was tested and confirmed to severely hurt decode throughput. It is only useful for debugging, not production.

---

## Interactive Chat

A streaming chat client is available at `examples/chat.py`:

```bash
source ~/.venvs/vllm-rocm/bin/activate
python3 ~/Schreibtisch/testVllm/examples/chat.py
```

Requires the vLLM server to be running on `localhost:8000`. The venv activation does not interfere with the running server process.
