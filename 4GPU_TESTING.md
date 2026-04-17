# 4-GPU vLLM Testing on ROCm

**Date:** April 2026  
**Hardware:** 4× AMD Radeon RX 7900 XTX (gfx1100, RDNA3), 24 GB VRAM each (96 GB total)  
**System RAM:** 62 GB usable  
**Stack:** ROCm 7.2 + PyTorch 2.11.0+rocm7.2 + vLLM 0.18.1rc1 (built from source)  
**Models tested:** Qwen3.5-4B (bf16), Qwen3.5-35B-A3B-GPTQ-Int4

---

## Working Launch Commands

### Qwen3.5-4B — TP=4, full precision

```bash
source ~/.venvs/vllm-rocm/bin/activate

HIP_VISIBLE_DEVICES=0,1,2,3 python -m vllm.entrypoints.openai.api_server \
  --model /home/marcolap/Schreibtisch/testVllm/models/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
  --served-model-name Qwen3.5-4B \
  --tensor-parallel-size 4 \
  --dtype bfloat16 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.90 \
  --trust-remote-code \
  --compilation-config '{"cudagraph_capture_sizes": [1, 2, 4, 8, 16, 32]}' \
  --host 0.0.0.0 \
  --port 8000
```

### Qwen3.5-35B-GPTQ — TP=4, 131K context

```bash
source ~/.venvs/vllm-rocm/bin/activate

HIP_VISIBLE_DEVICES=0,1,2,3 python -m vllm.entrypoints.openai.api_server \
  --model /home/marcolap/Schreibtisch/testVllm/models/hub/models--Qwen--Qwen3.5-35B-A3B-GPTQ-Int4/snapshots/33f4e5e615e1f29a7b218906555ea6fe2d09c741 \
  --served-model-name Qwen3.5-35B-GPTQ \
  --tensor-parallel-size 4 \
  --quantization gptq \
  --dtype float16 \
  --max-model-len 131072 \
  --gpu-memory-utilization 0.90 \
  --trust-remote-code \
  --compilation-config '{"cudagraph_capture_sizes": [1, 2, 4]}' \
  --host 0.0.0.0 \
  --port 8000
```

> For safer operation with 62 GB RAM, prefer `--max-model-len 32768` and `--gpu-memory-utilization 0.90`. See RAM constraints below.

---

## Constraint 1 — torch.compile hangs for ~11 minutes on first run

**Symptom:** Server appears hung. The EngineCore prints repeated shm_broadcast warnings every 60 seconds:
```
No available shared memory broadcast block found in 60 seconds.
```
Worker processes are still alive, compiling.

**Root cause:** On the very first launch, vLLM uses PyTorch Inductor to compile the model's computation graph into optimized HIP kernels. With TP=4 each of the 4 workers compiles independently. For Qwen3.5-4B this took **690 seconds (~11 minutes)**. This is a one-time cost.

**The cache:** Compiled artifacts are saved to:
```
~/.cache/vllm/torch_compile_cache/
```
On subsequent launches the compilation is skipped and loaded from cache in seconds. Confirm cache is being used by looking for:
```
Loaded AOT compiled function from ...
```
instead of `Compiling a graph for compile range...`

**Important:** If you kill the server during compilation, the worker processes may survive as zombies completing the compilation in the background (they are deep in a C++ call that ignores SIGINT). Wait for them to finish writing the cache before relaunching, otherwise the cache may be incomplete. Check with:
```bash
ps aux | grep python | grep -v grep
rocm-smi --showpids
```

**This is separate from CUDA graph capture.** torch.compile (Phase 1) and CUDA graph capture (Phase 2) are sequential and independent:
- **Phase 1 — torch.compile:** Compiles computation graph → saved to disk cache → only runs once
- **Phase 2 — CUDA graph capture:** Records replay sequences for each batch size → runs every startup

---

## Constraint 2 — Orphaned worker processes after Ctrl+C

**Symptom:** After pressing Ctrl+C, the API server exits cleanly but VRAM remains occupied and `rocm-smi --showpids` still shows worker PIDs.

**Root cause:** Workers blocked in a C++ call (NCCL collective, torch.compile, or kernel execution) do not respond to SIGINT. The parent (APIServer) exits but the children keep running.

**Solution:** Always check after shutdown:
```bash
rocm-smi --showpids
```
If workers are still listed, kill them explicitly:
```bash
kill -9 <PID1> <PID2> <PID3> <PID4>
```
Or kill all Python processes at once:
```bash
pkill -9 -f "vllm"
```

---

## Constraint 3 — System RAM ceiling due to ROCm HMM

**This is the most important architectural constraint for multi-GPU ROCm inference.**

### What happens

ROCm uses HMM (Heterogeneous Memory Management), which maps GPU VRAM allocations into the worker process's CPU virtual address space. The Linux kernel creates page table entries for every GPU page that is touched. These page tables are counted by the OS against the process's RSS (Resident Set Size).

As a result, every GPU page actively used by a worker process — weights, KV cache blocks, activation buffers — contributes to system RAM consumption.

### The formula

$$\text{Peak system RAM} \approx \text{TP size} \times \text{gpu\_memory\_utilization} \times \text{VRAM per GPU}$$

For this system at full utilization:

$$4 \times 0.90 \times 24\ \text{GB} = 86\ \text{GB}$$

This exceeds the available 62 GB. The system survives because page faults are lazy — VRAM pages are only mapped when first touched, not all at once. But under sustained use with large context windows, RAM fills incrementally until the OOM killer strikes.

### Two distinct RAM growth phases

| Phase | When | Controllable by |
|---|---|---|
| Startup spike | During CUDA graph capture | `cudagraph_capture_sizes` |
| Runtime growth | As KV cache blocks are touched during inference | `--max-model-len`, `--gpu-memory-utilization` |

Reducing `cudagraph_capture_sizes` only helps startup. It has **zero effect** on runtime RAM growth.

### Safe configuration for 62 GB system RAM

| Parameter | Safe value | Risky value |
|---|---|---|
| `--max-model-len` | 32768 | 131072 |
| `--gpu-memory-utilization` | 0.85–0.90 | 0.95 |
| `--tensor-parallel-size` | 2–3 | 4 (theoretical ceiling 86 GB) |

For 131K context with TP=4: use only for single, isolated requests while monitoring `free -h`. Do not run concurrent sessions.

### RAM does not release immediately after shutdown

After the server exits, Linux keeps freed pages in the reclaimable cache. This is normal — `free -h` will show high "used" but also high "available". Check the `available` column, not `used`.

### Why NVIDIA does not have this problem

CUDA manages GPU memory entirely within its own driver layer, invisible to the Linux VM subsystem. GPU allocations do not create HMM mappings and are not counted by the OOM killer. An equivalent NVIDIA system (4× RTX 3090/4090, 24 GB each) with 64 GB system RAM would handle a 75 GB model without issue.

HMM is an AMD architectural decision driven by the MI300X datacenter chip (unified CPU+GPU memory pool), applied uniformly across all ROCm-supported GPUs including consumer RDNA3. There is no current workaround at the software level other than reducing GPU memory utilization and context length.

### Loading models larger than available system RAM

The constraint extends to model loading. Even if a model's weights fit across GPU VRAM, loading them causes page faults that fill system RAM proportional to total weight size:

$$\text{System RAM required for loading} \gtrsim \text{total weight size across all GPUs}$$

A 4-bit quantized 120B model with ~75 GB weights would OOM during weight loading on this system (62 GB RAM), despite fitting in 96 GB VRAM.

---

## GPU Monitoring

`nvtop` crashes on modern amdgpu kernel drivers (known bug). Use `amdgpu_top` instead:

```bash
cargo install amdgpu_top
~/.cargo/bin/amdgpu_top
```

Or for a quick check:
```bash
watch -n1 rocm-smi
```

To monitor system RAM during startup:
```bash
watch -n2 free -h
```

---

## System RAM behaviour during inference

Observed RAM usage stabilizes after all KV cache blocks have been touched once. It does not grow unboundedly — it plateaus at a level proportional to the KV cache size that has been actively used. The KV cache itself lives in VRAM; system RAM grows only due to the HMM page table entries tracking those VRAM-backed pages.

CPU-side growth sources (minor):
- Conversation history in the client script (Python strings)
- vLLM's KV cache block allocator metadata
- Python heap fragmentation from request processing
