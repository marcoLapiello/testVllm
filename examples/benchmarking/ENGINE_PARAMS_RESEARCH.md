# vLLM Engine Parameters Research

**Last Updated:** 2024-12-16  
**Target Hardware:** AMD Radeon RX 7900 XTX (RDNA3 Architecture)  
**Environment:** ROCm 7+ in Docker Container

## Document Purpose

This document catalogs ALL available vLLM engine initialization parameters from official documentation. We're building this step-by-step to create a comprehensive configuration script optimized for RDNA3 architecture.

---

## Parameter Categories

### 1. **Model Loading & Basic Configuration**
Parameters that control which model to load and basic setup.

### 2. **Memory Management**
Parameters for GPU/CPU memory allocation and KV cache configuration.

### 3. **Performance & Parallelism**
Tensor parallelism, pipeline parallelism, and scheduling options.

### 4. **Quantization**
Support for AWQ, GPTQ, GGUF, FP8, and other quantization formats.

### 5. **RDNA3-Specific Optimizations**
Flash attention, aotriton, and ROCm-specific environment variables.

### 6. **Advanced Features**
Speculative decoding, prefix caching, compilation options, etc.

---

## Parameters Discovered So Far

### Model Loading & Basic Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | str | Required | Model name (HuggingFace ID) or path to local model/GGUF file |
| `tokenizer` | str | None | Tokenizer name or path. If None, uses same as model |
| `tokenizer_mode` | str | "auto" | Tokenizer mode: "auto", "slow", or "mistral" |
| `tokenizer_revision` | str | None | Specific tokenizer revision/branch/tag/commit |
| `revision` | str | None | Model revision (branch/tag/commit) on HuggingFace |
| `trust_remote_code` | bool | False | Allow custom code execution from model repo |
| `download_dir` | str | None | Directory for downloading/loading model weights |
| `load_format` | str | "auto" | Format: "auto", "pt", "safetensors", "npcache", "dummy", "tensorizer", "bitsandbytes", **"gguf"** |
| `config_format` | str | "auto" | Model config file format |
| `seed` | int | 0 | Random seed for reproducibility |

### Memory Management

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `gpu_memory_utilization` | float | 0.90 | Fraction of GPU memory for model weights (0.0-1.0) |
| `max_model_len` | int | None | Maximum model context length (auto-derived if None) |
| `block_size` | int | 16 | Token block size for contiguous memory in PagedAttention |
| `swap_space` | int | 4 | CPU swap space size in GiB for GPU-CPU memory swapping |
| `cpu_offload_gb` | float | 0 | GiB of model weights to offload to CPU per GPU |
| `enable_prefix_caching` | bool | False | Enable automatic prefix caching (vLLM caching) |
| `disable_sliding_window` | bool | False | Disable sliding window, capping to model's max length |
| `kv_cache_dtype` | str | "auto" | KV cache data type: "auto", "fp8", "fp8_e5m2", "fp8_e4m3" |

### Performance & Parallelism

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tensor_parallel_size` | int | 1 | Number of GPUs for tensor parallelism |
| `pipeline_parallel_size` | int | 1 | Number of pipeline stages for pipeline parallelism |
| `max_parallel_loading_workers` | int | None | Max parallel workers for model loading |
| `disable_custom_all_reduce` | bool | False | Disable custom all-reduce kernel, use NCCL |
| `enforce_eager` | bool | False | Enforce eager execution (disable CUDA graphs) |
| `max_context_len_to_capture` | int | None | Max context length for CUDA graph capture |
| `max_num_seqs` | int | 256 | Maximum number of sequences per iteration |
| `max_num_batched_tokens` | int | None | Maximum tokens to be processed in a single batch |
| `scheduler_delay_factor` | float | 0.0 | Apply delay to scheduler (0.0 = no delay, 1.0 = full delay) |
| `enable_chunked_prefill` | bool | None | Enable chunked prefill for prefill requests |
| `num_scheduler_steps` | int | 1 | Number of scheduler steps per iteration |

### Quantization

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `quantization` | str | None | Quantization method: "awq", "gptq", "squeezellm", "fp8", "compressed-tensors", "bitsandbytes", "gguf", "experts_int8" |
| `rope_scaling` | dict | None | RoPE scaling configuration |
| `rope_theta` | float | None | RoPE theta parameter |
| `gguf_model` | str | None | Path to GGUF model file (alternative to model param) |

### RDNA3-Specific & ROCm Optimizations

**PyTorch/ROCm Environment Variables** (set before importing vLLM):

| Variable | Type | Default | Description | RDNA3 Recommendation |
|----------|------|---------|-------------|---------------------|
| `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL` | int | 0 | Enable AOTriton (Ahead-Of-Time compiled Triton kernels) | **Set to 1** (Required for aotriton) |

**vLLM Environment Variables** (set before engine initialization):

| Variable | Type | Default | Description | RDNA3 Recommendation |
|----------|------|---------|-------------|---------------------|
| `VLLM_USE_TRITON_FLASH_ATTN` | int | 1 | Enable Triton Flash Attention | **Set to 0** (RDNA3 not supported) |
| `VLLM_ROCM_USE_AITER` | bool | False | Enable aiter (Attention Kernel Triton Ops) for ROCm | **Set to True** (Use with aotriton) |
| `VLLM_ROCM_USE_AITER_MHA` | bool | True | Enable aiter for Multi-Head Attention | **Keep True** |
| `VLLM_ROCM_USE_AITER_PAGED_ATTN` | bool | False | Enable paged attention using aiter kernels | **Try True** |
| `VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION` | bool | False | Enable Triton unified attention for V1 in ROCm | **Try True** |
| `VLLM_ROCM_USE_TRITON_ROPE` | bool | False | Enable aiter's Triton Rope kernel | **Try True** |
| `VLLM_ROCM_CUSTOM_PAGED_ATTN` | bool | True | Enable custom paged attention kernel for MI3* cards | Keep True |

### Advanced Features

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `speculative_config` | dict | None | Config for speculative decoding (draft model, method, tokens) |
| `compilation_config` | dict | None | PyTorch compilation configuration |
| `model_loader_extra_config` | dict | None | Extra config for model loader (e.g., tensorizer settings) |
| `mm_processor_cache_gb` | float | 2.0 | Multi-modal processor cache size in GB (0 to disable) |
| `mm_processor_cache_type` | str | "local" | MM cache type: "local" or "shm" (shared memory for TP) |
| `limit_mm_per_prompt` | dict | None | Limit multi-modal inputs per prompt type |
| `reasoning_parser` | str | None | Reasoning parser: "step3" or None |

---

## Next Steps

1. ✅ Research complete list of engine parameters
2. ⏳ Create configuration script with all parameters organized
3. ⏳ Add RDNA3-specific optimizations
4. ⏳ Test GGUF support
5. ⏳ Add inference benchmark functionality

---

## Key Insights for RDNA3 Optimization

### Problem 1: Quantization Performance (GPTQ/AWQ vs GGUF)
- **Solution**: Use `load_format="gguf"` parameter
- GGUF support is now native in vLLM
- Better performance expected on RDNA3 compared to AWQ/GPTQ

### Problem 2: Flash Attention Support
- **Issue**: Standard Triton Flash Attention doesn't support RDNA3
- **Solution**: Use aotriton (precompiled attention kernels)
- **Configuration**:
  1. Set `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` (PyTorch level)
  2. Set `VLLM_USE_TRITON_FLASH_ATTN=0` (disable standard flash attn)
  3. Enable aiter flags: `VLLM_ROCM_USE_AITER=True`
  4. Test various aiter sub-options (MHA, paged attention, unified attention)

### Problem 3: Granular Configuration
- All parameters will be exposed in a single configuration object
- Manual tuning before execution
- Educational approach to learn each parameter

---

## References

- vLLM Official Docs: https://docs.vllm.ai/
- vLLM GitHub: https://github.com/vllm-project/vllm
- ROCm Environment Variables: https://docs.vllm.ai/en/latest/configuration/env_vars
