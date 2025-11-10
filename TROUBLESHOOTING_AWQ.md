# AWQ Performance Issues on ROCm - Troubleshooting Guide

## Problem Summary

AWQ 4-bit quantized model (Qwen3-8B-AWQ) is **significantly slower** than full precision variant:
- **Full precision (Qwen3-8B)**: ~28+ tokens/sec
- **AWQ 4-bit (Qwen3-8B-AWQ)**: ~19.34 tokens/sec

**Expected behavior**: Quantized models should be faster due to reduced memory bandwidth and computation.

## Root Cause

**vLLM's AWQ implementation on ROCm (AMD GPUs) is not well-optimized compared to CUDA (NVIDIA)**

Key issues from logs:
1. Forced Triton AWQ path: `VLLM_USE_TRITON_AWQ` being auto-enabled
2. ROCm tunable warnings: `Failed validator: GCN_ARCH_NAME`
3. Slow model loading: 215 seconds (network issue but indicates other problems)

## Why This Happens

### 1. **Hardware Architecture Mismatch**
- AWQ kernels are primarily optimized for NVIDIA GPUs (CUDA)
- AMD GPUs use different instruction sets and memory hierarchies
- INT4 tensor cores may not be fully utilized on ROCm

### 2. **Triton Kernel Overhead**
- vLLM uses Triton for AWQ on ROCm
- Triton's ROCm backend has more overhead than CUDA backend
- Kernel compilation and execution may be suboptimal

### 3. **Memory Bandwidth Paradox**
- While AWQ uses less memory (4GB vs 16GB)
- The dequantization overhead on AMD GPUs can exceed bandwidth savings
- Full precision BF16/FP16 ops may be better optimized

## Solutions to Try

### Option 1: Test Different AWQ Configurations (Recommended First)

Run the optimization test script:

```bash
sudo docker exec -it vllm-rocm bash -c "cd /app/examples && python test_awq_optimization.py"
```

This will test:
- Default Triton AWQ (current)
- Disabled Triton AWQ (fallback path)
- Larger batch sizes
- Disabled torch compile

### Option 2: Disable Triton AWQ Manually

Edit `test_model.py` or set environment variable:

```python
import os
os.environ['VLLM_USE_TRITON_AWQ'] = '0'  # Add before LLM initialization

# Or run with:
# VLLM_USE_TRITON_AWQ=0 python test_model.py
```

### Option 3: Try GPTQ Instead of AWQ

GPTQ may have better ROCm support:

```python
model_config = {
    'name': "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4",  # Example
    # ...
}

vllm_params = {
    'quantization': 'gptq',  # Instead of 'awq'
    # ...
}
```

### Option 4: Stick with Full Precision (Recommended)

Since full precision gives you **28+ tokens/sec**, this is actually your best option:

```python
model_config = {
    'name': "Qwen/Qwen3-8B",  # Full precision
    'info': {
        'Size': '8.2 Billion parameters',
        'Precision': 'BF16 (full precision)',
        'Expected VRAM': '~16 GB',
    }
}

vllm_params = {
    'max_model_len': 8192,
    'gpu_memory_utilization': 0.9,
    'tensor_parallel_size': 1,
    # No quantization parameter
}
```

### Option 5: Use Smaller Full-Precision Models

Instead of quantizing an 8B model, use a smaller full-precision model:
- **Qwen2.5-7B** (full precision) - likely 25-30 tokens/sec
- **Mistral-7B** (full precision) - already in your workspace
- **Phi-3-mini-4k** (3.8B) - very fast and efficient

## Performance Expectations

### On AMD GPUs (ROCm):
| Model Type | Expected Speed | Your Results |
|------------|---------------|--------------|
| Full Precision 8B | 25-30 tok/s | ✅ 28+ tok/s |
| AWQ 4-bit 8B | 35-45 tok/s | ❌ 19.34 tok/s |
| Full Precision 7B | 30-35 tok/s | ? |

### On NVIDIA GPUs (CUDA) - For Comparison:
| Model Type | Expected Speed |
|------------|---------------|
| Full Precision 8B | 30-35 tok/s |
| AWQ 4-bit 8B | 50-70 tok/s |

## Testing Full Precision Qwen3-8B

To properly compare, test the full precision model with same settings:

```bash
# Edit test_model.py to use Qwen/Qwen3-8B (no AWQ)
# Then run:
sudo docker exec -it vllm-rocm bash -c "cd /app/examples && python test_model.py"
```

## Verification Commands

Check your GPU and ROCm setup:

```bash
# Inside container
rocm-smi  # Check GPU utilization during inference
rocminfo | grep "Name:"  # Check GPU model
hipcc --version  # Check ROCm version
python -c "import torch; print(f'ROCm: {torch.version.hip}')"
```

## When to Use AWQ on ROCm

AWQ quantization on ROCm makes sense when:
- ✅ You're **memory-constrained** (need to fit larger models)
- ✅ You can **sacrifice speed for capacity** (e.g., running 70B instead of 8B)
- ✅ You're doing **batch inference** (better utilization)
- ❌ NOT when full precision fits comfortably in VRAM

## Recommended Configuration for Your Setup

Based on your results:

```python
# Use full precision - it's faster!
model_config = {
    'name': "Qwen/Qwen3-8B",  # Full precision, not AWQ
}

vllm_params = {
    'max_model_len': 8192,
    'gpu_memory_utilization': 0.9,
    'tensor_parallel_size': 1,
    'dtype': 'bfloat16',  # Or 'float16'
}
```

## Further Investigation

If you want to dive deeper:

1. **Profile the model**:
```python
vllm_params['enforce_eager'] = True  # Disable graph capture
vllm_params['disable_log_stats'] = False  # Enable stats
```

2. **Check kernel performance**:
```bash
VLLM_LOGGING_LEVEL=DEBUG python test_model.py 2>&1 | grep -i "kernel"
```

3. **Compare with ONNX Runtime or llama.cpp**:
   - These may have better AMD GPU support for quantized models

## Conclusion

**For your AMD GPU setup with ROCm:**
- ✅ **Use full precision models** - you get better performance (28+ tok/s)
- ⚠️ **AWQ quantization is counterproductive** - slower AND lower quality
- 💡 **Save memory by using smaller models**, not quantization

The memory savings from AWQ (4GB vs 16GB) don't justify the 30% performance loss on your hardware.
