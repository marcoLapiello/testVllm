# AOTriton Configuration Guide for RDNA3

**Hardware:** AMD Radeon RX 7900 XTX (RDNA3 Architecture - gfx1100)  
**Issue:** Standard Triton Flash Attention doesn't support RDNA3  
**Solution:** Use AOTriton (Ahead-Of-Time compiled Triton kernels)

---

## What is AOTriton?

**AOTriton** (Ahead-Of-Time Triton) is a set of **precompiled attention kernels** specifically designed for AMD GPUs that don't have native Triton support. Instead of compiling Triton kernels at runtime (which doesn't work on RDNA3), AOTriton provides pre-built, optimized kernels.

### Why RDNA3 Needs AOTriton

| Feature | RDNA3 Status | Solution |
|---------|--------------|----------|
| Standard Triton Flash Attention | ❌ Not Supported | Use AOTriton |
| Runtime Triton Compilation | ❌ Doesn't Work | Use Precompiled Kernels |
| CK Flash Attention | ⚠️ Limited | Use AOTriton as Alternative |
| AOTriton Precompiled Kernels | ✅ **Supported** | **Recommended** |

---

## Configuration Layers

AOTriton configuration happens at **two levels**:

### 1. PyTorch/ROCm Level (Foundation)

```bash
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
```

**Purpose:** Enables AOTriton support at the PyTorch level  
**When to set:** Before importing any PyTorch or vLLM code  
**Effect:** Allows PyTorch to use precompiled Triton kernels instead of runtime compilation

### 2. vLLM Level (Application)

```bash
# Disable standard Triton Flash Attention (doesn't work on RDNA3)
export VLLM_USE_TRITON_FLASH_ATTN=0

# Enable vLLM's aiter ops (which use AOTriton)
export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_AITER_MHA=1           # Multi-head attention
export VLLM_ROCM_USE_AITER_PAGED_ATTN=1   # Paged attention (experimental)
```

**Purpose:** Tells vLLM to use the aiter operator set  
**When to set:** Before initializing vLLM engine  
**Effect:** vLLM will use aiter ops, which internally leverage AOTriton kernels

---

## Configuration in Our Script

The `engine_config.py` script handles this in the correct order:

```python
class RDNA3EnvironmentConfig:
    def __init__(self):
        # Step 1: PyTorch/ROCm level (MUST BE FIRST)
        self.TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL = 1
        
        # Step 2: vLLM level
        self.VLLM_USE_TRITON_FLASH_ATTN = 0  # Disable standard
        self.VLLM_ROCM_USE_AITER = True       # Enable aiter (uses AOTriton)
        self.VLLM_ROCM_USE_AITER_MHA = True
        # ... more aiter options
    
    def apply(self):
        # Sets environment variables in the correct order
        os.environ["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = "1"
        os.environ["VLLM_USE_TRITON_FLASH_ATTN"] = "0"
        os.environ["VLLM_ROCM_USE_AITER"] = "true"
        # ...
```

---

## How It Works Together

```
┌─────────────────────────────────────────────────────────┐
│ 1. TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1          │
│    ↓ Enables PyTorch AOTriton support                  │
├─────────────────────────────────────────────────────────┤
│ 2. VLLM_USE_TRITON_FLASH_ATTN=0                       │
│    ↓ Disables standard Triton Flash Attention          │
├─────────────────────────────────────────────────────────┤
│ 3. VLLM_ROCM_USE_AITER=1                              │
│    ↓ Enables vLLM aiter operators                      │
├─────────────────────────────────────────────────────────┤
│ 4. Aiter operators use AOTriton kernels                │
│    ↓ Precompiled kernels execute on RDNA3              │
├─────────────────────────────────────────────────────────┤
│ 5. ✅ Fast attention computation on RDNA3!             │
└─────────────────────────────────────────────────────────┘
```

---

## Available Aiter Operations for RDNA3

Once `VLLM_ROCM_USE_AITER=True`, you can enable specific operations:

| Operation | Variable | Default | RDNA3 Recommendation |
|-----------|----------|---------|---------------------|
| Master Switch | `VLLM_ROCM_USE_AITER` | False | **True** (required) |
| Multi-Head Attention | `VLLM_ROCM_USE_AITER_MHA` | True | **True** (keep enabled) |
| Paged Attention | `VLLM_ROCM_USE_AITER_PAGED_ATTN` | False | **Try True** |
| Unified Attention | `VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION` | False | **Try True** |
| RoPE Kernel | `VLLM_ROCM_USE_TRITON_ROPE` | False | **Try True** |
| MoE Operations | `VLLM_ROCM_USE_AITER_MOE` | True | True (if using MoE) |
| RMS Norm | `VLLM_ROCM_USE_AITER_RMSNORM` | True | True |
| Linear Ops | `VLLM_ROCM_USE_AITER_LINEAR` | True | True |
| FP8 BMM | `VLLM_ROCM_USE_AITER_FP8BMM` | True | True (if using FP8) |

---

## Verification

After setting up, you can verify AOTriton is working:

```python
import torch
import os

# Check environment variables
print("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL:", 
      os.environ.get("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"))
print("VLLM_USE_TRITON_FLASH_ATTN:", 
      os.environ.get("VLLM_USE_TRITON_FLASH_ATTN"))
print("VLLM_ROCM_USE_AITER:", 
      os.environ.get("VLLM_ROCM_USE_AITER"))

# Check GPU
print(f"\nGPU: {torch.cuda.get_device_name(0)}")
print(f"ROCm Version: {torch.version.hip}")
```

Expected output for RDNA3:
```
TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL: 1
VLLM_USE_TRITON_FLASH_ATTN: 0
VLLM_ROCM_USE_AITER: true
GPU: AMD Radeon RX 7900 XTX
ROCm Version: 7.x.x
```

---

## Common Issues

### Issue 1: "aiter not available"
**Cause:** AOTriton not enabled at PyTorch level  
**Solution:** Set `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` **before** importing vLLM

### Issue 2: Poor attention performance
**Cause:** Still using standard Triton flash attention  
**Solution:** Ensure `VLLM_USE_TRITON_FLASH_ATTN=0`

### Issue 3: Compilation errors
**Cause:** Conflicting flags  
**Solution:** Use our script which sets flags in the correct order

---

## Performance Expectations

With proper AOTriton configuration on RDNA3:

| Metric | Without AOTriton | With AOTriton |
|--------|------------------|---------------|
| Attention Speed | ❌ Slow/Broken | ✅ Fast |
| Compilation Time | ❌ Fails | ✅ Instant (precompiled) |
| Memory Efficiency | ❌ Poor | ✅ Optimized |
| Compatibility | ❌ RDNA3 unsupported | ✅ RDNA3 supported |

---

## References

- **PyTorch ROCm:** https://pytorch.org/get-started/locally/
- **vLLM Environment Variables:** https://docs.vllm.ai/en/latest/configuration/env_vars
- **AMD ROCm Documentation:** https://rocm.docs.amd.com/
- **Triton GitHub:** https://github.com/triton-lang/triton
- **ROCm Triton Fork:** https://github.com/ROCm/triton

---

## Summary

For RDNA3 (RX 7900 XTX) to work with vLLM:

1. ✅ Set `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` (PyTorch level)
2. ✅ Set `VLLM_USE_TRITON_FLASH_ATTN=0` (disable broken standard attention)
3. ✅ Set `VLLM_ROCM_USE_AITER=True` (enable aiter which uses AOTriton)
4. ✅ Experiment with other aiter flags for optimal performance

Our `engine_config.py` script handles all of this automatically! 🎉
