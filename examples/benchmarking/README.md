# vLLM Engine Configuration & Benchmarking for RDNA3

**Status:** 🚧 Work in Progress - Phase 1 Complete  
**Hardware:** AMD Radeon RX 7900 XTX (RDNA3 Architecture)  
**Environment:** ROCm 7+ in Docker Container

---

## 📁 Files in This Directory

### 1. `ENGINE_PARAMS_RESEARCH.md`
Comprehensive research document cataloging ALL vLLM engine parameters discovered from official documentation. Includes:
- Complete parameter tables organized by category
- Type information and default values
- RDNA3-specific recommendations
- Key insights for optimization

### 2. `engine_config.py` 
Main configuration and benchmarking script (IN PROGRESS). Currently implements:
- ✅ **Part 1: RDNA3-Specific Environment Variables** (COMPLETE)
  - Flash attention configuration
  - Aotriton/aiter settings for precompiled kernels
  - ROCm-specific optimizations
  
- 🚧 **Part 2: Engine Configuration** (PARTIAL - 2 of 6 categories)
  - ✅ Category 1: Model Loading & Basic Configuration
  - ✅ Category 2: Memory Management
  - ⏳ Category 3: Performance & Parallelism (NEXT)
  - ⏳ Category 4: Quantization
  - ⏳ Category 5: Advanced Features
  - ⏳ Category 6: Compilation Options

- ⏳ **Part 3: Inference & Benchmarking** (NOT STARTED)

---

## 🎯 Project Goals

### Primary Objectives
1. **GGUF Quantization Support** - Test new GGUF format for better performance on RDNA3
2. **RDNA3 Attention Optimization** - Use aotriton precompiled kernels instead of unsupported Triton Flash Attention
3. **Granular Configuration** - Expose ALL engine parameters in one place for manual tuning
4. **Educational Approach** - Learn every parameter step-by-step while optimizing

### Problems Being Solved
- ❌ AWQ/GPTQ quantization underperforms on RDNA3 → ✅ Use GGUF
- ❌ Triton Flash Attention not supported on RDNA3 → ✅ Use aotriton
- ❌ Lack of comprehensive configuration script → ✅ Building one parameter at a time

---

## 🚀 Current Status - Phase 1 Complete

### ✅ What's Done
1. **Research Phase**
   - Documented all vLLM engine parameters from official docs
   - Identified RDNA3-specific optimization strategies
   - Cataloged environment variables for ROCm/aotriton

2. **Environment Configuration**
   - Complete `RDNA3EnvironmentConfig` class with all ROCm flags
   - Automatic environment variable application
   - Detailed comments explaining each setting

3. **Partial Engine Configuration**
   - Model loading parameters (10 parameters)
   - Memory management parameters (8 parameters)
   - Type-annotated with comprehensive descriptions

### 🚧 What's Next (Phase 2)
We will proceed **step-by-step** to avoid overwhelming complexity:

**Next Immediate Step:**
- Add **Category 3: Performance & Parallelism** parameters to `EngineConfig`
  - `tensor_parallel_size`
  - `pipeline_parallel_size`
  - `max_num_seqs`
  - `max_num_batched_tokens`
  - `scheduler_delay_factor`
  - `enable_chunked_prefill`
  - `num_scheduler_steps`
  - `max_parallel_loading_workers`
  - `disable_custom_all_reduce`
  - `enforce_eager`
  - `max_context_len_to_capture`

**After That:**
- Add **Category 4: Quantization** parameters
- Add **Category 5: Advanced Features**
- Add **Category 6: Compilation Options**
- Implement engine initialization
- Add inference testing
- Add benchmarking functionality

---

## 📖 How to Use (Current Version)

### Running the Configuration Script

```bash
# Inside the vLLM Docker container
cd /app/examples/benchmarking
python engine_config.py
```

### What It Currently Does
1. Displays RDNA3-specific environment variable configuration
2. Shows comprehensive engine parameter documentation
3. Prepares configuration object (not yet initialized)

### Modifying Parameters

Edit the `engine_config.py` file directly:

```python
# Example: Configure for GGUF model with RDNA3 optimizations
def main():
    # Environment setup
    env_config = RDNA3EnvironmentConfig()
    env_config.VLLM_USE_TRITON_FLASH_ATTN = 0  # Disable for RDNA3
    env_config.VLLM_ROCM_USE_AITER = True      # Enable aotriton
    env_config.apply()
    
    # Engine config
    engine_config = EngineConfig()
    engine_config.model = "/path/to/model.gguf"
    engine_config.tokenizer = "meta-llama/Llama-3.2-3B-Instruct"
    engine_config.load_format = "gguf"
    engine_config.gpu_memory_utilization = 0.90
    engine_config.max_model_len = 4096
    
    engine_config.display()
```

---

## 🔬 RDNA3-Specific Optimizations

### Critical Settings for Your GPU

**Step 1: PyTorch/ROCm Level (Set FIRST)**
```python
# Enable AOTriton at PyTorch level
TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL = 1
```

**Step 2: vLLM Level**
```python
# In RDNA3EnvironmentConfig:
VLLM_USE_TRITON_FLASH_ATTN = 0        # MUST be 0 - not supported
VLLM_ROCM_USE_AITER = True            # Enable precompiled kernels
VLLM_ROCM_USE_AITER_MHA = True        # Multi-head attention via aiter
VLLM_ROCM_USE_AITER_PAGED_ATTN = True # Try for better paging
```

**Understanding the Relationship:**
- `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` → Enables aotriton at PyTorch level
- `VLLM_ROCM_USE_AITER=True` → Tells vLLM to use aiter ops (which use aotriton)
- Both must be enabled for RDNA3 to work optimally

### GGUF Quantization Setup

```python
# In EngineConfig:
model = "/path/to/model.gguf"
tokenizer = "organization/model-name"  # HuggingFace tokenizer required
load_format = "gguf"                    # Explicitly use GGUF loader
```

---

## 📊 Parameter Categories Overview

### ✅ Completed
1. **Model Loading** (10 params) - How to load and specify models
2. **Memory Management** (8 params) - GPU/CPU memory allocation

### ⏳ To Be Added
3. **Performance & Parallelism** (~11 params) - Multi-GPU, scheduling, throughput
4. **Quantization** (~5 params) - AWQ, GPTQ, GGUF, FP8 settings
5. **Advanced Features** (~8 params) - Speculative decoding, prefix caching, multi-modal
6. **Compilation** (~4 params) - PyTorch compilation, CUDA graphs

**Total Parameters:** ~46 main parameters + environment variables

---

## 🤝 Development Approach

We're building this **incrementally** to maintain clarity:

1. ✅ **Research & Document** - Gather all parameters
2. ✅ **Environment Setup** - RDNA3-specific flags
3. 🚧 **Engine Config** - Add parameters category by category
4. ⏳ **Initialization** - Actually create the engine
5. ⏳ **Inference** - Run test queries
6. ⏳ **Benchmarking** - Measure performance

**Philosophy:** Better to have a working, documented, educational script than a rushed, confusing one.

---

## 🐛 Known Issues

1. Import errors outside Docker container (expected - vLLM only in container)
2. Engine initialization not yet implemented (next phase)
3. No inference/benchmarking yet (future phase)

---

## 📚 References

- [vLLM Documentation](https://docs.vllm.ai/)
- [vLLM GitHub](https://github.com/vllm-project/vllm)
- [ROCm Environment Variables](https://docs.vllm.ai/en/latest/configuration/env_vars)
- [GGUF Support in vLLM](https://docs.vllm.ai/en/latest/features/quantization/gguf)
- [vLLM Quantization Guide](https://docs.vllm.ai/en/latest/features/quantization/)

---

## ❓ Questions or Issues?

This is a learning-focused project. Each parameter is documented with:
- Purpose and description
- Type and default value
- Options available
- RDNA3-specific recommendations

Feel free to experiment with different configurations!
