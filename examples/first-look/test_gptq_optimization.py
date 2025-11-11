"""
GPTQ Optimization Test Script for ROCm
Tests different GPTQ configurations to find optimal performance
"""

import warnings
import logging

# Suppress specific vLLM deprecation warnings that pollute output
warnings.filterwarnings('ignore', message='.*Processor has been moved.*')
# Set vLLM logger to ERROR level to suppress WARNING messages during generation
logging.getLogger('vllm').setLevel(logging.ERROR)

from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.sampling_params import RequestOutputKind
from vllm.v1.engine.async_llm import AsyncLLM
import asyncio
import time
import os
import sys
import contextlib


async def test_gptq_config(config_name: str, env_vars: dict, vllm_params: dict):
    """Test a specific GPTQ configuration"""
    print("\n" + "="*80)
    print(f"Testing Configuration: {config_name}")
    print("="*80)
    
    # Set environment variables
    for key, value in env_vars.items():
        os.environ[key] = str(value)
        print(f"  {key} = {value}")
    
    print("\nLoading model...")
    start_load = time.time()
    
    engine = None
    try:
        # Create AsyncLLM engine with REAL GPTQ model (not compressed-tensors)
        engine_args = AsyncEngineArgs(
            model="btbtyler09/Qwen3-Coder-30B-A3B-Instruct-gptq-4bit",
            **vllm_params
        )
        engine = AsyncLLM.from_engine_args(engine_args)
        load_time = time.time() - start_load
        print(f"✅ Loaded in {load_time:.2f}s")
        
        # Quick inference test with streaming
        sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.8,
            max_tokens=200,
            output_kind=RequestOutputKind.DELTA,  # Get only new tokens each iteration
        )
        
        prompt = "Write a Python function to calculate the Fibonacci sequence using dynamic programming."
        request_id = f"{config_name}-test"
        
        print("\nRunning inference test...")
        print(f"\n💬 Prompt: {prompt}")
        print(f"🤖 Response: ", end='', flush=True)
        print()  # Add newline before any warnings/response
        
        start_gen = time.time()
        
        # Stream tokens using AsyncLLM
        num_tokens = 0

        # Temporarily suppress vLLM warnings that may be emitted to stderr
        # (these can interleave with the streamed response). We raise the
        # vLLM logger level and redirect stderr to /dev/null only for the
        # duration of the async generation loop, then restore settings.
        vllm_logger = logging.getLogger("vllm")
        async_logger = logging.getLogger("vllm.v1.engine.async_llm")
        prev_vllm_level = vllm_logger.level
        prev_async_level = async_logger.level
        vllm_logger.setLevel(logging.ERROR)
        async_logger.setLevel(logging.ERROR)

        devnull = open(os.devnull, "w")
        try:
            with contextlib.redirect_stderr(devnull):
                async for output in engine.generate(
                    request_id=request_id,
                    prompt=prompt,
                    sampling_params=sampling_params,
                ):
                    # In DELTA mode, completion.text contains only new tokens
                    for completion in output.outputs:
                        new_text = completion.text
                        if new_text:
                            print(new_text, end="", flush=True)

                        # In DELTA mode, token_ids contains only new tokens per iteration
                        if hasattr(completion, "token_ids") and completion.token_ids:
                            num_tokens += len(completion.token_ids)

                    if output.finished:
                        break
        finally:
            devnull.close()
            # restore logger levels
            vllm_logger.setLevel(prev_vllm_level)
            async_logger.setLevel(prev_async_level)
        
        gen_time = time.time() - start_gen
        speed = num_tokens / gen_time if gen_time > 0 else 0
        
        print("\n\n📊 Results:")
        print(f"  Generation time: {gen_time:.2f}s")
        print(f"  Tokens generated: {num_tokens}")
        print(f"  Speed: {speed:.2f} tokens/sec")
        
        return speed
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 0
    finally:
        # Clean up
        if engine:
            engine.shutdown()
        import gc
        import torch
        gc.collect()
        torch.cuda.empty_cache()


async def main():
    print("="*80)
    print("🔬 GPTQ OPTIMIZATION TEST FOR ROCM")
    print("="*80)
    print("\nThis script tests different GPTQ configurations to find optimal performance")
    print("for AMD GPUs (ROCm). We'll compare multiple settings.\n")
    
    results = {}
    
    # Test 1: GPTQ Marlin kernel (recommended, faster and more stable)
    print("\n" + "🔹"*40)
    results['gptq_marlin'] = await test_gptq_config(
        "GPTQ Marlin Kernel",
        env_vars={},
        vllm_params={
            'max_model_len': 8192,
            'gpu_memory_utilization': 0.95,
            'tensor_parallel_size': 1,
            'quantization': 'gptq_marlin',  # Use optimized Marlin kernel
        }
    )
    
    # Test 2: Default GPTQ kernel (for comparison - has bugs warning)
    print("\n" + "🔹"*40)
    results['gptq_default'] = await test_gptq_config(
        "GPTQ Default Kernel",
        env_vars={},
        vllm_params={
            'max_model_len': 8192,
            'gpu_memory_utilization': 0.95,
            'tensor_parallel_size': 1,
            # Don't specify quantization - uses default gptq_gemm (buggy)
        }
    )
    
    # Test 3: GPTQ Marlin with larger batch
    print("\n" + "🔹"*40)
    results['marlin_large_batch'] = await test_gptq_config(
        "Marlin + Larger Batch",
        env_vars={},
        vllm_params={
            'max_model_len': 8192,
            'gpu_memory_utilization': 0.9,
            'tensor_parallel_size': 1,
            'quantization': 'gptq_marlin',
            'max_num_batched_tokens': 16384,  # Double the default
        }
    )
    
    # Summary
    print("\n" + "="*80)
    print("📊 PERFORMANCE SUMMARY")
    print("="*80)
    print("\nSpeed comparison (tokens/sec):")
    
    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    for config, speed in sorted_results:
        if speed > 0:
            print(f"  {config:25s}: {speed:6.2f} tokens/sec")
    
    best_config = sorted_results[0][0]
    best_speed = sorted_results[0][1]
    
    print(f"\n🏆 Best configuration: {best_config}")
    print(f"   Speed: {best_speed:.2f} tokens/sec")
    
    # Recommendation
    print("\n" + "="*80)
    print("💡 RECOMMENDATIONS")
    print("="*80)
    
    if best_speed < 25:
        print("\n⚠️  GPTQ quantization is slower than expected on your ROCm setup.")
        print("\nPossible reasons:")
        print("  1. vLLM's GPTQ kernels are not optimized for your AMD GPU architecture")
        print("  2. ROCm implementation has overhead compared to CUDA")
        print("  3. Your GPU may not have optimized INT4 compute paths")
        print("\nAlternatives to consider:")
        print("  ✅ Use full precision model (BF16/FP16)")
        print("  ✅ Try FP8 quantization if you have MI300+ GPU")
        print("  ✅ Check if PyTorch/ROCm updates improve INT4 performance")
    else:
        print(f"\n✅ Performance is good! Use the '{best_config}' configuration.")


if __name__ == "__main__":
    asyncio.run(main())
