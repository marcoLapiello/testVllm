"""
GPTQ Optimization Test Script for ROCm
Tests different GPTQ configurations to find optimal performance
"""

from vllm import LLM, SamplingParams
import time
import os


def test_gptq_config(config_name: str, env_vars: dict, vllm_params: dict):
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
    
    try:
        llm = LLM(model="kaitchup/Qwen3-8B-autoround-4bit-gptq", **vllm_params)
        load_time = time.time() - start_load
        print(f"✅ Loaded in {load_time:.2f}s")
        
        # Quick inference test
        sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.8,
            max_tokens=200
        )
        
        prompt = "Explain in one paragraph what quantum computing is."
        
        print("\nRunning inference test...")
        start_gen = time.time()
        outputs = llm.generate([prompt], sampling_params)
        gen_time = time.time() - start_gen
        
        num_tokens = len(outputs[0].outputs[0].token_ids)
        speed = num_tokens / gen_time
        
        print(f"\n📊 Results:")
        print(f"  Generation time: {gen_time:.2f}s")
        print(f"  Tokens generated: {num_tokens}")
        print(f"  Speed: {speed:.2f} tokens/sec")
        
        return speed
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 0
    finally:
        # Clean up
        del llm
        import gc
        import torch
        gc.collect()
        torch.cuda.empty_cache()


def main():
    print("="*80)
    print("🔬 GPTQ OPTIMIZATION TEST FOR ROCM")
    print("="*80)
    print("\nThis script tests different GPTQ configurations to find optimal performance")
    print("for AMD GPUs (ROCm). We'll compare multiple settings.\n")
    
    results = {}
    
    # Test 1: Default configuration (with torch.compile)
    print("\n" + "🔹"*40)
    results['default_compiled'] = test_gptq_config(
        "Default (Compiled)",
        env_vars={},
        vllm_params={
            'max_model_len': 8192,
            'gpu_memory_utilization': 0.9,
            'tensor_parallel_size': 1,
            'quantization': 'gptq',
        }
    )
    
    # Test 2: Larger batch size
    print("\n" + "🔹"*40)
    results['larger_batch'] = test_gptq_config(
        "Larger Batch Size",
        env_vars={},
        vllm_params={
            'max_model_len': 8192,
            'gpu_memory_utilization': 0.9,
            'tensor_parallel_size': 1,
            'quantization': 'gptq',
            'max_num_batched_tokens': 16384,  # Double the default
        }
    )
    
    # Test 3: Disable torch compile
    print("\n" + "🔹"*40)
    results['no_compile'] = test_gptq_config(
        "Disable Torch Compile",
        env_vars={'VLLM_TORCH_COMPILE_LEVEL': '0'},
        vllm_params={
            'max_model_len': 8192,
            'gpu_memory_utilization': 0.9,
            'tensor_parallel_size': 1,
            'quantization': 'gptq',
            'enforce_eager': True,
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
    main()
