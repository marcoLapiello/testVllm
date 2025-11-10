"""
Unified Model Testing Script for vLLM
Configure the model and test parameters at the top of the main() function
"""

from vllm import LLM, SamplingParams
import time
from typing import List, Dict, Optional


def print_model_info(model_config: Dict):
    """Print model information banner"""
    print("="*80)
    print(f"🚀 Testing: {model_config['name']}")
    print("="*80)
    print("\nModel Info:")
    for key, value in model_config['info'].items():
        print(f"  - {key}: {value}")
    print("\nLoading model...\n")


def run_test_cases(llm: LLM, test_cases: List[Dict], sampling_params: SamplingParams):
    """Run individual test cases and display results"""
    print("="*80)
    print("🧪 RUNNING TEST CASES")
    print("="*80)
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n{'='*80}")
        print(f"Test {i}/{len(test_cases)}: {test['description']}")
        print(f"{'='*80}")
        print(f"\n💬 Prompt:\n{test['prompt']}\n")
        
        start_gen = time.time()
        outputs = llm.generate([test['prompt']], sampling_params)
        gen_time = time.time() - start_gen
        
        generated = outputs[0].outputs[0].text
        num_tokens = len(outputs[0].outputs[0].token_ids)
        
        print(f"🤖 Response:\n{generated}\n")
        print(f"⏱️  Generation time: {gen_time:.2f}s")
        print(f"📊 Tokens generated: {num_tokens}")
        print(f"🚀 Speed: {num_tokens/gen_time:.2f} tokens/sec")
    
    print("\n" + "="*80)
    print("✅ ALL TESTS COMPLETED")
    print("="*80)


def run_batch_test(llm: LLM, batch_prompts: List[str], sampling_params: SamplingParams):
    """Run batch processing test"""
    print("\n" + "="*80)
    print("🔄 BONUS: Batch Processing Test")
    print("="*80)
    
    print(f"\nProcessing {len(batch_prompts)} prompts in batch...")
    start_batch = time.time()
    
    batch_outputs = llm.generate(batch_prompts, sampling_params)
    batch_time = time.time() - start_batch
    
    print(f"✅ Completed in {batch_time:.2f}s")
    print(f"⚡ Average: {batch_time/len(batch_prompts):.2f}s per prompt\n")
    
    for i, output in enumerate(batch_outputs, 1):
        response_text = output.outputs[0].text
        preview = response_text[:100] + "..." if len(response_text) > 100 else response_text
        print(f"{i}. Q: {output.prompt}")
        print(f"   A: {preview}\n")


def main():
    # ============================================================================
    # CONFIGURATION SECTION - Edit these parameters for different models
    # See MODEL_CONFIGS.md for ready-to-use configurations
    # ============================================================================
    
    # Testing: Qwen3-8B-AWQ (Quantized 4-bit, faster & less VRAM)
    model_config = {
        'name': "Qwen/Qwen3-8B-AWQ",
        'info': {
            'Size': '8.2B parameters (AWQ 4-bit quantized)',
            'Precision': 'AWQ 4-bit',
            'Expected VRAM': '~4-5 GB (vs 16GB for full precision)',
            'Context length': '32,768 tokens (32K native, 131K with YaRN)',
            'License': 'Apache 2.0',
            'Features': 'Thinking mode, reasoning capabilities, memory efficient',
        }
    }
    
    # vLLM model loading parameters
    vllm_params = {
        'max_model_len': 8192,           # Context window to use (8K for testing)
        'gpu_memory_utilization': 0.9,   # GPU memory usage (0.0 to 1.0)
        'tensor_parallel_size': 1,       # Number of GPUs for tensor parallelism
        'quantization': 'awq',           # AWQ quantization method
        # Note: To enable thinking mode parsing, add: 'reasoning_parser': 'qwen3'
    }
    
    # Sampling parameters for generation
    # Qwen3 recommendation for non-thinking mode: temp=0.7, top_p=0.8
    # For thinking mode: temp=0.6, top_p=0.95
    sampling_config = {
        'temperature': 0.7,              # Randomness (non-thinking mode)
        'top_p': 0.8,                    # Nucleus sampling (non-thinking mode)
        'max_tokens': 500,               # Maximum tokens to generate
        'presence_penalty': 1.5,         # Recommended for quantized models
    }
    
    # Test cases to run
    test_cases = [
        {
            "prompt": "/no_think Explain quantum computing in simple terms that a 10-year-old could understand.",
            "description": "Science explanation"
        },
        {
            "prompt": "/no_think Write a Python function to calculate the Fibonacci sequence using dynamic programming.",
            "description": "Code generation"
        },
        {
            "prompt": "/no_think What are the key differences between machine learning and deep learning?",
            "description": "Technical comparison"
        },
        {
            "prompt": "/no_think Tell me a creative short story about a robot learning to paint.",
            "description": "Creative writing"
        }
    ]
    
    # Batch test prompts (set to None to skip batch test)
    #batch_prompts = [
    #    "What is artificial intelligence?",
    #    "Explain neural networks briefly.",
    #    "What is the purpose of machine learning?"
    #]
    
    # ============================================================================
    # END CONFIGURATION SECTION
    # ============================================================================
    
    # Print model info
    print_model_info(model_config)
    
    # Load the model
    start_load = time.time()
    llm = LLM(model=model_config['name'], **vllm_params)
    load_time = time.time() - start_load
    print(f"✅ Model loaded in {load_time:.2f} seconds\n")
    
    # Create sampling parameters
    sampling_params = SamplingParams(**sampling_config)
    
    # Run test cases
    if test_cases:
        run_test_cases(llm, test_cases, sampling_params)
    
    # Run batch test
    #if batch_prompts:
    #    batch_sampling = SamplingParams(
    #        temperature=sampling_config['temperature'],
    #        max_tokens=100
    #    )
    #    run_batch_test(llm, batch_prompts, batch_sampling)
    
    print("\n" + "="*80)
    print("🎉 TESTING COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
