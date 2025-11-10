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
    
    # Default: Facebook OPT-125M (fast testing)
    # For better quality, use: mistralai/Mistral-7B-Instruct-v0.3
    model_config = {
        'name': "mistralai/Mistral-7B-Instruct-v0.3",
        'info': {
            'Size': '7 Billion parameters',
            'Precision': 'FP16 (full precision)',
            'Expected VRAM': '~14-15 GB',
            'Context length': '32,768 tokens (32K)',
            'License': 'Apache 2.0 (fully open)',
        }
    }
    
    # vLLM model loading parameters
    vllm_params = {
        'max_model_len': 4096,           # Context window to use
        'gpu_memory_utilization': 0.9,   # GPU memory usage (0.0 to 1.0)
        'tensor_parallel_size': 1,       # Number of GPUs for tensor parallelism
    }
    
    # Sampling parameters for generation
    sampling_config = {
        'temperature': 0.8,              # Randomness (0.0 = deterministic, 1.0+ = creative)
        'top_p': 0.95,                   # Nucleus sampling
        'max_tokens': 500,               # Maximum tokens to generate
    }
    
    # Test cases to run
    test_cases = [
        {
            "prompt": "Explain quantum computing in simple terms that a 10-year-old could understand.",
            "description": "Science explanation"
        },
        {
            "prompt": "Write a Python function to calculate the Fibonacci sequence using dynamic programming.",
            "description": "Code generation"
        },
        {
            "prompt": "What are the key differences between machine learning and deep learning?",
            "description": "Technical comparison"
        },
        {
            "prompt": "Tell me a creative short story about a robot learning to paint.",
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
