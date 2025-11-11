"""
Test Mistral 7B Instruct v0.3 - Full Precision (FP16)
This is an excellent open model that doesn't require authentication
"""

from vllm import LLM, SamplingParams
import time

def main():
    model_name = "mistralai/Mistral-7B-Instruct-v0.3"
    
    print("="*80)
    print(f"🌟 Testing: {model_name}")
    print("="*80)
    print("\nModel Info:")
    print("  - Size: 7 Billion parameters")
    print("  - Precision: FP16 (full precision)")
    print("  - Expected VRAM: ~14-15 GB")
    print("  - Context length: 32,768 tokens (32K)")
    print("  - License: Apache 2.0 (fully open)")
    print("\nLoading model...\n")
    
    start_load = time.time()
    
    # Load the model
    llm = LLM(
        model=model_name,
        max_model_len=4096,  # Use 4K context
        gpu_memory_utilization=0.9,
        tensor_parallel_size=1,
    )
    
    load_time = time.time() - start_load
    print(f"✅ Model loaded in {load_time:.2f} seconds\n")
    
    # Sampling parameters
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=300,
    )
    
    # Test prompts
    test_cases = [
        {
            "prompt": "Explain quantum computing in simple terms.",
            "description": "Science explanation"
        },
        {
            "prompt": "Write a Python function to find prime numbers up to n.",
            "description": "Code generation"
        },
        {
            "prompt": "What are the main differences between supervised and unsupervised learning?",
            "description": "Technical comparison"
        }
    ]
    
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

if __name__ == "__main__":
    main()
