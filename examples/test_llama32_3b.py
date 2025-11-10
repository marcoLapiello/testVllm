"""
Test Llama 3.2 3B Instruct - Full Precision (FP16)
This is Meta's latest small model with excellent quality
"""

from vllm import LLM, SamplingParams
import time

def main():
    model_name = "meta-llama/Llama-3.2-3B-Instruct"
    
    print("="*80)
    print(f"🦙 Testing: {model_name}")
    print("="*80)
    print("\nModel Info:")
    print("  - Size: 3 Billion parameters")
    print("  - Precision: FP16 (full precision)")
    print("  - Expected VRAM: ~6-7 GB")
    print("  - Context length: 131,072 tokens (128K!)")
    print("\nLoading model...\n")
    
    start_load = time.time()
    
    # Load the model
    llm = LLM(
        model=model_name,
        max_model_len=4096,  # Use 4K context (can go up to 128K)
        gpu_memory_utilization=0.9,  # Use 90% of GPU memory
        tensor_parallel_size=1,  # Single GPU
    )
    
    load_time = time.time() - start_load
    print(f"✅ Model loaded in {load_time:.2f} seconds\n")
    
    # Sampling parameters for good quality
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=300,
    )
    
    # Test prompts - Llama 3.2 uses chat format
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
    
    # Batch test
    print("\n" + "="*80)
    print("🔄 BONUS: Batch Processing Test")
    print("="*80)
    
    batch_prompts = [
        "What is artificial intelligence?",
        "Explain neural networks briefly.",
        "What is the purpose of machine learning?"
    ]
    
    print(f"\nProcessing {len(batch_prompts)} prompts in batch...")
    start_batch = time.time()
    batch_outputs = llm.generate(batch_prompts, SamplingParams(
        temperature=0.7,
        max_tokens=100
    ))
    batch_time = time.time() - start_batch
    
    print(f"✅ Completed in {batch_time:.2f}s")
    print(f"⚡ Average: {batch_time/len(batch_prompts):.2f}s per prompt\n")
    
    for i, output in enumerate(batch_outputs, 1):
        print(f"{i}. Q: {output.prompt}")
        print(f"   A: {output.outputs[0].text[:100]}...\n")

if __name__ == "__main__":
    main()
