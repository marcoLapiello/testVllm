"""
Batch inference example with vLLM on ROCm
Demonstrates processing multiple prompts efficiently
"""

from vllm import LLM, SamplingParams
import time

def main():
    # Use a small model for testing
    model_name = "facebook/opt-125m"
    
    print(f"Loading model: {model_name}")
    llm = LLM(
        model=model_name,
        tensor_parallel_size=1,  # Adjust based on your GPU setup
    )
    
    # Sampling parameters
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=50,
    )
    
    # Create a larger batch of prompts
    prompts = [
        "Artificial intelligence is",
        "The future of computing",
        "Machine learning algorithms",
        "Deep neural networks",
        "Natural language processing",
        "Computer vision applications",
        "Reinforcement learning enables",
        "Cloud computing provides",
        "Data science involves",
        "Python programming language",
    ]
    
    print(f"\nProcessing {len(prompts)} prompts in batch...")
    start_time = time.time()
    
    outputs = llm.generate(prompts, sampling_params)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print(f"\n✓ Completed in {elapsed_time:.2f} seconds")
    print(f"✓ Average time per prompt: {elapsed_time/len(prompts):.2f} seconds\n")
    
    # Display results
    for i, output in enumerate(outputs, 1):
        print(f"[{i}/{len(prompts)}] Prompt: {output.prompt}")
        print(f"Generated: {output.outputs[0].text}\n")
        print("-" * 80)

if __name__ == "__main__":
    main()
