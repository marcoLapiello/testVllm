"""
Simple vLLM inference example for ROCm
This script demonstrates basic text generation using vLLM
"""

from vllm import LLM, SamplingParams

def main():
    # Initialize the model
    # You can change this to any supported model from Hugging Face
    model_name = "facebook/opt-125m"  # Small model for testing
    
    print(f"Loading model: {model_name}")
    llm = LLM(model=model_name)
    
    # Define sampling parameters
    sampling_params = SamplingParams(
        temperature=0.8,
        top_p=0.95,
        max_tokens=100
    )
    
    # Example prompts
    prompts = [
        "Hello, my name is",
        "The capital of France is",
        "In machine learning,",
    ]
    
    print("\nGenerating responses...")
    outputs = llm.generate(prompts, sampling_params)
    
    # Print the results
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"\nPrompt: {prompt}")
        print(f"Generated: {generated_text}")
        print("-" * 80)

if __name__ == "__main__":
    main()
