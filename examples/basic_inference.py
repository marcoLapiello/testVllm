"""
Lesson 1: Basic Offline Inference with vLLM

This script demonstrates the simplest way to use vLLM for text generation.
It covers the core concepts of:
1. Loading a model with the LLM class
2. Configuring sampling parameters
3. Generating text from prompts
4. Processing and displaying results

Based on: https://github.com/vllm-project/vllm/blob/main/examples/offline_inference/basic/basic.py
"""

from vllm import LLM, SamplingParams


def main():
    """Main function demonstrating basic vLLM inference."""
    
    # ========================================
    # STEP 1: Prepare Your Prompts
    # ========================================
    # These are the text snippets the model will complete
    # Think of them as "fill in the blank" exercises
    prompts = [
        "Write a comprehensive essay about the impact of artificial intelligence on modern society, covering both the benefits (such as automation, healthcare improvements, and scientific research) and the potential risks (including job displacement, privacy concerns, and ethical dilemmas). Discuss how society should balance innovation with regulation:",
    ]
    
    print("=" * 70)
    print("vLLM Basic Inference Example")
    print("=" * 70)
    print(f"\nPrompts to complete: {len(prompts)}")
    print("Model: mistralai/Mistral-7B-Instruct-v0.3 (7B instruction-tuned)")
    print()
    
    # ========================================
    # STEP 2: Configure Sampling Parameters
    # ========================================
    # These control how the model generates text
    sampling_params = SamplingParams(
        temperature=0.7,  # Controls randomness (0.0 = deterministic, 1.0+ = creative)
        top_p=0.95,       # Nucleus sampling - consider top 95% probability mass
        max_tokens=400,   # Maximum length of generated text
    )
    
    print("Sampling Parameters:")
    print(f"  - Temperature: {sampling_params.temperature}")
    print(f"  - Top-p: {sampling_params.top_p}")
    print(f"  - Max tokens: {sampling_params.max_tokens}")
    print()
    
    # ========================================
    # STEP 3: Create the LLM Instance
    # ========================================
    # This loads the model into memory (GPU if available, otherwise CPU)
    print("Loading model... (this may take a minute on first run)")
    llm = LLM(
        model="mistralai/Mistral-7B-Instruct-v0.3",  # Instruction-tuned 7B model
        max_model_len=8192,  # Reduce to fit in GPU memory
        # Uncomment below if you have GPU memory issues:
        # gpu_memory_utilization=0.7,
    )
    print("Model loaded successfully! ✓")
    print()
    
    # ========================================
    # STEP 4: Generate Text
    # ========================================
    # The generate() method processes all prompts in a batch for efficiency
    print("Generating text...")
    outputs = llm.generate(prompts, sampling_params)
    print(f"Generated {len(outputs)} completions! ✓")
    print()
    
    # ========================================
    # STEP 5: Display Results
    # ========================================
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print()
    
    for i, output in enumerate(outputs, 1):
        prompt = output.prompt
        generated_text = output.outputs[0].text
        
        print(f"[{i}/{len(outputs)}]")
        print(f"Prompt:    {prompt!r}")
        print(f"Generated: {generated_text!r}")
        print("-" * 70)
        print()



if __name__ == "__main__":
    # This block runs when you execute the script directly
    # Example: python basic_inference.py
    main()
