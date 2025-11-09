"""
Test quantized models with vLLM on ROCm
Demonstrates using AWQ/GPTQ quantized models for faster inference
"""

from vllm import LLM, SamplingParams
import time

def test_model(model_name, prompt, max_tokens=200):
    """Test a model and measure performance"""
    print(f"\n{'='*80}")
    print(f"Testing: {model_name}")
    print(f"{'='*80}\n")
    
    try:
        # Load model
        start_time = time.time()
        llm = LLM(
            model=model_name,
            max_model_len=2048,  # Adjust based on model
            gpu_memory_utilization=0.9,  # Use 90% of GPU memory
        )
        load_time = time.time() - start_time
        print(f"✓ Model loaded in {load_time:.2f} seconds\n")
        
        # Configure sampling
        sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.9,
            max_tokens=max_tokens,
        )
        
        # Generate
        print(f"Prompt: {prompt}\n")
        start_time = time.time()
        outputs = llm.generate([prompt], sampling_params)
        gen_time = time.time() - start_time
        
        # Display results
        generated = outputs[0].outputs[0].text
        print(f"Generated:\n{generated}\n")
        print(f"✓ Generation time: {gen_time:.2f} seconds")
        print(f"✓ Tokens generated: {len(generated.split())}")
        print(f"✓ Speed: {len(generated.split())/gen_time:.2f} tokens/sec (approx)\n")
        
    except Exception as e:
        print(f"❌ Error: {e}\n")

def main():
    # Test prompt
    prompt = """Explain what machine learning is and give a practical example 
    of how it's used in everyday life."""
    
    print("\n🧪 vLLM Quantized Model Testing")
    print("=" * 80)
    
    # You can test different models by uncommenting them:
    
    # Small model (baseline)
    # test_model("facebook/opt-1.3b", prompt, max_tokens=150)
    
    # AWQ Quantized (recommended - requires AWQ support)
    # test_model("TheBloke/Mistral-7B-Instruct-v0.2-AWQ", prompt)
    
    # For now, let's test with a standard quantization
    # Meta Llama 3.2 3B (no special quantization needed, small enough)
    print("\n⚠️  Note: Some quantized models may require additional setup.")
    print("For AWQ/GPTQ, vLLM should auto-detect and use them.\n")
    
    # Test a reasonably sized model
    test_model("facebook/opt-1.3b", prompt, max_tokens=150)
    
    print("\n" + "="*80)
    print("🎯 To test larger/quantized models, edit this script and uncomment:")
    print("   - TheBloke/Mistral-7B-Instruct-v0.2-AWQ")
    print("   - meta-llama/Llama-3.2-3B-Instruct")
    print("="*80)

if __name__ == "__main__":
    main()
