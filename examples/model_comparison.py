"""
Compare different model sizes and quantization methods
"""

from vllm import LLM, SamplingParams

# Model recommendations for AMD RX 7900 XTX (24GB VRAM)

MODELS = {
    "tiny": {
        "name": "facebook/opt-125m",
        "vram": "~1GB",
        "description": "Tiny model for testing"
    },
    "small": {
        "name": "facebook/opt-1.3b", 
        "vram": "~3GB",
        "description": "Small but coherent"
    },
    "medium_fp16": {
        "name": "mistralai/Mistral-7B-Instruct-v0.3",
        "vram": "~14GB", 
        "description": "Full precision 7B model"
    },
    "medium_awq": {
        "name": "TheBloke/Mistral-7B-Instruct-v0.2-AWQ",
        "vram": "~4GB",
        "description": "4-bit quantized, 75% memory savings"
    },
    "large_llama": {
        "name": "meta-llama/Llama-3.2-3B-Instruct",
        "vram": "~6GB",
        "description": "Meta's latest small model"
    },
    "expert_mixtral": {
        "name": "TheBloke/Mixtral-8x7B-Instruct-v0.1-AWQ",
        "vram": "~12GB",
        "description": "Mixture of Experts, 47B params but efficient"
    }
}

def show_recommendations():
    """Display model recommendations"""
    print("\n" + "="*80)
    print("📊 MODEL RECOMMENDATIONS FOR AMD RX 7900 XTX (24GB VRAM)")
    print("="*80 + "\n")
    
    for key, info in MODELS.items():
        print(f"🔹 {key.upper().replace('_', ' ')}")
        print(f"   Model: {info['name']}")
        print(f"   VRAM:  {info['vram']}")
        print(f"   Info:  {info['description']}")
        print()
    
    print("="*80)
    print("\n💡 TIPS:")
    print("   • Start with 'small' or 'medium_awq' for best balance")
    print("   • AWQ models are 4-bit quantized (faster + less memory)")
    print("   • FP16 models have slightly better quality but use 4x memory")
    print("   • Mixtral uses Mixture of Experts (best quality for size)")
    print("\n" + "="*80)

def test_model_quick(model_key):
    """Quick test of a specific model"""
    if model_key not in MODELS:
        print(f"❌ Model key '{model_key}' not found")
        print(f"Available: {', '.join(MODELS.keys())}")
        return
    
    model_info = MODELS[model_key]
    print(f"\n🧪 Testing: {model_info['name']}")
    print(f"Expected VRAM: {model_info['vram']}\n")
    
    llm = LLM(
        model=model_info['name'],
        gpu_memory_utilization=0.9
    )
    
    prompt = "What are the benefits of machine learning?"
    outputs = llm.generate([prompt], SamplingParams(temperature=0.7, max_tokens=100))
    
    print(f"\nPrompt: {prompt}")
    print(f"Response:\n{outputs[0].outputs[0].text}\n")

if __name__ == "__main__":
    show_recommendations()
    
    # Uncomment to test a specific model:
    # test_model_quick("small")
    # test_model_quick("medium_awq")
    
