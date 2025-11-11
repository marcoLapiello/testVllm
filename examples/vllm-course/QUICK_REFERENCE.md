# vLLM Quick Reference Guide

A cheat sheet for common vLLM operations and patterns. Bookmark this page for quick lookups!

## 🚀 Basic Patterns

### Minimal Example
```python
from vllm import LLM, SamplingParams

llm = LLM(model="facebook/opt-125m")
outputs = llm.generate(["Hello, my name is"], SamplingParams(temperature=0.8))
print(outputs[0].outputs[0].text)
```

### With Command-Line Arguments
```python
from vllm import LLM, EngineArgs
from vllm.utils.argparse_utils import FlexibleArgumentParser

parser = FlexibleArgumentParser()
EngineArgs.add_cli_args(parser)
args = vars(parser.parse_args())

llm = LLM(**args)
outputs = llm.generate(prompts, sampling_params)
```

## 📋 Common Commands

### Run with Custom Model
```bash
python script.py --model facebook/opt-1.3b
```

### Adjust Sampling
```bash
python script.py --temperature 0.5 --top-p 0.9 --max-tokens 200
```

### GPU Memory Control
```bash
python script.py --gpu-memory-utilization 0.7
```

### Use Quantization
```bash
python script.py --model <model-path> --quantization awq
```

### CPU Offload
```bash
python script.py --cpu-offload-gb 10
```

## 🎛️ Sampling Parameters Quick Reference

| Use Case | temperature | top_p | top_k | Notes |
|----------|-------------|-------|-------|-------|
| **Factual Q&A** | 0.0 | 1.0 | -1 | Deterministic |
| **Code Generation** | 0.2 | 0.9 | 40 | Focused |
| **General Chat** | 0.7 | 0.95 | -1 | Balanced |
| **Creative Writing** | 1.0 | 0.95 | -1 | Diverse |
| **Brainstorming** | 1.2-1.5 | 0.98 | -1 | Very creative |

## 🔧 Common Configurations

### Low Memory Setup
```python
llm = LLM(
    model="facebook/opt-125m",
    gpu_memory_utilization=0.6,
    max_model_len=1024,
)
```

### High Performance Setup
```python
llm = LLM(
    model="facebook/opt-1.3b",
    gpu_memory_utilization=0.9,
    max_num_batched_tokens=8192,
)
```

### Quantized Model
```python
llm = LLM(
    model="TheBloke/Llama-2-7B-GPTQ",
    quantization="gptq",
)
```

## 📊 Accessing Output Information

```python
output = outputs[0]

# Basic info
prompt = output.prompt
text = output.outputs[0].text

# Detailed info
tokens = output.outputs[0].token_ids
num_tokens = len(tokens)
finish_reason = output.outputs[0].finish_reason
log_prob = output.outputs[0].cumulative_logprob
```

## 🔍 Debugging Tips

### Check GPU Usage
```bash
watch -n 1 nvidia-smi
```

### Enable Verbose Logging
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Test with Smaller Model First
```python
# Use this for debugging
llm = LLM(model="facebook/opt-125m")

# Then switch to production model
# llm = LLM(model="meta-llama/Llama-2-7b-hf")
```

### Force CPU Mode
```python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
llm = LLM(model="facebook/opt-125m")
```

## 🐛 Common Issues & Fixes

### Issue: CUDA Out of Memory
```python
# Solution 1: Reduce memory utilization
llm = LLM(model="...", gpu_memory_utilization=0.7)

# Solution 2: Use smaller model
llm = LLM(model="facebook/opt-125m")

# Solution 3: Reduce max length
llm = LLM(model="...", max_model_len=1024)
```

### Issue: Model Loading Slow
```bash
# Pre-download models
huggingface-cli download facebook/opt-125m
```

### Issue: Import Error
```bash
# Ensure vLLM is installed
pip install vllm

# Or upgrade
pip install --upgrade vllm
```

## 📦 Model Selection Guide

| Model Size | Parameters | VRAM Needed | Use Case |
|------------|------------|-------------|----------|
| **Tiny** | 125M-350M | 1-2 GB | Learning, testing |
| **Small** | 1B-3B | 4-8 GB | Development, demos |
| **Medium** | 7B-13B | 16-24 GB | Production, quality |
| **Large** | 30B-70B | 80+ GB | High-end applications |

## 🎨 Code Snippets

### Batch Different Prompts
```python
prompts = [
    "Complete this: Once upon",
    "Translate to French: Hello",
    "Summarize: Long text here...",
]
outputs = llm.generate(prompts, sampling_params)
```

### Generate Multiple Completions per Prompt
```python
sampling_params = SamplingParams(
    n=3,  # Generate 3 completions
    temperature=0.8,
)
outputs = llm.generate([prompt], sampling_params)

for i, completion in enumerate(outputs[0].outputs):
    print(f"Completion {i+1}: {completion.text}")
```

### Stream Tokens (Not Available in Offline Mode)
*Note: Use the Online API Server for streaming*

### Use Stop Sequences
```python
sampling_params = SamplingParams(
    stop=["\n\n", "END", "###"],
    max_tokens=200,
)
```

## 🔗 Useful Links

- [Official Documentation](https://docs.vllm.ai/)
- [GitHub Repository](https://github.com/vllm-project/vllm)
- [Model Support List](https://docs.vllm.ai/en/latest/models/supported_models.html)
- [API Reference](https://docs.vllm.ai/en/latest/api/)

## 💾 Save This Script Template

```python
#!/usr/bin/env python3
"""
Template for vLLM inference scripts.
"""
from vllm import LLM, EngineArgs, SamplingParams
from vllm.utils.argparse_utils import FlexibleArgumentParser


def create_parser():
    parser = FlexibleArgumentParser()
    EngineArgs.add_cli_args(parser)
    parser.set_defaults(model="facebook/opt-125m")
    
    # Add custom args here
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=100)
    
    return parser


def main():
    parser = create_parser()
    args = vars(parser.parse_args())
    
    # Extract custom args
    temperature = args.pop("temperature")
    max_tokens = args.pop("max_tokens")
    
    # Create LLM
    llm = LLM(**args)
    
    # Create sampling params
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
    )
    
    # Your logic here
    prompts = ["Your prompt"]
    outputs = llm.generate(prompts, sampling_params)
    
    for output in outputs:
        print(output.outputs[0].text)


if __name__ == "__main__":
    main()
```

---

**Pro Tip**: Keep this reference open in a separate tab while working through the lessons!
