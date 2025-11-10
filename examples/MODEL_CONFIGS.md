# Model Configuration Examples

This document provides ready-to-use configurations for the models available in this workspace.

## How to Use

1. Open `test_model.py`
2. Find the **CONFIGURATION SECTION** in the `main()` function
3. Copy one of the configurations below and replace the existing configuration
4. Run the script inside the Docker container: `sudo docker exec -it vllm-rocm bash -c "cd /app/examples && python test_model.py"`

---

## Available Models

### Facebook OPT-125M (Tiny, Fast Testing) ⚡

**Status:** ✅ Downloaded and ready to use

Perfect for quick testing and validation!

```python
model_config = {
    'name': "facebook/opt-125m",
    'info': {
        'Size': '125 Million parameters',
        'Precision': 'FP16',
        'Expected VRAM': '~500 MB',
        'Context length': '2,048 tokens',
        'Notes': 'Very small, fast for quick tests',
    }
}
```

---

### Mistral 7B Instruct v0.3 (High Quality) 🌟

**Status:** ✅ Downloaded and ready to use

Excellent open model with strong performance!

```python
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
```

---

## Parameter Reference

### vllm_params

- **max_model_len**: Maximum context length to use (tokens)
- **gpu_memory_utilization**: Fraction of GPU memory to use (0.0-1.0)
- **tensor_parallel_size**: Number of GPUs for tensor parallelism
- **quantization**: Quantization method ('awq', 'gptq', etc.)

### sampling_config

- **temperature**: Controls randomness (0.0 = deterministic, 1.0+ = very creative)
- **top_p**: Nucleus sampling threshold (0.9 = use top 90% probability mass)
- **max_tokens**: Maximum number of tokens to generate
- **top_k**: Consider only top k tokens (optional)
- **repetition_penalty**: Penalty for repeating tokens (optional, default 1.0)

---

## Custom Test Cases

You can also customize the test cases. Here are some examples:

### Code-Focused Tests

```python
test_cases = [
    {
        "prompt": "Write a Python function to implement binary search.",
        "description": "Algorithm implementation"
    },
    {
        "prompt": "Explain the difference between a list and a tuple in Python.",
        "description": "Language concepts"
    },
    {
        "prompt": "Debug this code: for i in range(10) print(i)",
        "description": "Debugging"
    }
]
```

### Creative Writing Tests

```python
test_cases = [
    {
        "prompt": "Write a haiku about machine learning.",
        "description": "Poetry"
    },
    {
        "prompt": "Create a short story opening about a time traveler.",
        "description": "Story writing"
    },
    {
        "prompt": "Write a product description for a smart coffee maker.",
        "description": "Marketing copy"
    }
]
```

### Technical Q&A Tests

```python
test_cases = [
    {
        "prompt": "What is the CAP theorem in distributed systems?",
        "description": "Distributed systems"
    },
    {
        "prompt": "Explain REST API best practices.",
        "description": "API design"
    },
    {
        "prompt": "What are the SOLID principles in software engineering?",
        "description": "Design patterns"
    }
]
```

---

## Tips

2. **Check VRAM**: Use `rocm-smi` to monitor GPU memory usage
3. **Adjust Memory**: Lower `gpu_memory_utilization` if you get OOM errors
4. **Context Length**: Larger `max_model_len` requires more VRAM
5. **Batch Testing**: Disable by setting `batch_prompts = None` if not needed
6. **Temperature**: Lower (0.3-0.5) for factual, higher (0.8-1.2) for creative tasks
