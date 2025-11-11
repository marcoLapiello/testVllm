# Lesson 2: Exercises

Practice building flexible, production-ready vLLM scripts with command-line arguments!

## 🟢 Beginner Exercises

### Exercise 2.1: Run with Different Models
**Task**: Run the script with these models and compare outputs:
```bash
python text_generation.py --model facebook/opt-125m
python text_generation.py --model facebook/opt-350m
```

**Questions**:
- Which model gives better quality outputs?
- How much longer does the larger model take to load?
- Can you notice a difference in generation speed?

### Exercise 2.2: Use the Help System
**Task**: Explore available arguments:
```bash
python text_generation.py --help
```

**Questions**:
- How many argument groups are there?
- What's the default value for `--gpu-memory-utilization`?
- What quantization options are available?

### Exercise 2.3: Experiment with Temperature
**Task**: Run the script with different temperatures:
```bash
python text_generation.py --temperature 0.0
python text_generation.py --temperature 0.5
python text_generation.py --temperature 1.0
python text_generation.py --temperature 1.5
```

**Question**: At what temperature do outputs become nonsensical?

### Exercise 2.4: Adjust Max Tokens
**Task**: Generate different length outputs:
```bash
python text_generation.py --max-tokens 10
python text_generation.py --max-tokens 100
python text_generation.py --max-tokens 500
```

**Question**: Does setting a high max-tokens always result in that many tokens being generated? Why or why not?

## 🟡 Intermediate Exercises

### Exercise 2.5: Combine Multiple Arguments
**Task**: Create a configuration that produces:
- High-quality, focused outputs
- Exactly 150 tokens
- Using top-k sampling

**Hint**: Use temperature around 0.3-0.5, low top-p, and set top-k to 20.

### Exercise 2.6: GPU Memory Management
**Task**: If you have a GPU, experiment with memory utilization:
```bash
python text_generation.py --gpu-memory-utilization 0.5
python text_generation.py --gpu-memory-utilization 0.9
```

Monitor GPU usage with:
```bash
watch -n 1 nvidia-smi
```

**Question**: How does this affect performance and stability?

### Exercise 2.7: Create a Configuration File
**Task**: Create a JSON file `config.json` with your preferred settings:
```json
{
    "model": "facebook/opt-125m",
    "temperature": 0.7,
    "top_p": 0.95,
    "max_tokens": 100
}
```

Then modify the script to load these defaults.

**Hint**: Use Python's `json` module and `parser.set_defaults()`.

### Exercise 2.8: Add Custom Prompts Argument
**Task**: Modify `text_generation.py` to accept custom prompts via command line:
```bash
python text_generation.py --prompt "Once upon a time" --prompt "In the future"
```

**Hint**: Use `parser.add_argument("--prompt", action="append")`.

## 🔴 Advanced Exercises

### Exercise 2.9: Build a Batch Processing Script
**Task**: Create `batch_generate.sh` that:
1. Reads prompts from a file (one per line)
2. Generates completions for each
3. Saves results to separate output files

**Example**:
```bash
#!/bin/bash
while IFS= read -r prompt; do
    python text_generation.py \
        --temperature 0.8 \
        --max-tokens 100 \
        # How would you pass the prompt and save output?
done < prompts.txt
```

### Exercise 2.10: Parameter Sweep
**Task**: Write a script that tests different temperature/top-p combinations:
```python
temperatures = [0.3, 0.5, 0.7, 0.9, 1.1]
top_ps = [0.7, 0.8, 0.9, 0.95, 0.99]

for temp in temperatures:
    for top_p in top_ps:
        # Run generation and save results
        # Save to: results_temp{temp}_topp{top_p}.txt
```

**Goal**: Find the best parameter combination for your use case.

### Exercise 2.11: Add Logging
**Task**: Modify the script to log:
- Timestamp of each generation
- All parameters used
- Time taken for generation
- Output statistics (tokens, length, etc.)

**Hint**: Use Python's `logging` module or write to a CSV file.

### Exercise 2.12: Compare Quantization Methods
**Task**: If you have a larger model available, compare:
```bash
# Full precision
python text_generation.py --model <model-path>

# AWQ quantization
python text_generation.py --model <model-path> --quantization awq

# GPTQ quantization
python text_generation.py --model <model-path> --quantization gptq
```

**Question**: How do quality, speed, and memory usage compare?

### Exercise 2.13: Build a Simple Web API
**Task**: Create a Flask/FastAPI wrapper around the script:
```python
from flask import Flask, request, jsonify

app = Flask(__name__)
llm = None  # Initialize once

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    prompt = data.get('prompt')
    temperature = data.get('temperature', 0.7)
    # Generate and return
    return jsonify({'output': '...'})
```

### Exercise 2.14: Implement Result Caching
**Task**: Cache results to avoid regenerating the same prompt:
```python
import hashlib
import json
import os

def get_cache_key(prompt, params):
    """Create unique hash for prompt + parameters."""
    data = f"{prompt}{params}"
    return hashlib.md5(data.encode()).hexdigest()

def cache_result(key, result):
    """Save result to cache."""
    # Implement this
    pass

def get_cached_result(key):
    """Retrieve from cache if exists."""
    # Implement this
    pass
```

### Exercise 2.15: Add Validation and Error Handling
**Task**: Enhance the script with:
- Parameter validation (e.g., temperature >= 0)
- Graceful error handling
- Retry logic for transient failures
- Clear error messages for users

**Example validation**:
```python
def validate_args(args):
    if args.get('temperature') and args['temperature'] < 0:
        raise ValueError("Temperature must be non-negative")
    
    if args.get('max_tokens') and args['max_tokens'] > 4096:
        print("WARNING: max_tokens is very high, may be slow")
    
    # Add more validations
    return args
```

## 🎯 Challenge Project

### Project: Smart Content Generator
**Task**: Build a tool that:
1. Accepts a topic via command line
2. Generates multiple completions with different temperatures
3. Ranks outputs by quality (use length, diversity, coherence metrics)
4. Saves the best outputs to a markdown file
5. Creates a comparison table showing parameter effects

**Features to implement**:
- Multiple generation strategies
- Automatic quality assessment
- Formatted output with markdown
- Summary statistics
- Visualization of parameter effects (optional)

**Example usage**:
```bash
python smart_generator.py \
    --topic "Future of renewable energy" \
    --num-variations 5 \
    --output results.md
```

## 📊 Bonus: Performance Benchmarking

Create a benchmarking script that measures:
- Tokens per second
- Time to first token
- Total generation time
- Memory usage (GPU and RAM)

Compare across different:
- Models sizes
- Batch sizes
- Parameter configurations

## ✅ Self-Assessment

Before moving to Lesson 3, you should be able to:
- [ ] Use command-line arguments effectively
- [ ] Understand what EngineArgs provides
- [ ] Separate sampling params from engine args
- [ ] Override default parameters selectively
- [ ] Create reusable, configurable scripts
- [ ] Debug configuration issues
- [ ] Read and understand vLLM logs

## ➡️ Next Steps

Completed the exercises? Move on to [Lesson 3: Chat Interface](../lesson-03-chat-interface/README.md) to learn conversational AI!

---

**Need help?** Review the [concepts documentation](../concepts/) or revisit [Lesson 1](../lesson-01-basic-inference/README.md) for fundamentals.
