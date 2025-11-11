# Lesson 2: Text Generation with Parameters

## 🎯 Learning Objectives

By the end of this lesson, you will:
- Use command-line arguments to make scripts flexible
- Understand the `EngineArgs` class for model configuration
- Dynamically configure sampling parameters
- Work with vLLM's `FlexibleArgumentParser`
- Build reusable, production-ready inference scripts

## 📖 Concepts

### Why Command-Line Arguments?

In Lesson 1, we hardcoded everything:
```python
llm = LLM(model="facebook/opt-125m")
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
```

This is fine for learning, but in real applications you want **flexibility**:
- Try different models without editing code
- Tune parameters without restarting Python
- Share scripts with others who can customize behavior
- Automate experiments with shell scripts

### The FlexibleArgumentParser

vLLM provides `FlexibleArgumentParser`, which extends Python's standard `ArgumentParser` with:
- Better handling of boolean flags
- Underscore/hyphen compatibility (`--max-tokens` = `--max_tokens`)
- Integration with vLLM's configuration system

### EngineArgs: Model Configuration

The `EngineArgs` class contains ALL configuration options for the vLLM engine:
- Model loading (`model`, `tokenizer`, `revision`)
- Quantization (`quantization`, `load-format`)
- GPU settings (`tensor-parallel-size`, `gpu-memory-utilization`)
- Performance tuning (`max-model-len`, `block-size`)

By using `EngineArgs.add_cli_args(parser)`, we automatically get access to all these options!

### Separating Concerns

Good script structure:
```
1. Parse arguments (engine + sampling parameters)
2. Extract sampling params from args
3. Create LLM with engine args
4. Get default sampling params
5. Override with user values
6. Generate text
```

This separation allows maximum flexibility while keeping code clean.

## 💻 Code Walkthrough

### Script: `text_generation.py`

Let's break down the key parts:

#### 1. **Argument Parser Setup**
```python
from vllm import LLM, EngineArgs
from vllm.utils.argparse_utils import FlexibleArgumentParser

def create_parser():
    parser = FlexibleArgumentParser()
    
    # Add ALL vLLM engine arguments
    EngineArgs.add_cli_args(parser)
    
    # Set a default model
    parser.set_defaults(model="facebook/opt-125m")
    
    # Add custom sampling parameters
    sampling_group = parser.add_argument_group("Sampling parameters")
    sampling_group.add_argument("--max-tokens", type=int)
    sampling_group.add_argument("--temperature", type=float)
    sampling_group.add_argument("--top-p", type=float)
    sampling_group.add_argument("--top-k", type=int)
    
    return parser
```

**Key points**:
- `EngineArgs.add_cli_args()` adds ~50+ arguments automatically
- We create a separate group for sampling parameters (better `--help` output)
- Default values can be set with `set_defaults()`

#### 2. **Extracting Arguments**
```python
def main(args: dict):
    # Pop sampling params (not used by LLM constructor)
    max_tokens = args.pop("max_tokens")
    temperature = args.pop("temperature")
    top_p = args.pop("top_p")
    top_k = args.pop("top_k")
    
    # Remaining args go to LLM
    llm = LLM(**args)
```

**Why `pop()`?**
- `args` is a dictionary of all parsed arguments
- `LLM()` only accepts engine arguments
- We `pop()` sampling params to remove them from the dict
- Remaining args can be safely passed to `LLM(**args)`

#### 3. **Smart Default Handling**
```python
# Get model's default sampling params
sampling_params = llm.get_default_sampling_params()

# Override with user-provided values
if max_tokens is not None:
    sampling_params.max_tokens = max_tokens
if temperature is not None:
    sampling_params.temperature = temperature
# ... etc
```

**Benefits**:
- Users can override just what they want
- Unspecified params use model defaults
- No need to set every parameter

## 🚀 Running the Example

### Basic Usage
```bash
cd vllm-course/lesson-02-text-generation
python text_generation.py
```

### With Custom Model
```bash
python text_generation.py --model facebook/opt-1.3b
```

### With Custom Sampling
```bash
python text_generation.py \
    --temperature 0.5 \
    --top-p 0.9 \
    --max-tokens 100
```

### With Multiple Options
```bash
python text_generation.py \
    --model facebook/opt-350m \
    --temperature 1.0 \
    --top-p 0.95 \
    --max-tokens 150 \
    --gpu-memory-utilization 0.8
```

### See All Available Options
```bash
python text_generation.py --help
```

This shows ALL available arguments (there are many!)

## 🎛️ Common Engine Arguments

Here are some frequently used engine arguments:

### Model Selection
```bash
--model facebook/opt-125m          # HuggingFace model ID
--model /path/to/local/model       # Local model path
--tokenizer facebook/opt-125m      # Separate tokenizer (optional)
```

### GPU Configuration
```bash
--gpu-memory-utilization 0.9       # Use 90% of GPU memory (default: 0.9)
--tensor-parallel-size 2           # Use 2 GPUs for one model
--max-model-len 2048               # Override max sequence length
```

### Quantization
```bash
--quantization awq                 # Use AWQ quantization
--quantization gptq                # Use GPTQ quantization
--load-format auto                 # Auto-detect format
```

### Performance
```bash
--enforce-eager                    # Disable CUDA graphs (for debugging)
--max-num-batched-tokens 8192     # Max tokens per batch
--max-num-seqs 256                # Max sequences per iteration
```

### CPU Offload
```bash
--cpu-offload-gb 10               # Offload 10GB to CPU memory
```

## 📊 Practical Examples

### Example 1: Low Memory Setup
```bash
python text_generation.py \
    --model facebook/opt-125m \
    --gpu-memory-utilization 0.7 \
    --max-model-len 1024 \
    --temperature 0.8
```

### Example 2: High Quality Generation
```bash
python text_generation.py \
    --model facebook/opt-1.3b \
    --temperature 0.3 \
    --top-p 0.7 \
    --max-tokens 200
```

### Example 3: Creative Story Generation
```bash
python text_generation.py \
    --temperature 1.2 \
    --top-p 0.98 \
    --presence-penalty 0.5 \
    --max-tokens 300
```

### Example 4: Batch Processing Script
```bash
#!/bin/bash
# Run multiple experiments with different temperatures

for temp in 0.0 0.5 1.0 1.5; do
    echo "Testing temperature: $temp"
    python text_generation.py --temperature $temp > output_temp_${temp}.txt
done
```

## 🔍 Understanding the Output

The script provides detailed output:

```
Configuration:
  Model: facebook/opt-125m
  Temperature: 0.8
  Top-p: 0.95
  Max tokens: 50

Generated Outputs:
--------------------------------------------------------------------------------
Prompt: "Hello, my name is"
Generated text: " John Smith and I am a software engineer..."
--------------------------------------------------------------------------------
```

### Reading the Logs

vLLM also prints initialization logs:
```
INFO: Initializing an LLM engine with config:
INFO:   model='facebook/opt-125m'
INFO:   tokenizer='facebook/opt-125m'
INFO:   tensor_parallel_size=1
INFO:   gpu_memory_utilization=0.9
```

These help you understand exactly what configuration is being used.

## 🤔 Common Patterns

### Pattern 1: Environment-Specific Defaults
```python
parser.set_defaults(
    model="facebook/opt-125m" if DEBUG else "facebook/opt-1.3b",
    gpu_memory_utilization=0.7 if DEBUG else 0.9,
)
```

### Pattern 2: Config Files
```python
import json

# Load config from file
with open("config.json") as f:
    config = json.load(f)

# Apply to parser
parser.set_defaults(**config)
```

### Pattern 3: Validation
```python
def validate_args(args):
    if args["temperature"] < 0:
        raise ValueError("Temperature must be non-negative")
    if args["max_tokens"] > 2048:
        print("Warning: max_tokens is very high")
    return args
```

## 📝 Key Takeaways

1. ✅ `FlexibleArgumentParser` makes CLI argument handling easy
2. ✅ `EngineArgs.add_cli_args()` gives access to all vLLM options
3. ✅ Separate sampling params from engine args using `pop()`
4. ✅ Use `get_default_sampling_params()` for smart defaults
5. ✅ Command-line arguments make scripts reusable and flexible
6. ✅ Always provide `--help` for documentation
7. ✅ You can override any parameter without editing code

## ✏️ Exercises

See `exercises.md` for hands-on practice!

## ➡️ Next Lesson

Ready to build conversational AI? In [Lesson 3](../lesson-03-chat-interface/README.md), we'll learn:
- Chat-style interactions with message roles
- Using chat templates
- Building multi-turn conversations
- Batching conversations efficiently

---

**Need clarification?** Review the [Engine Arguments concepts](../concepts/engine-arguments.md) or check [Lesson 1](../lesson-01-basic-inference/README.md) for basics.
