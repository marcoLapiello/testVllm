# Lesson 1: Basic Offline Inference

## 🎯 Learning Objectives

By the end of this lesson, you will:
- Understand what "offline inference" means in the context of vLLM
- Know how to create and use the `LLM` class
- Generate text from simple prompts
- Configure basic sampling parameters

## 📖 Concepts

### What is Offline Inference?

**Offline inference** means running a language model locally without using a separate inference server. You interact with the model directly through Python code, which is perfect for:
- Learning and experimentation
- Batch processing
- Scripts and automation
- Applications where you want full control

This is different from **online inference** (using an API server), which we'll explore in later lessons.

### The LLM Class

The `LLM` class is vLLM's main interface for offline inference. It:
- Loads a model into memory (GPU or CPU)
- Manages the inference engine
- Provides methods for text generation
- Handles batching automatically

### Sampling Parameters

When generating text, we don't always want the "most likely" next word. **Sampling parameters** control the randomness and creativity of generated text:

- **`temperature`**: Controls randomness (0.0 = deterministic, higher = more random)
  - `0.0-0.3`: Focused and deterministic (good for factual tasks)
  - `0.7-0.9`: Balanced creativity (good for general use)
  - `1.0+`: Very creative/random (good for brainstorming)

- **`top_p`** (nucleus sampling): Considers tokens that make up the top p% probability mass
  - `0.95`: Most common setting (considers 95% probability mass)
  - Lower values = more focused, higher = more diverse

## 💻 Code Walkthrough

### Script: `basic_inference.py`

```python
from vllm import LLM, SamplingParams

# Step 1: Define your prompts
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

# Step 2: Create sampling parameters
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

# Step 3: Create an LLM instance
llm = LLM(model="facebook/opt-125m")

# Step 4: Generate text
outputs = llm.generate(prompts, sampling_params)

# Step 5: Print results
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}")
    print(f"Output: {generated_text!r}")
    print("-" * 60)
```

### Breaking It Down

#### 1. **Import Required Classes**
```python
from vllm import LLM, SamplingParams
```
- `LLM`: The main class for loading and running models
- `SamplingParams`: Configuration for text generation behavior

#### 2. **Prepare Prompts**
```python
prompts = [
    "Hello, my name is",
    "The president of the United States is",
]
```
- Prompts are simply strings that the model will complete
- You can provide a list for batch processing (more efficient than one-by-one)

#### 3. **Configure Sampling**
```python
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
```
- `temperature=0.8`: Moderately creative responses
- `top_p=0.95`: Consider top 95% probability mass

#### 4. **Load the Model**
```python
llm = LLM(model="facebook/opt-125m")
```
- `model`: Can be a HuggingFace model ID or local path
- `facebook/opt-125m`: A small 125 million parameter model (great for learning!)
- This loads the model into memory (GPU if available, CPU otherwise)

#### 5. **Generate Text**
```python
outputs = llm.generate(prompts, sampling_params)
```
- Returns a list of `RequestOutput` objects
- Each output contains the prompt and generated text
- Batching happens automatically for efficiency

#### 6. **Process Results**
```python
for output in outputs:
    prompt = output.prompt          # Original prompt
    generated_text = output.outputs[0].text  # Generated continuation
```
- `output.outputs` is a list (we only requested one completion per prompt)
- `output.outputs[0].text` gives the actual generated text

## 🚀 Running the Example

### Option 1: Direct Execution
```bash
cd vllm-course/lesson-01-basic-inference
python basic_inference.py
```

### Option 2: With a Different Model
Modify the script to try different models:
```python
# Smaller model (faster, less capable)
llm = LLM(model="facebook/opt-125m")

# Slightly larger (1.3B parameters)
llm = LLM(model="facebook/opt-1.3b")
```

### Expected Output
```
Prompt: 'Hello, my name is'
Output: ' John Smith and I am a software engineer...'
------------------------------------------------------------
Prompt: 'The capital of France is'
Output: ' Paris, which is located in the north...'
------------------------------------------------------------
```

## 🎛️ Experiment with Parameters

Try modifying the sampling parameters to see how they affect output:

### Deterministic Output (temperature=0)
```python
sampling_params = SamplingParams(temperature=0.0, top_p=1.0)
```
- Produces the same output every time
- Good for reproducible results

### Creative Output (temperature=1.2)
```python
sampling_params = SamplingParams(temperature=1.2, top_p=0.95)
```
- More random and creative
- May produce unexpected results

### Focused Output (low top_p)
```python
sampling_params = SamplingParams(temperature=0.8, top_p=0.5)
```
- Considers fewer token options
- More predictable than high top_p

## 📊 Understanding the Output

Each `RequestOutput` object contains:
- **`prompt`**: Your input text
- **`outputs`**: List of completions (usually just one)
  - **`text`**: The generated text
  - **`token_ids`**: Token IDs of the generated text
  - **`cumulative_logprob`**: Log probability score
  - **`finish_reason`**: Why generation stopped (length, stop token, etc.)

## 🤔 Common Issues & Solutions

### Issue: "CUDA out of memory"
**Solution**: Use a smaller model or add `--gpu-memory-utilization 0.7`
```python
llm = LLM(model="facebook/opt-125m", gpu_memory_utilization=0.7)
```

### Issue: Model downloads slowly
**Solution**: Pre-download models:
```bash
huggingface-cli download facebook/opt-125m
```

### Issue: "No GPU found"
**Solution**: vLLM will automatically use CPU, but it's slower. This is fine for learning!

## 📝 Key Takeaways

1. ✅ The `LLM` class is your main entry point for offline inference
2. ✅ `SamplingParams` controls how creative/random the output is
3. ✅ vLLM automatically handles batching for efficiency
4. ✅ Start with small models (`opt-125m`) for learning
5. ✅ Temperature and top_p are your main creativity knobs

## ✏️ Exercises

See `exercises.md` for hands-on practice!

## ➡️ Next Lesson

Ready for more control? In [Lesson 2](../lesson-02-text-generation/README.md), we'll learn how to:
- Use command-line arguments
- Dynamically configure parameters
- Work with different sampling strategies
- Understand more advanced model options

---

**Questions or stuck?** Review the [concepts documentation](../concepts/sampling-parameters.md) or check the official [vLLM docs](https://docs.vllm.ai/).
