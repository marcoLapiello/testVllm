# Sampling Parameters Deep Dive

Understanding how to control text generation is crucial for getting the best results from language models. This guide explains all the key sampling parameters in vLLM.

## 📊 Overview

When a language model generates text, it doesn't just pick the "best" word each time. Instead, it **samples** from a probability distribution of possible next tokens. Sampling parameters control this process.

## 🎯 Core Parameters

### 1. Temperature

**What it does**: Controls the randomness of predictions by scaling the logits (raw model outputs) before sampling.

**Range**: 0.0 to ∞ (typically 0.0-2.0)

**How it works**:
```
temperature = 0.0 → Always pick the most likely token (greedy decoding)
temperature = 0.5 → Favor likely tokens, but with some variation
temperature = 1.0 → Use raw probabilities from the model
temperature = 2.0 → Very random, even unlikely tokens get a chance
```

**Visual Example**:
```
Original probabilities: [0.5, 0.3, 0.15, 0.05]
Temperature = 0.5:      [0.65, 0.25, 0.08, 0.02]  (sharper)
Temperature = 1.0:      [0.5, 0.3, 0.15, 0.05]    (unchanged)
Temperature = 2.0:      [0.35, 0.30, 0.22, 0.13]  (flatter)
```

**Use cases**:
- **0.0**: Factual Q&A, code generation, translations (deterministic)
- **0.3-0.7**: General tasks requiring accuracy with slight variation
- **0.8-1.0**: Creative writing, brainstorming
- **1.5+**: Experimental, surreal, or highly diverse outputs

**Code example**:
```python
from vllm import SamplingParams

# Deterministic (always the same output)
params_factual = SamplingParams(temperature=0.0)

# Balanced (good for most tasks)
params_balanced = SamplingParams(temperature=0.7)

# Creative (for story writing)
params_creative = SamplingParams(temperature=1.2)
```

---

### 2. Top-p (Nucleus Sampling)

**What it does**: Only considers the smallest set of tokens whose cumulative probability exceeds `p`.

**Range**: 0.0 to 1.0

**How it works**:
1. Sort all tokens by probability (highest first)
2. Keep adding tokens until cumulative probability ≥ `p`
3. Sample only from these tokens
4. Ignore all other tokens

**Visual Example**:
```
All tokens sorted by probability:
Token A: 40%  ├──┐
Token B: 25%  │  │
Token C: 15%  │  ├─ These 3 tokens = 80% (top_p = 0.8)
Token D: 10%  │  │
Token E: 5%   ├──┘
Token F: 3%   ← Ignored
Token G: 2%   ← Ignored
```

With `top_p = 0.8`, only tokens A, B, and C are considered.

**Use cases**:
- **0.5-0.7**: Very focused, predictable outputs
- **0.9**: Balanced (most common setting)
- **0.95-0.99**: More diverse, includes rarer tokens
- **1.0**: Consider all tokens (no filtering)

**Code example**:
```python
# Focused generation (only top tokens)
params_focused = SamplingParams(top_p=0.5)

# Standard generation
params_standard = SamplingParams(top_p=0.95)

# Maximum diversity
params_diverse = SamplingParams(top_p=1.0)
```

---

### 3. Top-k

**What it does**: Only considers the top `k` most likely tokens.

**Range**: 1 to vocabulary size (typically 1-100)

**How it works**:
1. Sort all tokens by probability
2. Keep only the top `k` tokens
3. Sample from these `k` tokens
4. Ignore all others

**Comparison with top-p**:
```
top_k = 5  → Always consider exactly 5 tokens
top_p = 0.9 → Consider variable number of tokens (might be 3, might be 10)
```

**Use cases**:
- **1**: Greedy decoding (always pick the top token)
- **10-20**: Focused generation
- **40-50**: Balanced diversity
- **100+**: Very diverse (but may include nonsense)

**Code example**:
```python
# Only top 10 tokens
params_k10 = SamplingParams(top_k=10)

# Top 50 tokens (more diverse)
params_k50 = SamplingParams(top_k=50)
```

**Note**: Can be combined with `top_p`:
```python
# First apply top_k, then top_p on remaining tokens
params_combined = SamplingParams(top_k=50, top_p=0.95)
```

---

### 4. Max Tokens

**What it does**: Maximum number of tokens to generate.

**Range**: 1 to model's context length

**Important**: This is an upper limit. Generation might stop earlier due to:
- Stop sequences (like `\n\n` or special tokens)
- End-of-sequence token
- Natural completion

**Code example**:
```python
# Short completion
params_short = SamplingParams(max_tokens=10)

# Medium completion
params_medium = SamplingParams(max_tokens=100)

# Long completion
params_long = SamplingParams(max_tokens=512)
```

---

### 5. Presence Penalty

**What it does**: Reduces probability of tokens that have already appeared (encourages diversity).

**Range**: -2.0 to 2.0 (typically 0.0 to 1.0)

**How it works**:
- **0.0**: No penalty (default)
- **Positive values**: Discourage repetition
- **Negative values**: Encourage repetition (rare)

**Use case**: Prevent the model from repeating the same phrases.

```python
params = SamplingParams(presence_penalty=0.5)
```

---

### 6. Frequency Penalty

**What it does**: Reduces probability based on how often a token has appeared.

**Range**: -2.0 to 2.0 (typically 0.0 to 1.0)

**Difference from presence penalty**:
- **Presence penalty**: Applies same penalty regardless of count (appeared once = appeared 10 times)
- **Frequency penalty**: Stronger penalty for tokens that appear more often

**Use case**: Strongly discourage repetitive text.

```python
params = SamplingParams(frequency_penalty=0.7)
```

---

## 🎨 Common Parameter Combinations

### Factual/Deterministic
```python
SamplingParams(
    temperature=0.0,
    top_p=1.0,
    max_tokens=100
)
```
**Use for**: Factual Q&A, code generation, data extraction

### Balanced/General Purpose
```python
SamplingParams(
    temperature=0.7,
    top_p=0.95,
    max_tokens=150
)
```
**Use for**: Chatbots, content generation, most applications

### Creative/Diverse
```python
SamplingParams(
    temperature=1.0,
    top_p=0.95,
    presence_penalty=0.5,
    max_tokens=200
)
```
**Use for**: Story writing, brainstorming, creative tasks

### Focused/Consistent
```python
SamplingParams(
    temperature=0.3,
    top_p=0.7,
    top_k=20,
    max_tokens=100
)
```
**Use for**: Summaries, formal writing, technical documentation

---

## 🔬 Experimentation Guide

### Finding the Right Parameters

1. **Start with defaults**:
   ```python
   params = SamplingParams(temperature=0.7, top_p=0.95)
   ```

2. **Adjust temperature first**:
   - Too boring/repetitive? → Increase temperature
   - Too random/nonsensical? → Decrease temperature

3. **Fine-tune with top_p**:
   - Need more consistency? → Lower top_p (0.7-0.8)
   - Want more variety? → Raise top_p (0.98-0.99)

4. **Add penalties if needed**:
   - Model repeating itself? → Add presence_penalty
   - Specific phrases repeating? → Add frequency_penalty

### Interactive Testing Script

```python
from vllm import LLM, SamplingParams

def test_parameters():
    llm = LLM(model="facebook/opt-125m")
    prompt = "The future of artificial intelligence is"
    
    configs = [
        ("Deterministic", {"temperature": 0.0}),
        ("Conservative", {"temperature": 0.3, "top_p": 0.7}),
        ("Balanced", {"temperature": 0.7, "top_p": 0.95}),
        ("Creative", {"temperature": 1.2, "top_p": 0.95}),
    ]
    
    for name, params in configs:
        sampling_params = SamplingParams(**params, max_tokens=50)
        output = llm.generate([prompt], sampling_params)[0]
        print(f"\n{name}:")
        print(output.outputs[0].text)
        print("-" * 60)

if __name__ == "__main__":
    test_parameters()
```

---

## 📚 Quick Reference

| Parameter | Typical Range | Default | Effect |
|-----------|---------------|---------|--------|
| `temperature` | 0.0 - 2.0 | 1.0 | Randomness |
| `top_p` | 0.0 - 1.0 | 1.0 | Diversity (nucleus) |
| `top_k` | 1 - 100 | -1 (off) | Diversity (count) |
| `max_tokens` | 1 - context_len | 16 | Length limit |
| `presence_penalty` | 0.0 - 2.0 | 0.0 | Discourage repetition |
| `frequency_penalty` | 0.0 - 2.0 | 0.0 | Discourage frequency |

---

## 🎓 Key Takeaways

1. ✅ **Temperature** is your main "creativity knob"
2. ✅ **Top-p** provides dynamic filtering based on probability
3. ✅ **Top-k** provides fixed filtering based on count
4. ✅ Start with `temperature=0.7, top_p=0.95` for most tasks
5. ✅ Use `temperature=0.0` for deterministic, reproducible outputs
6. ✅ Penalties help reduce repetition in longer generations
7. ✅ Always experiment to find what works for your use case

---

## 🔗 Further Reading

- [vLLM Sampling Parameters API](https://docs.vllm.ai/en/latest/api/inference_params.html#sampling-parameters)
- [The Illustrated GPT-2 (Sampling)](http://jalammar.github.io/illustrated-gpt2/)
- [How to generate text: using different decoding methods](https://huggingface.co/blog/how-to-generate)

---

Ready to put this knowledge into practice? Return to [Lesson 1](../lesson-01-basic-inference/README.md) or move to [Lesson 2](../lesson-02-text-generation/README.md)!
