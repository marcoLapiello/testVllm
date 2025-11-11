# Lesson 1: Exercises

Practice what you've learned with these hands-on exercises! Start with the easier ones and work your way up.

## 🟢 Beginner Exercises

### Exercise 1.1: Modify Prompts
**Task**: Change the prompts in `basic_inference.py` to complete these sentences:
- "Once upon a time"
- "The best way to learn programming is"
- "In the year 2030"

**Goal**: Get comfortable editing and running the script.

### Exercise 1.2: Adjust Temperature
**Task**: Run the script three times with different temperatures:
1. `temperature=0.0` (deterministic)
2. `temperature=0.5` (balanced)
3. `temperature=1.5` (very creative)

**Question**: How does the output change? Which temperature produces the most sensible results?

### Exercise 1.3: Change Max Tokens
**Task**: Modify `max_tokens` to generate:
- Very short completions (10 tokens)
- Medium completions (50 tokens)
- Long completions (200 tokens)

**Question**: What happens when you set max_tokens very high? Does the model always use all tokens?

## 🟡 Intermediate Exercises

### Exercise 1.4: Experiment with top_p
**Task**: Keep `temperature=0.8` constant and try different `top_p` values:
- `top_p=0.3` (very focused)
- `top_p=0.7` (moderately focused)
- `top_p=0.99` (very diverse)

**Question**: How does top_p affect the creativity and coherence of outputs?

### Exercise 1.5: Process Multiple Prompts Efficiently
**Task**: Create a list of 10 different prompts and generate completions for all of them.

**Challenge**: Measure how long it takes. Then try generating them one at a time in a loop. Which is faster and why?

```python
import time

# Batch processing
start = time.time()
outputs = llm.generate(prompts, sampling_params)
batch_time = time.time() - start

print(f"Batch processing took: {batch_time:.2f} seconds")
```

### Exercise 1.6: Extract Additional Information
**Task**: For each generated output, print:
- The number of tokens generated
- The finish reason
- The cumulative log probability

**Hint**: Look at the "BONUS" section in `basic_inference.py`.

## 🔴 Advanced Exercises

### Exercise 1.7: Compare Different Models
**Task**: Run the same prompts with different models and compare results:
- `facebook/opt-125m` (125M parameters - fast)
- `facebook/opt-350m` (350M parameters - medium)
- `facebook/opt-1.3b` (1.3B parameters - slower but better)

**Question**: How does model size affect:
- Quality of outputs?
- Speed of generation?
- Memory usage?

**Note**: Larger models require more VRAM/RAM.

### Exercise 1.8: Create a Custom Use Case
**Task**: Pick a real-world use case and implement it:
- **Story continuation**: Given a story opening, continue the narrative
- **Code completion**: Complete Python function signatures
- **Email writing**: Draft professional emails from bullet points

**Example**:
```python
prompts = [
    "def calculate_fibonacci(n):",
    "Subject: Meeting Request\n\nDear Team,",
    "# TODO: Add error handling\n",
]
```

### Exercise 1.9: Handle Edge Cases
**Task**: What happens when you:
- Use an empty prompt `""`?
- Use a very long prompt (1000+ words)?
- Request 0 max_tokens?
- Use temperature=0 and run multiple times?

**Goal**: Understand the boundaries and behavior of the system.

### Exercise 1.10: Build a Simple CLI Tool
**Task**: Create a command-line tool that:
1. Accepts a prompt from the user
2. Generates a completion
3. Asks if the user wants to try again

**Starter code**:
```python
from vllm import LLM, SamplingParams

def interactive_generation():
    llm = LLM(model="facebook/opt-125m")
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
    
    while True:
        prompt = input("\nEnter your prompt (or 'quit' to exit): ")
        if prompt.lower() == 'quit':
            break
        
        # Your code here
        outputs = llm.generate([prompt], sampling_params)
        print(f"Completion: {outputs[0].outputs[0].text}")

if __name__ == "__main__":
    interactive_generation()
```

## 🎯 Challenge Project

### Project: Smart Sentence Completer
**Task**: Build a tool that:
1. Reads incomplete sentences from a file
2. Generates completions with multiple temperature settings
3. Saves all results to an output file with formatting
4. Calculates and displays statistics (average tokens, completion times)

**Skills practiced**:
- File I/O
- Batch processing
- Parameter experimentation
- Result analysis

## ✅ Solutions

Solutions for these exercises can be found in `solutions/` (to be created), but try solving them yourself first!

## 💡 Tips for Success

1. **Experiment freely**: You can't break anything by changing parameters
2. **Read error messages**: They usually tell you exactly what's wrong
3. **Start small**: Use tiny models and short prompts while learning
4. **Take notes**: Document what you observe for different parameter combinations
5. **Have fun**: The best learning happens through exploration!

## 📊 Self-Assessment

Before moving to Lesson 2, make sure you can:
- [ ] Load a model with the LLM class
- [ ] Configure basic sampling parameters
- [ ] Generate text from single and multiple prompts
- [ ] Access and interpret output information
- [ ] Explain what temperature and top_p do
- [ ] Troubleshoot common errors

## ➡️ Ready for More?

Once you've completed at least exercises 1.1-1.6, you're ready for [Lesson 2: Text Generation with Parameters](../lesson-02-text-generation/README.md)!
