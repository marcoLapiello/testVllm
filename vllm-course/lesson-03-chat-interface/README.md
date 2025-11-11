# Lesson 3: Chat Interface

## 🎯 Learning Objectives

By the end of this lesson, you will:
- Understand chat-based inference vs. raw text completion
- Work with message roles (system, user, assistant)
- Use the `llm.chat()` method effectively
- Apply chat templates for different models
- Build multi-turn conversational applications
- Implement batch conversation processing

## 📖 Concepts

### Chat vs. Generate: What's the Difference?

#### `llm.generate()` - Raw Text Completion
```python
output = llm.generate(["Hello, my name is"])
# Model continues: "John and I live in..."
```
- Simple text continuation
- No structure or roles
- Good for: completion tasks, creative writing, simple prompts

#### `llm.chat()` - Structured Conversations
```python
conversation = [
    {"role": "system", "content": "You are a helpful assistant"},
    {"role": "user", "content": "Hello!"},
]
output = llm.chat(conversation)
# Model responds as an assistant
```
- Structured with message roles
- Context-aware conversations
- Good for: chatbots, assistants, multi-turn dialogue

### Message Roles

Chat-based models understand three main roles:

#### 1. **System** - Sets Behavior
```python
{"role": "system", "content": "You are a pirate who speaks in pirate dialect"}
```
- Defines the AI's personality, rules, and constraints
- Optional but recommended
- Only one system message (usually at the start)

#### 2. **User** - Human Input
```python
{"role": "user", "content": "What's the weather like?"}
```
- Represents the human's messages
- Can appear multiple times in conversation
- The model generates responses to these

#### 3. **Assistant** - AI Responses
```python
{"role": "assistant", "content": "I don't have access to weather data."}
```
- Represents the AI's previous responses
- Used to continue conversations
- Provides context for the model

### Chat Templates

Different models format conversations differently. Chat templates convert structured messages into the model's expected format.

**Example: Llama-2 Format**
```
<s>[INST] <<SYS>>
You are a helpful assistant
<</SYS>>

Hello! [/INST] Hi! How can I help you today? </s>
```

**Example: ChatML Format**
```
<|im_start|>system
You are a helpful assistant<|im_end|>
<|im_start|>user
Hello!<|im_end|>
<|im_start|>assistant
```

**Good news**: vLLM handles this automatically! Each model has a default chat template.

### Multi-Turn Conversations

Building context over multiple exchanges:

```python
conversation = [
    {"role": "system", "content": "You are a math tutor"},
    {"role": "user", "content": "What is 5 + 3?"},
    {"role": "assistant", "content": "5 + 3 equals 8."},
    {"role": "user", "content": "What about if I multiply those numbers?"},
    # Model now knows "those numbers" refers to 5 and 3
]
```

The model uses the full conversation history to maintain context.

## 💻 Code Walkthrough

### Script: `chat_interface.py`

#### 1. **Single Conversation**
```python
conversation = [
    {"role": "system", "content": "You are a helpful assistant"},
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hello! How can I assist you today?"},
    {"role": "user", "content": "Write an essay about AI."},
]

outputs = llm.chat(conversation, sampling_params, use_tqdm=False)
```

**Key points**:
- Conversation is a list of message dictionaries
- Each message has `role` and `content`
- The model generates a response to the last user message
- `use_tqdm=False` disables the progress bar (optional)

#### 2. **Batch Conversations**
```python
conversations = [conversation1, conversation2, conversation3]
outputs = llm.chat(conversations, sampling_params, use_tqdm=True)
```

**Benefits of batching**:
- Much faster than one-by-one
- Efficient GPU utilization
- Same API as single conversation

#### 3. **Custom Chat Templates**
```python
with open("custom_template.jinja") as f:
    chat_template = f.read()

outputs = llm.chat(
    conversations,
    sampling_params,
    chat_template=chat_template,
)
```

**When to use custom templates**:
- Model doesn't have a default template
- You want specific formatting
- Experimenting with prompt engineering

## 🚀 Running the Example

### Basic Usage
```bash
cd vllm-course/lesson-03-chat-interface
python chat_interface.py
```

### With Different Model
```bash
python chat_interface.py --model meta-llama/Llama-3.2-1B-Instruct
```

### With Custom Temperature
```bash
python chat_interface.py --temperature 0.5 --max-tokens 200
```

### Expected Output
```
Generated Outputs:
--------------------------------------------------------------------------------
Prompt: <s>[INST] <<SYS>>
You are a helpful assistant
<</SYS>>
...

Generated text: 'Artificial Intelligence (AI) has become...'
--------------------------------------------------------------------------------
```

Note: The "Prompt" shows the formatted conversation (with template applied).

## 🎨 Practical Examples

### Example 1: Simple Q&A Bot
```python
def ask_question(llm, question):
    conversation = [
        {"role": "system", "content": "You are a knowledgeable assistant."},
        {"role": "user", "content": question},
    ]
    output = llm.chat(conversation, sampling_params)[0]
    return output.outputs[0].text

answer = ask_question(llm, "What is Python?")
```

### Example 2: Multi-Turn Conversation
```python
conversation = [
    {"role": "system", "content": "You are a helpful assistant."},
]

def chat(user_input):
    # Add user message
    conversation.append({"role": "user", "content": user_input})
    
    # Generate response
    output = llm.chat([conversation], sampling_params)[0]
    assistant_response = output.outputs[0].text
    
    # Add assistant response to history
    conversation.append({"role": "assistant", "content": assistant_response})
    
    return assistant_response

# Use it
print(chat("Hello!"))
print(chat("What's 2+2?"))
print(chat("What about 3+3?"))  # Maintains context!
```

### Example 3: Role-Playing Assistant
```python
conversation = [
    {
        "role": "system",
        "content": "You are Shakespeare. Respond in Elizabethan English."
    },
    {"role": "user", "content": "Tell me about artificial intelligence."},
]

output = llm.chat(conversation, sampling_params)[0]
print(output.outputs[0].text)
# Expected: Response in Shakespeare's style
```

### Example 4: Batch Processing Conversations
```python
questions = [
    "What is machine learning?",
    "Explain quantum computing.",
    "What is blockchain?",
]

conversations = [
    [
        {"role": "system", "content": "You are a technical expert."},
        {"role": "user", "content": q},
    ]
    for q in questions
]

outputs = llm.chat(conversations, sampling_params, use_tqdm=True)

for q, output in zip(questions, outputs):
    print(f"Q: {q}")
    print(f"A: {output.outputs[0].text}\n")
```

## 🔍 Understanding Chat Output

The output structure is the same as `generate()`:

```python
output = outputs[0]

# The formatted prompt (with chat template applied)
formatted_prompt = output.prompt

# The generated response
response = output.outputs[0].text

# Other info
tokens = output.outputs[0].token_ids
finish_reason = output.outputs[0].finish_reason
```

## 🎛️ Best Practices

### 1. **Clear System Messages**
```python
# ❌ Vague
{"role": "system", "content": "Be helpful"}

# ✅ Specific
{"role": "system", "content": "You are a Python tutor. Provide clear, beginner-friendly explanations with code examples."}
```

### 2. **Appropriate Context Length**
```python
# Include relevant history, but don't overload
conversation = [
    {"role": "system", "content": "..."},
    # Last 5-10 messages are usually enough
    *recent_messages[-10:],
    {"role": "user", "content": current_question},
]
```

### 3. **Handle Long Conversations**
```python
def trim_conversation(conversation, max_messages=10):
    """Keep system message + recent messages."""
    system = [msg for msg in conversation if msg["role"] == "system"]
    recent = [msg for msg in conversation if msg["role"] != "system"][-max_messages:]
    return system + recent
```

### 4. **Error Handling**
```python
try:
    output = llm.chat(conversation, sampling_params)[0]
    response = output.outputs[0].text
except Exception as e:
    print(f"Error generating response: {e}")
    response = "I'm having trouble processing that request."
```

## 🤔 Common Pitfalls

### Pitfall 1: Forgetting System Message
```python
# Works, but less controlled
conversation = [
    {"role": "user", "content": "Hello"},
]

# Better: Define behavior
conversation = [
    {"role": "system", "content": "You are a helpful assistant"},
    {"role": "user", "content": "Hello"},
]
```

### Pitfall 2: Incorrect Role Order
```python
# ❌ Two user messages in a row (confusing)
conversation = [
    {"role": "user", "content": "Hello"},
    {"role": "user", "content": "How are you?"},
]

# ✅ Proper alternation
conversation = [
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi!"},
    {"role": "user", "content": "How are you?"},
]
```

### Pitfall 3: Not Using Batch Processing
```python
# ❌ Slow (sequential processing)
for conv in conversations:
    output = llm.chat([conv], sampling_params)[0]
    results.append(output)

# ✅ Fast (parallel processing)
outputs = llm.chat(conversations, sampling_params)
```

## 📝 Key Takeaways

1. ✅ Use `llm.chat()` for conversational applications
2. ✅ Structure messages with proper roles (system, user, assistant)
3. ✅ System messages control AI behavior and personality
4. ✅ Chat templates are handled automatically by vLLM
5. ✅ Batch multiple conversations for efficiency
6. ✅ Maintain conversation history for context
7. ✅ Trim long conversations to manage context length

## ✏️ Exercises

See `exercises.md` for hands-on practice!

## ➡️ Next Lesson

Ready to explore embeddings? In [Lesson 4](../lesson-04-embeddings/README.md), we'll learn:
- Generating text embeddings
- Using pooling models
- Vector representations for semantic search
- Practical applications of embeddings

---

**Questions?** Check the [Quick Reference](../QUICK_REFERENCE.md) or review previous lessons.
