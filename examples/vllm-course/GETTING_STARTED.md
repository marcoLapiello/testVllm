# Getting Started with vLLM Course

Welcome! This guide will help you start your vLLM learning journey.

## 📋 Prerequisites Check

Before starting, ensure you have:

### Software Requirements
- [ ] Python 3.8+ installed
- [ ] pip package manager
- [ ] 16GB+ RAM (8GB minimum for tiny models)
- [ ] Git (optional, for cloning repositories)

### Hardware Recommendations
- **GPU**: NVIDIA GPU with CUDA support (optional but recommended)
- **CPU-only**: Works fine for learning with small models
- **Storage**: 10GB+ free space for models

### Knowledge Prerequisites
- [ ] Basic Python (variables, functions, loops, dictionaries)
- [ ] Command-line basics (cd, ls, running scripts)
- [ ] (Optional) Basic understanding of machine learning concepts

## 🚀 Installation

### Step 1: Install vLLM

```bash
# Install vLLM (this may take a few minutes)
pip install vllm

# Verify installation
python -c "import vllm; print(vllm.__version__)"
```

**Troubleshooting**:
- If installation fails, try: `pip install --upgrade pip`
- For CPU-only: vLLM works but will be slower
- Check [official docs](https://docs.vllm.ai/en/latest/getting_started/installation.html) for advanced options

### Step 2: Install Optional Dependencies

```bash
# For downloading models
pip install huggingface_hub

# For better progress bars (optional)
pip install tqdm
```

### Step 3: Test Your Setup

Run this simple test:

```python
# test_installation.py
from vllm import LLM, SamplingParams

print("Loading model...")
llm = LLM(model="facebook/opt-125m")

print("Generating text...")
output = llm.generate(["Hello, world!"], SamplingParams(temperature=0.8))

print("Result:", output[0].outputs[0].text)
print("\n✓ vLLM is working!")
```

```bash
python test_installation.py
```

**Expected**: Should download opt-125m (~250MB) and generate text.

## 📚 Course Structure

### Learning Path

```
START HERE
    ↓
Lesson 1: Basic Inference (1-2 hours)
├── Understand LLM class
├── Learn sampling parameters
└── Generate simple text
    ↓
Lesson 2: Parameters & CLI (2-3 hours)
├── Command-line arguments
├── Engine configuration
└── Flexible scripts
    ↓
Lesson 3: Chat Interface (2-3 hours)
├── Conversational AI
├── Message roles
└── Multi-turn dialogue
    ↓
Lesson 4: Embeddings (1-2 hours)
├── Vector representations
├── Semantic search
└── Similarity calculations
    ↓
Lesson 5: Classification (1-2 hours)
├── Text classification
├── Probability distributions
└── Real-world applications
```

**Total estimated time**: 8-12 hours

### How to Progress

1. **Read the lesson README** - Understand concepts first
2. **Study the code** - Read through the Python script
3. **Run the examples** - Execute and observe outputs
4. **Experiment** - Modify parameters and prompts
5. **Do exercises** - Practice makes perfect
6. **Review concepts/** - Dive deeper when needed

## 🎯 Recommended Schedule

### For Complete Beginners (2 weeks)
- **Week 1**: Lessons 1-2, experiment thoroughly
- **Week 2**: Lessons 3-5, build a small project

### For Experienced Developers (3-5 days)
- **Day 1**: Lessons 1-2
- **Day 2**: Lesson 3 + exercises
- **Days 3-4**: Lessons 4-5 + project
- **Day 5**: Review and build something useful

### Weekend Crash Course
- **Saturday Morning**: Lessons 1-2
- **Saturday Afternoon**: Lesson 3
- **Sunday**: Lessons 4-5 + quick project

## 📖 Using This Course

### Folder Structure
```
vllm-course/
├── README.md                    # Course overview
├── GETTING_STARTED.md          # This file
├── QUICK_REFERENCE.md          # Cheat sheet
├── concepts/                    # Deep dives into topics
├── lesson-01-basic-inference/  # Start here!
├── lesson-02-text-generation/
├── lesson-03-chat-interface/
├── lesson-04-embeddings/
└── lesson-05-classification/
```

### Each Lesson Contains
- **README.md** - Concept explanations and examples
- **[lesson].py** - Working code with comments
- **exercises.md** - Practice problems

## 💡 Learning Tips

### 1. Start Small
```python
# Begin with tiny models
llm = LLM(model="facebook/opt-125m")  # 125M parameters

# Graduate to larger ones
# llm = LLM(model="facebook/opt-1.3b")  # 1.3B parameters
```

### 2. Experiment Freely
```python
# Try different temperatures
for temp in [0.0, 0.5, 1.0, 1.5]:
    outputs = llm.generate(["Once upon a time"], 
                          SamplingParams(temperature=temp))
    print(f"Temp {temp}: {outputs[0].outputs[0].text}\n")
```

### 3. Use the REPL
```bash
# Interactive Python is great for learning
python
>>> from vllm import LLM, SamplingParams
>>> llm = LLM(model="facebook/opt-125m")
>>> # Experiment here!
```

### 4. Monitor Resources
```bash
# Watch GPU usage (if you have a GPU)
watch -n 1 nvidia-smi

# Check RAM usage
htop
```

### 5. Keep Notes
Document what you learn:
- What parameters work best for your use cases?
- What models perform well?
- What errors did you encounter and how did you fix them?

## 🐛 Common Issues

### Issue 1: Import Error
```
ModuleNotFoundError: No module named 'vllm'
```
**Solution**: `pip install vllm`

### Issue 2: CUDA Out of Memory
```
torch.cuda.OutOfMemoryError
```
**Solutions**:
- Use smaller model: `facebook/opt-125m`
- Reduce memory: `gpu_memory_utilization=0.7`
- Reduce context: `max_model_len=1024`

### Issue 3: Slow Download
```
Model downloading very slowly...
```
**Solutions**:
- Wait patiently (first time only)
- Use a VPN if HuggingFace is slow
- Pre-download: `huggingface-cli download model-name`

### Issue 4: No GPU Found
```
WARNING: No GPU found, using CPU
```
**Not a problem!** vLLM works on CPU, just slower. Use small models.

## 🔗 Resources

### Official Documentation
- [vLLM Docs](https://docs.vllm.ai/)
- [API Reference](https://docs.vllm.ai/en/latest/api/)
- [Supported Models](https://docs.vllm.ai/en/latest/models/supported_models.html)

### Community
- [GitHub Discussions](https://github.com/vllm-project/vllm/discussions)
- [GitHub Issues](https://github.com/vllm-project/vllm/issues)

### Related Learning
- [HuggingFace Transformers](https://huggingface.co/docs/transformers)
- [Understanding LLMs](https://ig.ft.com/generative-ai/)

## ✅ Ready to Start?

### Quick Start Checklist
- [ ] vLLM installed and tested
- [ ] Can run the test script successfully
- [ ] Understand the course structure
- [ ] Know where to get help

### Your First Steps

1. **Open Lesson 1**:
   ```bash
   cd vllm-course/lesson-01-basic-inference
   cat README.md  # Read the lesson
   ```

2. **Run the example**:
   ```bash
   python basic_inference.py
   ```

3. **Experiment**: Try changing temperature or prompts

4. **Do exercises**: Open `exercises.md` and try 1-3 exercises

5. **Move forward**: When comfortable, proceed to Lesson 2

## 🎓 What You'll Build

By the end of this course, you'll be able to:
- ✅ Run any vLLM-supported model
- ✅ Configure inference for your needs
- ✅ Build conversational AI applications
- ✅ Generate embeddings for semantic tasks
- ✅ Implement text classification
- ✅ Optimize performance for production
- ✅ Troubleshoot common issues

## 🚀 Let's Begin!

**Head to [Lesson 1: Basic Offline Inference](./lesson-01-basic-inference/README.md)**

---

*Happy learning! Remember: experimentation is key. Don't be afraid to break things – that's how you learn!*

## ❓ Questions?

- Check the [Quick Reference](./QUICK_REFERENCE.md) for common patterns
- Review [concepts/](./concepts/) for deep dives
- Consult [official vLLM docs](https://docs.vllm.ai/)
- Search [GitHub discussions](https://github.com/vllm-project/vllm/discussions)

**Now go build something amazing! 🎉**
