# vLLM Learning Course

A hands-on, beginner-friendly course to master vLLM (Very Large Language Model) inference engine. This course follows the official vLLM examples and documentation, breaking down complex concepts into digestible lessons with practical code examples.

## 🎯 Course Objectives

By completing this course, you will:
- Understand the fundamentals of offline LLM inference with vLLM
- Learn how to configure and optimize model parameters
- Master different inference modes (generation, chat, embeddings, classification)
- Know how to work with quantized models for efficient deployment
- Build practical AI applications using vLLM

## 📚 Prerequisites

- Basic Python knowledge (functions, classes, dictionaries)
- Understanding of command-line interfaces
- Familiarity with machine learning concepts (optional but helpful)
- GPU with CUDA support (or willingness to use smaller models on CPU)

## 🛠️ Setup

### Installation
```bash
# Install vLLM
pip install vllm

# Optional: Install additional dependencies
pip install huggingface_hub
```

### Hardware Requirements
- **Minimum**: 8GB RAM, CPU (for small models like opt-125m)
- **Recommended**: 16GB+ VRAM GPU, 32GB+ RAM (for models up to 7B)
- **Optimal**: 24GB+ VRAM GPU (for models up to 13B)

## 📖 Course Structure

### **Lesson 1: Basic Offline Inference** 
📂 `lesson-01-basic-inference/`
- Introduction to the `LLM` class
- Loading your first model
- Simple text generation
- Understanding `SamplingParams`

### **Lesson 2: Text Generation with Parameters**
📂 `lesson-02-text-generation/`
- Configurable sampling parameters
- Temperature, top-p, top-k explained
- Using argument parsers for flexibility
- Batch inference basics

### **Lesson 3: Chat Interface**
📂 `lesson-03-chat-interface/`
- Conversational AI patterns
- Message roles (system, user, assistant)
- Chat templates
- Multi-turn conversations

### **Lesson 4: Embeddings**
📂 `lesson-04-embeddings/`
- Generating text embeddings
- Using pooling models
- Vector representations
- Practical applications (semantic search, clustering)

### **Lesson 5: Classification**
📂 `lesson-05-classification/`
- Text classification with vLLM
- Understanding class probabilities
- Zero-shot and few-shot classification
- Real-world use cases

## 📂 Repository Structure

```
vllm-course/
├── README.md                          # This file
├── concepts/                          # Core concepts explained
│   ├── sampling-parameters.md
│   ├── model-quantization.md
│   └── engine-arguments.md
├── lesson-01-basic-inference/
│   ├── README.md                      # Lesson guide
│   ├── basic_inference.py             # Working code
│   └── exercises.md                   # Practice problems
├── lesson-02-text-generation/
│   ├── README.md
│   ├── text_generation.py
│   └── exercises.md
└── ... (more lessons)
```

## 🚀 How to Use This Course

1. **Read each lesson's README** - Start with the conceptual overview
2. **Study the code** - Examine the Python scripts with inline comments
3. **Run the examples** - Execute the code and observe the outputs
4. **Modify and experiment** - Try different parameters and prompts
5. **Complete exercises** - Test your understanding with practice problems
6. **Review concepts/** - Dive deeper into specific topics as needed

## 🔗 Resources

- [Official vLLM Documentation](https://docs.vllm.ai/)
- [vLLM GitHub Repository](https://github.com/vllm-project/vllm)
- [Official Examples](https://github.com/vllm-project/vllm/tree/main/examples/offline_inference/basic)

## 📝 Learning Tips

- **Start small**: Begin with tiny models (opt-125m) before moving to larger ones
- **Experiment freely**: Change parameters to understand their effects
- **Read error messages**: vLLM provides helpful error messages
- **Check GPU usage**: Use `nvidia-smi` to monitor resource utilization
- **Take notes**: Document what you learn in your own words

## 🤝 Contributing

Feel free to add your own examples, exercises, or improvements to this course!

## 📄 License

This educational material is provided as-is for learning purposes. vLLM itself is licensed under Apache-2.0.

---

**Ready to start?** Head to [Lesson 1: Basic Offline Inference](./lesson-01-basic-inference/README.md) 🚀
