# vLLM Course Map

A visual guide to navigating this course and understanding how concepts build on each other.

## 🗺️ Course Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         vLLM COURSE                             │
│                 From Zero to Production-Ready                   │
└─────────────────────────────────────────────────────────────────┘

                    📚 PREREQUISITES
                          │
            ┌─────────────┼─────────────┐
            │             │             │
         Python      Command Line    Basic ML
        (required)    (required)     (helpful)


                    🏁 START HERE
                          │
                          ▼
        ┌──────────────────────────────────────┐
        │   Lesson 1: Basic Inference          │
        │   • LLM class                        │
        │   • SamplingParams                   │
        │   • Simple generation                │
        │   ⏱️  1-2 hours                      │
        └──────────────────────────────────────┘
                          │
                          ▼
        ┌──────────────────────────────────────┐
        │   Lesson 2: Parameters & CLI         │
        │   • FlexibleArgumentParser           │
        │   • EngineArgs                       │
        │   • Dynamic configuration            │
        │   ⏱️  2-3 hours                      │
        └──────────────────────────────────────┘
                          │
                          ▼
        ┌──────────────────────────────────────┐
        │   Lesson 3: Chat Interface           │
        │   • Message roles                    │
        │   • llm.chat() method                │
        │   • Multi-turn conversations         │
        │   ⏱️  2-3 hours                      │
        └──────────────────────────────────────┘
                          │
                ┌─────────┴─────────┐
                ▼                   ▼
    ┌────────────────────┐  ┌────────────────────┐
    │ Lesson 4:          │  │ Lesson 5:          │
    │ Embeddings         │  │ Classification     │
    │ • Vector reps      │  │ • Text classes     │
    │ • Semantic search  │  │ • Probabilities    │
    │ ⏱️  1-2 hours      │  │ ⏱️  1-2 hours      │
    └────────────────────┘  └────────────────────┘
                │                   │
                └─────────┬─────────┘
                          ▼
              🎯 BUILD YOUR PROJECT!
```

## 📊 Skill Progression

```
Beginner          Intermediate        Advanced
───────────────────────────────────────────────
Lesson 1          Lesson 2            Lessons 4-5
  ↓                 ↓                    ↓
Basic             Flexible            Specialized
Concepts        Configuration        Applications
```

## 🎓 Learning Objectives by Lesson

### Lesson 1: Foundation
```
Input:  No vLLM knowledge
Output: Can run basic inference

Skills Acquired:
✓ Load models
✓ Generate text
✓ Understand sampling
✓ Process outputs
```

### Lesson 2: Flexibility
```
Input:  Basic inference knowledge
Output: Production-ready scripts

Skills Acquired:
✓ CLI arguments
✓ Engine configuration
✓ Parameter tuning
✓ Batch processing
```

### Lesson 3: Conversations
```
Input:  Parameter knowledge
Output: Chatbot-ready

Skills Acquired:
✓ Chat formatting
✓ Role management
✓ Context handling
✓ Template usage
```

### Lesson 4: Embeddings
```
Input:  Core vLLM skills
Output: Vector search capability

Skills Acquired:
✓ Generate embeddings
✓ Semantic similarity
✓ Vector operations
✓ Search applications
```

### Lesson 5: Classification
```
Input:  Core vLLM skills
Output: Classification capability

Skills Acquired:
✓ Text classification
✓ Probability interpretation
✓ Zero-shot learning
✓ Category prediction
```

## 🛠️ Tools & Concepts by Lesson

```
┌─────────┬─────────────┬──────────────┬───────────────┐
│ Lesson  │ Main Class  │ Key Concept  │ Output Type   │
├─────────┼─────────────┼──────────────┼───────────────┤
│ 1       │ LLM         │ Sampling     │ Text          │
│ 2       │ EngineArgs  │ CLI Args     │ Text          │
│ 3       │ LLM.chat()  │ Roles        │ Text          │
│ 4       │ LLM.embed() │ Vectors      │ Embeddings    │
│ 5       │ LLM.classify│ Probability  │ Classes       │
└─────────┴─────────────┴──────────────┴───────────────┘
```

## 🎯 Project Ideas by Skill Level

### After Lesson 1
- [ ] Quote generator
- [ ] Story continuation tool
- [ ] Simple text completer

### After Lesson 2
- [ ] Batch document generator
- [ ] Parameter experiment tool
- [ ] Multi-model comparison script

### After Lesson 3
- [ ] Command-line chatbot
- [ ] Customer service simulator
- [ ] Educational tutor

### After Lesson 4
- [ ] Document similarity finder
- [ ] Semantic search engine
- [ ] Content recommendation system

### After Lesson 5
- [ ] Sentiment analyzer
- [ ] Topic classifier
- [ ] Content moderator

### Final Project (All Lessons)
- [ ] Multi-modal AI assistant
- [ ] Intelligent content platform
- [ ] Custom AI application

## 📈 Difficulty Curve

```
Difficulty
    ▲
    │                                    ┌───┐
5   │                               ┌───┤ 5 │
    │                          ┌────┤ 4 └───┘
4   │                     ┌────┤    └───┘
    │                ┌────┤ 3  │
3   │           ┌────┤    └────┘
    │      ┌────┤ 2  │
2   │ ┌────┤    └────┘
    │ │ 1  │
1   │ │    │
    └─┴────┴────┴────┴────┴────────────────►
      1    2    3    4    5        Lessons

Legend:
1 = Gentle introduction
2 = Building complexity  
3 = Core concepts
4-5 = Specialized applications
```

## 🧭 Navigation Guide

### Quick Access

| Need... | Go to... |
|---------|----------|
| **Start learning** | [Lesson 1 README](./lesson-01-basic-inference/README.md) |
| **Quick reference** | [QUICK_REFERENCE.md](./QUICK_REFERENCE.md) |
| **Installation help** | [GETTING_STARTED.md](./GETTING_STARTED.md) |
| **Deep dive on sampling** | [concepts/sampling-parameters.md](./concepts/sampling-parameters.md) |
| **Practice problems** | Any lesson's `exercises.md` |

### When You're Stuck

```
Problem?
   │
   ├─ Code not working? ──→ Check lesson README
   │
   ├─ Concept unclear? ───→ Read concepts/ docs
   │
   ├─ Need quick answer? ─→ Check QUICK_REFERENCE
   │
   └─ Still stuck? ───────→ Check official docs
```

## 📦 What's in Each Lesson Folder?

```
lesson-XX-topic/
│
├── README.md          ← Start here! Concepts explained
│                        • Theory
│                        • Examples  
│                        • Best practices
│
├── [script].py        ← Working code
│                        • Fully commented
│                        • Ready to run
│                        • Production patterns
│
└── exercises.md       ← Practice problems
                         • Beginner level
                         • Intermediate level
                         • Advanced level
```

## 🔄 Recommended Learning Flow

### Linear Path (Recommended for Beginners)
```
1 → 2 → 3 → 4 → 5 → Project
```
**Time**: ~10-12 hours

### Accelerated Path (For Experienced Devs)
```
1 → 2 → 3 → Pick (4 or 5) → Project
```
**Time**: ~6-8 hours

### Targeted Path (Specific Goal)
```
Know your goal?
├─ Chatbot? ────→ 1 → 2 → 3 → Build
├─ Search? ─────→ 1 → 2 → 4 → Build  
└─ Classifier? ─→ 1 → 2 → 5 → Build
```

### Deep Learning Path
```
Each lesson:
1. Read README completely
2. Study code in detail
3. Run examples
4. Complete ALL exercises
5. Build small project
6. Move to next lesson
```
**Time**: ~15-20 hours

## 🎯 Success Markers

### After Lesson 1
✓ "I can load a model and generate text"

### After Lesson 2  
✓ "I can configure vLLM for different use cases"

### After Lesson 3
✓ "I can build a conversational AI"

### After Lessons 4-5
✓ "I understand vLLM's full capabilities"

### Course Complete
✓ "I can build production vLLM applications"

## 📚 Supplementary Materials

### Core Documents
- `README.md` - Course overview
- `GETTING_STARTED.md` - Setup and installation
- `QUICK_REFERENCE.md` - Cheat sheet
- `COURSE_MAP.md` - This file!

### Concepts (Deep Dives)
- `concepts/sampling-parameters.md` - How generation works
- `concepts/engine-arguments.md` - (To be added)
- `concepts/model-quantization.md` - (To be added)

## 🏆 Completion Goals

By the end of this course, you should be able to:

```
☑ Explain core vLLM concepts
☑ Choose appropriate models for tasks
☑ Configure inference parameters
☑ Build conversational applications
☑ Generate and use embeddings
☑ Implement classification
☑ Optimize for production
☑ Troubleshoot common issues
☑ Read and understand vLLM docs
☑ Build your own AI applications
```

## 🚀 Ready to Start?

**Your learning journey begins here:**

👉 [GETTING_STARTED.md](./GETTING_STARTED.md) - Set up your environment

👉 [Lesson 1: Basic Inference](./lesson-01-basic-inference/README.md) - Your first vLLM script

---

**Remember**: Learning is not linear. Feel free to jump around, revisit lessons, and most importantly - **experiment!** 🎉
