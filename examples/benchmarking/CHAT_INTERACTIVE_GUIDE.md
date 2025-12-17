# vLLM Interactive Chat Guide

## Overview

`chat_interactive.py` is a simplified, user-friendly script for interactive chat testing with vLLM on RDNA3 GPUs. It focuses on real-world conversational scenarios with a streamlined configuration.

## Features

- 🎯 **Simplified Configuration**: Only essential parameters, pre-configured for RDNA3
- 💬 **Interactive Chat**: Terminal-based conversation with persistent history
- 🔄 **Real-time Responses**: Immediate model responses in chat format
- 🎨 **Colored Output**: Easy-to-read terminal interface
- 📝 **Conversation Management**: Reset, view history, customize system prompt
- ⚡ **RDNA3 Optimized**: Pre-configured environment variables for best performance

## Quick Start

### 1. Basic Usage

```bash
python chat_interactive.py
```

The script will:
1. Apply RDNA3 optimizations automatically
2. Display the configuration
3. Load the model
4. Start the interactive chat

### 2. Chat Commands

Once in the chat interface, you can use these commands:

| Command | Description |
|---------|-------------|
| `/help` | Show help message |
| `/reset` | Clear conversation history |
| `/system <text>` | Set a custom system prompt |
| `/history` | View full conversation history |
| `/quit` or `/exit` | Exit the chat |

### 3. Example Chat Session

```
You: Hello! Who are you?