# Finding Real GPTQ Quantized Models for Qwen3-Coder-30B

## Problem
The current model `cpatonn/Qwen3-Coder-30B-A3B-Instruct-GPTQ-4bit` uses **compressed-tensors** format (not real GPTQ), which is very slow on ROCm (51 t/s vs 130 t/s with GGUF).

## What to Look For
Real GPTQ models should have:
- `"quant_method": "gptq"` in config.json
- NOT `"format": "pack-quantized"` or `"quant_method": "compressed-tensors"`
- Usually quantized by: TheBloke, LoneStriker, or turboderp

## Recommended GPTQ Models to Try

### Option 1: Qwen2.5-Coder-32B-Instruct GPTQ
Search on HuggingFace for:
- `Qwen2.5-Coder-32B-Instruct GPTQ TheBloke`
- `Qwen2.5-Coder-32B-Instruct GPTQ LoneStriker`

### Option 2: Smaller Qwen Models with Real GPTQ
Known working models:
- `TheBloke/Qwen-14B-GPTQ` (older but real GPTQ)
- Check for `Qwen2-Coder-32B-Instruct-GPTQ` variants

### Option 3: Alternative Large Coding Models
If Qwen3/Qwen2.5 Coder 30B+ doesn't have real GPTQ:
- `TheBloke/CodeLlama-34B-Instruct-GPTQ`
- `TheBloke/deepseek-coder-33b-instruct-GPTQ`

## How to Verify Before Downloading

1. Go to model page on HuggingFace
2. Look at config.json (click "Files and versions")
3. Check for `"quant_method": "gptq"` (NOT compressed-tensors)
4. Download using: `huggingface-cli download MODEL_NAME --local-dir /path/to/models`

## Current Working Models
- ✅ `kaitchup/Qwen3-8B-autoround-4bit-gptq` - Real GPTQ (8B model)
- ❌ `cpatonn/Qwen3-Coder-30B-A3B-Instruct-GPTQ-4bit` - compressed-tensors (slow)
