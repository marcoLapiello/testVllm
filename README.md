# vLLM on ROCm - Quick Start Guide

This project provides a complete setup for running vLLM (Very Large Language Models) on AMD GPUs with ROCm support.

## 🖥️ System Information

**Verified Configuration:**
- OS: Linux (Ubuntu)
- Python: 3.12.3
- GPU: AMD Radeon RX 7900 XTX (gfx1100)
- ROCm: Installed and working
- vLLM: 0.11.1rc6 (via Docker)

## 📁 Project Structure

```
testVllm/
├── README.md                 # This file
├── .gitignore               # Git ignore rules
├── docker-compose.yml       # Docker container configuration
├── models/                  # Model storage (created automatically)
├── examples/               # Example scripts
│   ├── simple_inference.py    # Basic text generation
│   ├── batch_inference.py     # Batch processing example
│   └── api_server.py         # API server instructions
└── scripts/                # Helper scripts
    ├── install_docker.sh     # Docker installation
    ├── start_docker.sh       # Start vLLM container
    └── install_native.sh     # Native installation (alternative)
```

## 🚀 Quick Start (Docker - Recommended)

### 1. Start the Container

```bash
bash scripts/start_docker.sh
```

This will:
- Pull the latest vLLM ROCm image
- Create a `models/` directory for caching
- Start the container with GPU access

### 2. Access the Container

```bash
sudo docker exec -it vllm-rocm bash
```

### 3. Run Example Scripts

Inside the container:

```bash
# Simple inference example
cd /app/examples
python simple_inference.py

# Batch inference
python batch_inference.py
```

### 4. Run as API Server

Inside the container:

```bash
python -m vllm.entrypoints.openai.api_server \
    --model facebook/opt-125m \
    --host 0.0.0.0 \
    --port 8000
```

Then test from your host machine:

```bash
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "facebook/opt-125m",
        "prompt": "Hello, my name is",
        "max_tokens": 50,
        "temperature": 0.8
    }'
```

## 🔧 Container Management

```bash
# View container logs
sudo docker compose logs -f

# Stop the container
sudo docker compose down

# Restart the container
sudo docker compose restart

# Check container status
sudo docker ps
```

## 📦 Using Different Models

### Option 1: Download Inside Container

```bash
sudo docker exec -it vllm-rocm bash
python -c "from huggingface_hub import snapshot_download; snapshot_download('facebook/opt-125m')"
```

### Option 2: Mount Local Models

1. Download models to `./models/` on your host
2. They'll be automatically available at `/app/models/` in the container

### Popular Models to Try

```python
# Small models for testing
"facebook/opt-125m"          # 125M parameters
"facebook/opt-350m"          # 350M parameters

# Larger models (require more VRAM)
"meta-llama/Llama-2-7b-hf"   # 7B parameters
"mistralai/Mistral-7B-v0.1"  # 7B parameters
```

## 🎯 Example Code

### Basic Inference

```python
from vllm import LLM, SamplingParams

# Load model
llm = LLM(model="facebook/opt-125m")

# Configure sampling
sampling_params = SamplingParams(
    temperature=0.8,
    top_p=0.95,
    max_tokens=100
)

# Generate text
prompts = ["Hello, my name is", "The future of AI is"]
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(f"Prompt: {output.prompt}")
    print(f"Generated: {output.outputs[0].text}\n")
```

## 🔍 Troubleshooting

### Container Won't Start

```bash
# Check Docker is running
sudo systemctl status docker

# Check GPU is visible
rocminfo | grep gfx

# View container logs
sudo docker compose logs
```

### Out of Memory Errors

Reduce model size or adjust parameters:

```python
llm = LLM(
    model="facebook/opt-125m",
    max_model_len=2048,  # Reduce context length
    gpu_memory_utilization=0.8  # Use 80% of GPU memory
)
```

### Permission Issues

If you get permission errors with Docker:

```bash
# Add your user to docker group
sudo usermod -aG docker $USER

# Log out and log back in, then verify
groups | grep docker
```

## 📊 Performance Optimization

For your AMD Radeon RX 7900 XTX (gfx1100):

```python
llm = LLM(
    model="your-model",
    tensor_parallel_size=1,
    gpu_memory_utilization=0.9,
    max_model_len=4096,
    dtype="auto",
)
```

## 🔄 Alternative: Native Installation

If you prefer not to use Docker:

```bash
bash scripts/install_native.sh
```

This will:
- Create a Python virtual environment
- Install PyTorch for ROCm
- Build Triton and Flash Attention
- Compile vLLM from source for gfx1100

**Note:** Native installation takes 10-20 minutes.

## 📚 Resources

- [vLLM Documentation](https://docs.vllm.ai/)
- [vLLM ROCm Guide](https://docs.vllm.ai/en/latest/getting_started/amd-installation.html)
- [ROCm Documentation](https://rocm.docs.amd.com/)
- [Hugging Face Models](https://huggingface.co/models)

## 🤝 Contributing

Feel free to submit issues or pull requests!

## 📝 License

This project setup is provided as-is for educational purposes.

---

**System Specs:**
- GPU: AMD Radeon RX 7900 XTX (gfx1100)
- ROCm: 7.1
- vLLM: 0.11.1rc6
- Docker: 28.5.2
