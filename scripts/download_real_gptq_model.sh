#!/bin/bash
# Download real GPTQ version of Qwen3-Coder-30B

MODEL_CHOICE=${1:-1}

if [ "$MODEL_CHOICE" == "1" ]; then
    MODEL_NAME="btbtyler09/Qwen3-Coder-30B-A3B-Instruct-gptq-4bit"
    echo "📥 Downloading: $MODEL_NAME (group_size=32)"
elif [ "$MODEL_CHOICE" == "2" ]; then
    MODEL_NAME="pramjan/Qwen3-Coder-30B-A3B-Instruct-4bit-GPTQ"
    echo "📥 Downloading: $MODEL_NAME (group_size=128)"
else
    echo "Usage: $0 [1|2]"
    echo "  1: btbtyler09 (group_size=32, recommended)"
    echo "  2: pramjan (group_size=128, potentially faster)"
    exit 1
fi

MODELS_DIR="/home/marcolap/Schreibtisch/testVllm/models"

echo "Model: $MODEL_NAME"
echo "Destination: $MODELS_DIR"
echo ""
echo "This will download approximately 17-20 GB"
read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    exit 1
fi

# Download using huggingface-cli
sudo docker exec -it vllm-rocm bash -c "
    cd /app/models && \
    huggingface-cli download $MODEL_NAME \
        --local-dir $MODEL_NAME \
        --local-dir-use-symlinks False
"

echo ""
echo "✅ Download complete!"
echo ""
echo "To test the model, update your script to use:"
echo "  model=\"$MODEL_NAME\""
