#!/bin/bash
# Script to start vLLM with Docker

echo "🚀 Starting vLLM ROCm container..."

# Create models directory if it doesn't exist
mkdir -p models

# Pull the latest image
echo "Pulling latest vLLM ROCm image..."
docker pull rocm/vllm-dev:nightly

# Start the container
docker compose up -d

echo "✅ Container started!"
echo ""
echo "To access the container, run:"
echo "  docker exec -it vllm-rocm bash"
echo ""
echo "To view logs:"
echo "  docker compose logs -f"
echo ""
echo "To stop the container:"
echo "  docker compose down"
