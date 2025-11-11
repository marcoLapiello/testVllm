#!/bin/bash
# Clear vLLM torch.compile cache to avoid corruption issues
# This is especially useful when:
# - Switching between different models
# - After interrupted runs
# - When seeing FXGraphCacheMiss errors

echo "🧹 Clearing vLLM torch.compile cache..."
sudo docker exec -it vllm-rocm bash -c "rm -rf /root/.cache/vllm/torch_compile_cache/* && echo '✅ Cache cleared successfully'"
