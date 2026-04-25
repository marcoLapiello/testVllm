source ~/.venvs/vllm-rocm/bin/activate

huggingface-cli download btbtyler09/Qwen3.6-35B-A3B-GPTQ-4bit \
  --cache-dir /home/marcolap/Schreibtisch/testVllm/models/hub \
  --local-dir-use-symlinks True