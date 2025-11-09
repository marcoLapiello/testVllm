"""
Example of running vLLM as an OpenAI-compatible API server

Usage:
    python api_server.py

Then test with:
    curl http://localhost:8000/v1/completions \
        -H "Content-Type: application/json" \
        -d '{
            "model": "facebook/opt-125m",
            "prompt": "Hello, my name is",
            "max_tokens": 50,
            "temperature": 0.8
        }'
"""

# Note: This is typically run via command line:
# python -m vllm.entrypoints.openai.api_server \
#     --model facebook/opt-125m \
#     --host 0.0.0.0 \
#     --port 8000

print("""
To run the vLLM API server, use the following command:

python -m vllm.entrypoints.openai.api_server \\
    --model facebook/opt-125m \\
    --host 0.0.0.0 \\
    --port 8000 \\
    --dtype auto \\
    --max-model-len 2048

Then test with:
curl http://localhost:8000/v1/completions \\
    -H "Content-Type: application/json" \\
    -d '{
        "model": "facebook/opt-125m",
        "prompt": "Hello, my name is",
        "max_tokens": 50,
        "temperature": 0.8
    }'

Or test chat completion:
curl http://localhost:8000/v1/chat/completions \\
    -H "Content-Type: application/json" \\
    -d '{
        "model": "facebook/opt-125m",
        "messages": [
            {"role": "user", "content": "Hello, how are you?"}
        ],
        "max_tokens": 50
    }'
""")
