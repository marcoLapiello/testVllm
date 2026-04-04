from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="x")
history = [{"role": "system", "content": "You are a helpful assistant."}]

print("Chat with Qwen3.5-35B-GPTQ (type 'quit' to exit)\n")

while True:
    user_input = input("You: ").strip()
    if user_input.lower() in ("quit", "exit", "q"):
        break
    if not user_input:
        continue

    history.append({"role": "user", "content": user_input})

    print("Assistant: ", end="", flush=True)
    stream = client.chat.completions.create(
        model="Qwen3.5-35B-GPTQ",
        messages=history,
        stream=True,
        temperature=0.7,
        max_tokens=4096,
    )

    reply = ""
    for chunk in stream:
        delta = chunk.choices[0].delta.content or ""
        print(delta, end="", flush=True)
        reply += delta

    print()
    history.append({"role": "assistant", "content": reply})
