"""
Lesson 3: Chat Interface with vLLM

This script demonstrates how to use vLLM's chat API for conversational AI:
1. Structured conversations with message roles
2. Single and batch conversation processing
3. Multi-turn dialogue with context
4. Optional custom chat templates

The chat interface is ideal for building chatbots, virtual assistants,
and any application requiring conversational interactions.

Based on: https://github.com/vllm-project/vllm/blob/main/examples/offline_inference/basic/chat.py
"""

from vllm import LLM, EngineArgs
from vllm.utils.argparse_utils import FlexibleArgumentParser


def create_parser():
    """Create argument parser with engine and sampling parameters."""
    parser = FlexibleArgumentParser(
        description="vLLM Chat Interface - Conversational AI"
    )
    
    # Add engine arguments
    EngineArgs.add_cli_args(parser)
    
    # Default to a chat-optimized model
    parser.set_defaults(
        model="meta-llama/Llama-3.2-1B-Instruct"
    )
    
    # Sampling parameters
    sampling_group = parser.add_argument_group("Sampling parameters")
    sampling_group.add_argument("--max-tokens", type=int)
    sampling_group.add_argument("--temperature", type=float)
    sampling_group.add_argument("--top-p", type=float)
    sampling_group.add_argument("--top-k", type=int)
    
    # Chat-specific arguments
    parser.add_argument(
        "--chat-template-path",
        type=str,
        help="Path to custom chat template file (Jinja2 format)"
    )
    
    return parser


def print_outputs(outputs, title="Generated Outputs"):
    """Pretty print the generated outputs."""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    print()
    
    for i, output in enumerate(outputs, 1):
        prompt = output.prompt
        generated_text = output.outputs[0].text
        
        print(f"[{i}/{len(outputs)}]")
        print(f"Formatted Prompt:\n{prompt[:200]}...")  # Show first 200 chars
        print()
        print(f"Generated Response:\n{generated_text}")
        print("-" * 80)
        print()


def main(args: dict):
    """Main function for chat-based inference."""
    
    # Extract sampling parameters
    max_tokens = args.pop("max_tokens")
    temperature = args.pop("temperature")
    top_p = args.pop("top_p")
    top_k = args.pop("top_k")
    chat_template_path = args.pop("chat_template_path")
    
    # Create LLM
    print("Initializing chat-enabled LLM...")
    llm = LLM(**args)
    print("✓ LLM ready!\n")
    
    # Configure sampling parameters
    sampling_params = llm.get_default_sampling_params()
    if max_tokens is not None:
        sampling_params.max_tokens = max_tokens
    if temperature is not None:
        sampling_params.temperature = temperature
    if top_p is not None:
        sampling_params.top_p = top_p
    if top_k is not None:
        sampling_params.top_k = top_k
    
    print("Configuration:")
    print(f"  Temperature: {sampling_params.temperature}")
    print(f"  Top-p: {sampling_params.top_p}")
    print(f"  Max tokens: {sampling_params.max_tokens}")
    print()
    
    # ========================================
    # EXAMPLE 1: Single Conversation
    # ========================================
    print("=" * 80)
    print("EXAMPLE 1: Single Conversation")
    print("=" * 80)
    print()
    
    # A conversation is a list of messages with roles
    # Roles: "system", "user", "assistant"
    conversation = [
        # System message: Sets the AI's behavior and personality
        {
            "role": "system",
            "content": "You are a helpful assistant specializing in technology."
        },
        
        # User message: The human's input
        {
            "role": "user",
            "content": "Hello! What can you help me with?"
        },
        
        # Assistant message: Previous AI response (provides context)
        {
            "role": "assistant",
            "content": "Hello! I can help you with technology-related questions, "
                      "programming concepts, and tech recommendations."
        },
        
        # User message: The current query we want a response to
        {
            "role": "user",
            "content": "Great! Can you write a short essay about the importance "
                      "of artificial intelligence in education?"
        },
    ]
    
    print("Conversation structure:")
    for msg in conversation:
        print(f"  [{msg['role']}]: {msg['content'][:50]}...")
    print()
    
    print("Generating response...")
    outputs = llm.chat(conversation, sampling_params, use_tqdm=False)
    print_outputs(outputs, "Single Conversation Result")
    
    # ========================================
    # EXAMPLE 2: Batch Conversations
    # ========================================
    print("=" * 80)
    print("EXAMPLE 2: Batch Conversations (Multiple in Parallel)")
    print("=" * 80)
    print()
    
    # Create multiple conversations to process in parallel
    conversations = [
        [
            {"role": "system", "content": "You are a creative writer."},
            {"role": "user", "content": "Write a haiku about programming."},
        ],
        [
            {"role": "system", "content": "You are a historian."},
            {"role": "user", "content": "Summarize the Renaissance in 2 sentences."},
        ],
        [
            {"role": "system", "content": "You are a science teacher."},
            {"role": "user", "content": "Explain photosynthesis simply."},
        ],
    ]
    
    print(f"Processing {len(conversations)} conversations in parallel...")
    
    # Batch processing is much faster than sequential!
    # use_tqdm=True shows a progress bar
    outputs = llm.chat(conversations, sampling_params, use_tqdm=True)
    
    print_outputs(outputs, "Batch Conversation Results")
    
    # ========================================
    # EXAMPLE 3: Custom Chat Template (Optional)
    # ========================================
    if chat_template_path is not None:
        print("=" * 80)
        print("EXAMPLE 3: Custom Chat Template")
        print("=" * 80)
        print()
        
        with open(chat_template_path) as f:
            chat_template = f.read()
        
        print(f"Loaded custom template from: {chat_template_path}")
        print("Applying template to conversations...")
        
        outputs = llm.chat(
            conversations,
            sampling_params,
            use_tqdm=False,
            chat_template=chat_template,
        )
        
        print_outputs(outputs, "Custom Template Results")
    
    # ========================================
    # EXAMPLE 4: Interactive-Style Pattern
    # ========================================
    print("=" * 80)
    print("EXAMPLE 4: Building a Multi-Turn Conversation")
    print("=" * 80)
    print()
    
    # Start with system message
    multi_turn = [
        {"role": "system", "content": "You are a friendly math tutor."},
    ]
    
    # Simulate a conversation
    user_queries = [
        "What is 15 + 27?",
        "Can you show me how you got that answer?",
        "What if I multiply those two numbers instead?",
    ]
    
    for i, query in enumerate(user_queries, 1):
        print(f"Turn {i}:")
        print(f"  User: {query}")
        
        # Add user message
        multi_turn.append({"role": "user", "content": query})
        
        # Generate response
        output = llm.chat([multi_turn], sampling_params, use_tqdm=False)[0]
        assistant_response = output.outputs[0].text
        
        print(f"  Assistant: {assistant_response}")
        print()
        
        # Add assistant response to maintain context
        multi_turn.append({"role": "assistant", "content": assistant_response})
    
    print("✓ Multi-turn conversation completed!")
    print(f"  Total messages in conversation: {len(multi_turn)}")
    print()
    
    # ========================================
    # SUMMARY
    # ========================================
    print("=" * 80)
    print("KEY POINTS")
    print("=" * 80)
    print("""
1. Conversations are lists of message dictionaries with 'role' and 'content'
2. Three roles: 'system' (behavior), 'user' (input), 'assistant' (responses)
3. llm.chat() handles chat template formatting automatically
4. Batch processing multiple conversations is much faster
5. Maintain conversation history for multi-turn context
6. System messages define the AI's personality and constraints
    """)
    
    print("=" * 80)
    print("✓ All examples completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    parser = create_parser()
    args: dict = vars(parser.parse_args())
    main(args)
