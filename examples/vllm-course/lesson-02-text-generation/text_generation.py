"""
Lesson 2: Text Generation with Command-Line Parameters

This script demonstrates how to build a flexible, production-ready vLLM script that:
1. Accepts command-line arguments for configuration
2. Uses EngineArgs for comprehensive engine configuration
3. Allows dynamic sampling parameter adjustment
4. Provides helpful documentation via --help

This approach makes your scripts reusable without code modifications.

Based on: https://github.com/vllm-project/vllm/blob/main/examples/offline_inference/basic/generate.py
"""

from vllm import LLM, EngineArgs
from vllm.utils.argparse_utils import FlexibleArgumentParser


def create_parser():
    """
    Create an argument parser with both engine and sampling parameters.
    
    Returns:
        FlexibleArgumentParser: Configured argument parser
    """
    parser = FlexibleArgumentParser(
        description="vLLM Text Generation with Configurable Parameters"
    )
    
    # ========================================
    # ENGINE ARGUMENTS
    # ========================================
    # This single line adds ~50+ arguments for model configuration!
    # Includes: model path, quantization, GPU settings, performance tuning, etc.
    EngineArgs.add_cli_args(parser)
    
    # Set sensible defaults for our use case
    parser.set_defaults(
        model="facebook/opt-125m",  # Small model for learning
    )
    
    # ========================================
    # SAMPLING PARAMETERS
    # ========================================
    # Create a separate argument group for better --help organization
    sampling_group = parser.add_argument_group(
        "Sampling parameters",
        "Control text generation behavior and creativity"
    )
    
    sampling_group.add_argument(
        "--max-tokens",
        type=int,
        help="Maximum number of tokens to generate (default: use model default)"
    )
    
    sampling_group.add_argument(
        "--temperature",
        type=float,
        help="Sampling temperature (0.0 = deterministic, higher = more random)"
    )
    
    sampling_group.add_argument(
        "--top-p",
        type=float,
        help="Nucleus sampling threshold (0.0-1.0)"
    )
    
    sampling_group.add_argument(
        "--top-k",
        type=int,
        help="Top-k sampling (number of top tokens to consider)"
    )
    
    return parser


def print_configuration(args: dict, sampling_params):
    """Print the current configuration for transparency."""
    print("=" * 80)
    print("CONFIGURATION")
    print("=" * 80)
    print(f"\nEngine Settings:")
    print(f"  Model: {args.get('model', 'N/A')}")
    print(f"  Tokenizer: {args.get('tokenizer', 'Same as model')}")
    print(f"  GPU Memory Utilization: {args.get('gpu_memory_utilization', 0.9)}")
    
    print(f"\nSampling Parameters:")
    print(f"  Temperature: {sampling_params.temperature}")
    print(f"  Top-p: {sampling_params.top_p}")
    print(f"  Top-k: {sampling_params.top_k}")
    print(f"  Max tokens: {sampling_params.max_tokens}")
    print()


def main(args: dict):
    """
    Main function for text generation.
    
    Args:
        args: Dictionary of parsed command-line arguments
    """
    
    # ========================================
    # STEP 1: Extract Sampling Parameters
    # ========================================
    # These are NOT engine arguments, so we remove them from args dict
    # Using .pop() removes them and returns their value (or None if not provided)
    max_tokens = args.pop("max_tokens")
    temperature = args.pop("temperature")
    top_p = args.pop("top_p")
    top_k = args.pop("top_k")
    
    # ========================================
    # STEP 2: Create the LLM
    # ========================================
    # All remaining args are valid engine arguments
    # This includes model, quantization, GPU settings, etc.
    print("Initializing LLM engine...")
    llm = LLM(**args)
    print("✓ Engine initialized successfully!\n")
    
    # ========================================
    # STEP 3: Configure Sampling Parameters
    # ========================================
    # Start with model's default sampling parameters
    sampling_params = llm.get_default_sampling_params()
    
    # Override only the parameters provided by the user
    # This allows users to change just what they need
    if max_tokens is not None:
        sampling_params.max_tokens = max_tokens
    if temperature is not None:
        sampling_params.temperature = temperature
    if top_p is not None:
        sampling_params.top_p = top_p
    if top_k is not None:
        sampling_params.top_k = top_k
    
    # Display the final configuration
    print_configuration(args, sampling_params)
    
    # ========================================
    # STEP 4: Prepare Prompts
    # ========================================
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    
    print(f"Prompts to complete: {len(prompts)}")
    print()
    
    # ========================================
    # STEP 5: Generate Text
    # ========================================
    print("Generating text...")
    outputs = llm.generate(prompts, sampling_params)
    print(f"✓ Generated {len(outputs)} completions!\n")
    
    # ========================================
    # STEP 6: Display Results
    # ========================================
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    print()
    
    for i, output in enumerate(outputs, 1):
        prompt = output.prompt
        generated_text = output.outputs[0].text
        
        print(f"[{i}/{len(outputs)}]")
        print(f"Prompt: {prompt!r}")
        print(f"Output: {generated_text!r}")
        print("-" * 80)
        print()


if __name__ == "__main__":
    # ========================================
    # PARSE COMMAND-LINE ARGUMENTS
    # ========================================
    parser = create_parser()
    
    # Parse arguments and convert to dictionary
    # This allows us to manipulate them easily (e.g., pop sampling params)
    args: dict = vars(parser.parse_args())
    
    # Run the main function
    main(args)
