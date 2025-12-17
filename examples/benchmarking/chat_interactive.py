"""
vLLM Interactive Chat Script for RDNA3
=======================================

Purpose:
    - Simple terminal-based chat interface for real-world interaction testing
    - Streamlined configuration with only essential parameters
    - Optimized for AMD Radeon RX 7900 XTX (RDNA3)
    
Features:
    - Interactive chat loop with conversation history
    - Real-time token-by-token streaming responses
    - Easy model switching
    - Pre-configured RDNA3 optimizations
    - AsyncLLM engine for true streaming support
    
Usage:
    python chat_interactive.py
    
Author: Generated for RDNA3 Testing
Date: 2024-12-17
"""

import os
import sys
import asyncio

# =============================================================================
# CRITICAL: RDNA3 Environment Setup (MUST be set BEFORE importing vLLM)
# =============================================================================
os.environ["ROCBLAS_USE_HIPBLASLT"] = "0"  # Force rocBLAS (avoid hipBLASLt RDNA3 issues)
os.environ["PYTORCH_TUNABLEOP_ENABLED"] = "0"  # Disable auto-tuning
os.environ["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = "1"  # Enable AOTriton
os.environ["VLLM_USE_TRITON_FLASH_ATTN"] = "1"  # Enable Flash Attention
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_LOGGING_LEVEL"] = "WARNING"  # Less verbose for chat

print("=" * 80)
print("RDNA3 OPTIMIZATIONS APPLIED")
print("=" * 80)
print("✓ rocBLAS forced (hipBLASLt disabled)")
print("✓ Flash Attention enabled")
print("✓ AOTriton enabled")
print("=" * 80)
print()

from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.sampling_params import RequestOutputKind
from vllm.v1.engine.async_llm import AsyncLLM
from typing import List, Dict


# =============================================================================
# SIMPLIFIED CONFIGURATION
# =============================================================================

class SimpleChatConfig:
    """
    Simplified configuration with only commonly used parameters.
    Pre-configured for RDNA3 optimization.
    """
    
    def __init__(self):
        # =====================================================================
        # MODEL CONFIGURATION
        # =====================================================================
        
        # Model path or HuggingFace ID
        self.model: str = "/app/models/hub/models--btbtyler09--Qwen3-Coder-30B-A3B-Instruct-gptq-4bit/snapshots/147d73a63bffec1c8f3cabc21ed4dbd2cf5b1cd1"
        # Examples:
        #   - "meta-llama/Llama-3.2-3B-Instruct"
        #   - "/path/to/local/model"
        #   - "/path/to/model.gguf" (for GGUF files)
        
        # Tokenizer (None = use same as model)
        self.tokenizer: str = None
        
        # Trust remote code (required for some models like Qwen, Phi, Gemma)
        self.trust_remote_code: bool = True
        
        # Load format (auto, gguf, safetensors, pt)
        self.load_format: str = "auto"
        
        # Data type (auto, float16, bfloat16)
        self.dtype: str = "auto"
        
        # =====================================================================
        # MEMORY CONFIGURATION
        # =====================================================================
        
        # GPU memory utilization (0.0-1.0)
        self.gpu_memory_utilization: float = 0.90
        # RDNA3 24GB: 0.90 is safe for most models
        
        # Maximum model length (tokens)
        self.max_model_len: int = 8192
        # For chat: 4096-8192 is usually sufficient
        # Lower values = less memory, more concurrent users
        
        # KV cache dtype (auto, fp8_e4m3, fp8_e5m2)
        self.kv_cache_dtype: str = "fp8_e4m3"
        # fp8_e4m3: 50% memory savings vs FP16, minimal quality loss
        
        # =====================================================================
        # PERFORMANCE CONFIGURATION
        # =====================================================================
        
        # Maximum concurrent sequences
        self.max_num_seqs: int = 64
        # Higher = more concurrent users, more memory
        # RDNA3: 64-256 typical range
        
        # Enable CUDA graphs (False for debugging)
        self.enforce_eager: bool = False
        # False = Use CUDA graphs (2-3x faster)
        # True = Eager mode (easier debugging)
        
        # =====================================================================
        # QUANTIZATION (if using quantized models)
        # =====================================================================
        
        # Quantization method (None, gptq, awq, gguf)
        self.quantization: str = None
        # None = Auto-detect
        # Set to "gguf" if using .gguf files
        
        # =====================================================================
        # SAMPLING DEFAULTS (can override per message)
        # =====================================================================
        
        self.temperature: float = 0.7
        self.top_p: float = 0.95
        self.max_tokens: int = 4096  # Increased for long-form responses (was 512)
        self.top_k: int = 50
        
        # CRITICAL: Prevent infinite repetition loops
        self.repetition_penalty: float = 1.1  # Penalize repeated tokens
        self.presence_penalty: float = 0.0    # Alternative penalty (use one or the other)
    
    def to_engine_args(self) -> AsyncEngineArgs:
        """Convert to AsyncEngineArgs for AsyncLLM initialization."""
        return AsyncEngineArgs(
            model=self.model,
            tokenizer=self.tokenizer,
            trust_remote_code=self.trust_remote_code,
            load_format=self.load_format,
            dtype=self.dtype,
            gpu_memory_utilization=self.gpu_memory_utilization,
            max_model_len=self.max_model_len,
            kv_cache_dtype=self.kv_cache_dtype,
            max_num_seqs=self.max_num_seqs,
            enforce_eager=self.enforce_eager,
            quantization=self.quantization,
        )
    
    def to_sampling_params(self) -> SamplingParams:
        """Convert to SamplingParams with DELTA mode for streaming."""
        return SamplingParams(
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            top_k=self.top_k,
            repetition_penalty=self.repetition_penalty,
            presence_penalty=self.presence_penalty,
            output_kind=RequestOutputKind.DELTA,  # Enable streaming mode
            stop=["User:", "System:", "\n\nUser:", "\n\nSystem:"],  # Stop at conversation markers
            skip_special_tokens=True,  # Skip EOS and other special tokens in output
        )
    
    def display(self):
        """Display configuration."""
        print("=" * 80)
        print("CONFIGURATION")
        print("=" * 80)
        print()
        print("MODEL:")
        print(f"  Model: {self.model}")
        print(f"  Tokenizer: {self.tokenizer or '(same as model)'}")
        print(f"  Trust remote code: {self.trust_remote_code}")
        print(f"  Load format: {self.load_format}")
        print(f"  Data type: {self.dtype}")
        print()
        print("MEMORY:")
        print(f"  GPU memory utilization: {self.gpu_memory_utilization}")
        print(f"  Max model length: {self.max_model_len} tokens")
        print(f"  KV cache dtype: {self.kv_cache_dtype}")
        print()
        print("PERFORMANCE:")
        print(f"  Max concurrent sequences: {self.max_num_seqs}")
        print(f"  CUDA graphs: {'Disabled (Eager mode)' if self.enforce_eager else 'Enabled'}")
        print()
        print("SAMPLING DEFAULTS:")
        print(f"  Temperature: {self.temperature}")
        print(f"  Top-p: {self.top_p}")
        print(f"  Max tokens: {self.max_tokens}")
        print(f"  Top-k: {self.top_k}")
        print(f"  Repetition penalty: {self.repetition_penalty}")
        print()
        print("=" * 80)
        print()


# =============================================================================
# CHAT INTERFACE
# =============================================================================

class ChatInterface:
    """
    Interactive terminal-based chat interface with streaming support.
    Maintains conversation history and handles user interaction.
    Uses AsyncLLM for real-time token-by-token streaming.
    """
    
    def __init__(self, llm: AsyncLLM, sampling_params: SamplingParams):
        self.llm = llm
        self.sampling_params = sampling_params
        self.conversation_history: List[Dict[str, str]] = []
        self.system_prompt = "You are a helpful AI assistant. Respond naturally and conversationally without meta-commentary about your responses or instructions."
        self.request_counter = 0
    
    def set_system_prompt(self, prompt: str):
        """Set the system prompt."""
        self.system_prompt = prompt
        print(f"✓ System prompt set to: {prompt}")
    
    def reset_conversation(self):
        """Clear conversation history."""
        self.conversation_history = []
        print("✓ Conversation history cleared")
    
    def build_conversation(self) -> List[Dict[str, str]]:
        """Build full conversation including system prompt."""
        conversation = [
            {"role": "system", "content": self.system_prompt}
        ]
        conversation.extend(self.conversation_history)
        return conversation
    
    async def generate_response_streaming(self, user_message: str) -> str:
        """Generate AI response to user message with streaming and built-in vLLM metrics."""
        # Add user message to history
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })
        
        # Build full conversation
        conversation = self.build_conversation()
        
        # Convert conversation to prompt using chat template
        # We'll use the engine's generate method directly
        # For simplicity, we'll manually format the conversation
        # (In production, you'd use the tokenizer's chat template)
        
        # Generate unique request ID
        self.request_counter += 1
        request_id = f"chat-{self.request_counter}"
        
        # Convert conversation to a single prompt
        # Note: This is a simplified approach. For better results with chat models,
        # you should use the model's proper chat template
        prompt_parts = []
        for msg in conversation:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
        
        prompt_parts.append("Assistant:")
        prompt = "\n\n".join(prompt_parts)
        
        # Stream the response and capture final output with metrics
        response_text = ""
        final_output = None
        async for output in self.llm.generate(
            request_id=request_id,
            prompt=prompt,
            sampling_params=self.sampling_params
        ):
            # Keep reference to the last (final) output for metrics
            final_output = output
            
            # In DELTA mode, we get only new tokens
            for completion in output.outputs:
                new_text = completion.text
                if new_text:
                    print(new_text, end="", flush=True)
                    response_text += new_text
        
        # Display vLLM's built-in performance metrics
        print("\n")
        print("─" * 80)
        print("📊 vLLM Performance Metrics:")
        
        if final_output and final_output.metrics:
            metrics = final_output.metrics
            
            # TTFT (Time To First Token) - from arrival to first token
            if hasattr(metrics, 'first_token_latency') and metrics.first_token_latency:
                print(f"   • TTFT (Time To First Token): {metrics.first_token_latency:.3f}s")
            
            # Tokens generated
            if hasattr(metrics, 'num_generation_tokens'):
                num_tokens = metrics.num_generation_tokens
                print(f"   • Tokens Generated: {num_tokens}")
                
                # Calculate tokens per second
                # Decode time is from first token to last token (excludes prefill)
                if hasattr(metrics, 'last_token_ts') and hasattr(metrics, 'first_token_ts'):
                    decode_time = metrics.last_token_ts - metrics.first_token_ts
                    if decode_time > 0 and num_tokens > 1:
                        # Exclude the first token from tokens/sec calculation
                        tokens_per_second = (num_tokens - 1) / decode_time
                        print(f"   • Generation Speed: {tokens_per_second:.2f} tokens/s")
                        print(f"   • Inter-Token Latency (avg): {(decode_time / (num_tokens - 1) * 1000):.2f}ms")
            
            # Queue time - time spent waiting in queue
            if hasattr(metrics, 'scheduled_ts') and hasattr(metrics, 'queued_ts'):
                queue_time = metrics.scheduled_ts - metrics.queued_ts
                print(f"   • Queue Time: {queue_time:.3f}s")
            
            # Prefill time - time to process prompt and generate first token
            if hasattr(metrics, 'first_token_ts') and hasattr(metrics, 'scheduled_ts'):
                prefill_time = metrics.first_token_ts - metrics.scheduled_ts
                print(f"   • Prefill Time: {prefill_time:.3f}s")
            
            # Decode time - time to generate all tokens after first
            if hasattr(metrics, 'last_token_ts') and hasattr(metrics, 'first_token_ts'):
                decode_time = metrics.last_token_ts - metrics.first_token_ts
                print(f"   • Decode Time: {decode_time:.3f}s")
            
            # Total inference time (prefill + decode)
            if hasattr(metrics, 'last_token_ts') and hasattr(metrics, 'scheduled_ts'):
                inference_time = metrics.last_token_ts - metrics.scheduled_ts
                print(f"   • Total Inference Time: {inference_time:.3f}s")
            
            # End-to-end latency (from arrival to last token)
            if hasattr(metrics, 'arrival_time') and hasattr(metrics, 'last_token_ts'):
                e2e_latency = metrics.last_token_ts - metrics.arrival_time
                print(f"   • End-to-End Latency: {e2e_latency:.3f}s")
        else:
            print("   • Metrics not available (may need to enable with log_stats=True)")
        
        print("─" * 80)
        
        # Add assistant response to history
        self.conversation_history.append({
            "role": "assistant",
            "content": response_text
        })
        
        return response_text
    
    def display_help(self):
        """Display help message."""
        print()
        print("=" * 80)
        print("CHAT COMMANDS")
        print("=" * 80)
        print()
        print("  /help          - Show this help message")
        print("  /reset         - Clear conversation history")
        print("  /system <text> - Set system prompt")
        print("  /history       - Show conversation history")
        print("  /quit or /exit - Exit the chat")
        print()
        print("  Just type your message to chat!")
        print()
        print("=" * 80)
        print()
    
    def display_history(self):
        """Display conversation history."""
        print()
        print("=" * 80)
        print("CONVERSATION HISTORY")
        print("=" * 80)
        print()
        print(f"System: {self.system_prompt}")
        print()
        for i, message in enumerate(self.conversation_history, 1):
            role = message["role"].capitalize()
            content = message["content"]
            print(f"{i}. {role}: {content}")
            print()
        print("=" * 80)
        print()
    
    async def run(self):
        """Run the interactive chat loop (async)."""
        print()
        print("╔" + "=" * 78 + "╗")
        print("║" + " " * 78 + "║")
        print("║" + "  vLLM INTERACTIVE CHAT (Streaming)".center(78) + "║")
        print("║" + "  Type /help for commands".center(78) + "║")
        print("║" + " " * 78 + "║")
        print("╚" + "=" * 78 + "╝")
        print()
        print(f"System prompt: {self.system_prompt}")
        print("Type your message and press Enter to chat.")
        print("Type /quit or /exit to end the conversation.")
        print()
        
        while True:
            try:
                # Get user input (using asyncio to avoid blocking)
                user_input = await asyncio.to_thread(input, "\n\033[1;34mYou:\033[0m ")
                user_input = user_input.strip()
                
                # Handle empty input
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.startswith("/"):
                    command_parts = user_input.split(maxsplit=1)
                    command = command_parts[0].lower()
                    
                    if command in ["/quit", "/exit"]:
                        print("\n✓ Goodbye!")
                        break
                    
                    elif command == "/help":
                        self.display_help()
                        continue
                    
                    elif command == "/reset":
                        self.reset_conversation()
                        continue
                    
                    elif command == "/history":
                        self.display_history()
                        continue
                    
                    elif command == "/system":
                        if len(command_parts) > 1:
                            self.set_system_prompt(command_parts[1])
                        else:
                            print("Usage: /system <system prompt text>")
                        continue
                    
                    else:
                        print(f"Unknown command: {command}")
                        print("Type /help for available commands")
                        continue
                
                # Generate and display response with streaming
                print("\n\033[1;32mAssistant:\033[0m ", end="", flush=True)
                await self.generate_response_streaming(user_input)
                print()  # New line after response
                
            except KeyboardInterrupt:
                print("\n\n✓ Interrupted. Type /quit to exit or continue chatting.")
                continue
            
            except Exception as e:
                print(f"\n\n❌ Error: {e}")
                print("Type /quit to exit or continue chatting.")
                continue


# =============================================================================
# MAIN
# =============================================================================

async def main():
    """Main execution (async)."""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 78 + "║")
    print("║" + "  vLLM INTERACTIVE CHAT - RDNA3 OPTIMIZED (STREAMING)".center(78) + "║")
    print("║" + " " * 78 + "║")
    print("╚" + "=" * 78 + "╝")
    print("\n")
    
    # Load configuration
    config = SimpleChatConfig()
    config.display()
    
    # Initialize AsyncLLM
    print("INITIALIZING MODEL...")
    print("(This may take a moment on first run)")
    print()
    
    try:
        # Create engine args and initialize AsyncLLM
        engine_args = config.to_engine_args()
        llm = AsyncLLM.from_engine_args(engine_args)
        print("✓ Model loaded successfully!")
        print()
        
        # Get sampling parameters
        sampling_params = config.to_sampling_params()
        
        # Start chat interface
        chat = ChatInterface(llm, sampling_params)
        await chat.run()
        
    except KeyboardInterrupt:
        print("\n\n✓ Interrupted during initialization. Exiting.")
        sys.exit(0)
    
    except Exception as e:
        print("=" * 80)
        print("❌ ERROR DURING INITIALIZATION")
        print("=" * 80)
        print()
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Message: {str(e)}")
        print()
        print("Troubleshooting:")
        print("  1. Check if model path is correct")
        print("  2. Verify GPU memory is sufficient")
        print("  3. Try with a smaller model")
        print("  4. Check RDNA3 environment variables")
        print()
        sys.exit(1)
    
    finally:
        # Cleanup
        try:
            if 'llm' in locals():
                # Shutdown AsyncLLM
                llm.shutdown()
                print("\n✓ Cleaned up resources")
        except Exception as cleanup_error:
            # Cleanup errors are usually harmless
            pass


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n✓ Exiting...")
        sys.exit(0)
