"""
vLLM Engine Configuration Script for RDNA3 Architecture
========================================================

Purpose:
    - Comprehensive engine configuration for AMD Radeon RX 7900 XTX (RDNA3)
    - Manual parameter tuning for optimization and learning
    - Support for GGUF quantization format
    - RDNA3-specific optimizations (aotriton, attention kernels)
    
Target Hardware:
    - GPU: AMD Radeon RX 7900 XTX
    - Architecture: RDNA3
    - Environment: ROCm 7+ in Docker Container
    
Author: Generated for RDNA3 Optimization
Date: 2024-12-16
"""

import os

# =============================================================================
# CRITICAL: Set ROCBLAS_USE_HIPBLASLT=0 BEFORE importing vLLM/PyTorch
# =============================================================================
# hipBLASLt on RDNA3 (gfx1100) lacks tuning database and causes:
# - "Failed validator: GCN_ARCH_NAME" warnings
# - GPU underutilization (260W vs 330W)
# - Poor batch performance
# Setting this to 0 forces stable rocBLAS backend with proper gfx1100 support
os.environ["ROCBLAS_USE_HIPBLASLT"] = "0"
# Alternative: Disable tunableop entirely (more aggressive)
os.environ["PYTORCH_TUNABLEOP_ENABLED"] = "0"
print("=" * 80)
print("CRITICAL RDNA3 FIX:")
print("  ROCBLAS_USE_HIPBLASLT = 0 (force rocBLAS)")
print("  PYTORCH_TUNABLEOP_ENABLED = 0 (disable auto-tuning)")
print("This must be set BEFORE importing PyTorch/vLLM to avoid hipBLASLt issues")
print("=" * 80)
print()

from typing import Optional, Dict, Any
from vllm import LLM, SamplingParams


# =============================================================================
# PART 1: RDNA3-SPECIFIC ENVIRONMENT VARIABLES
# =============================================================================
# These MUST be set BEFORE importing or initializing vLLM engine
# They control low-level ROCm and attention kernel behavior

class RDNA3EnvironmentConfig:
    """
    Environment variables for RDNA3-specific optimizations.
    
    Key Issues Addressed:
    1. Triton Flash Attention not supported on RDNA3
    2. Aotriton (precompiled kernels) work better for RDNA3
    3. Various aiter (Attention Kernel Triton Ops) options to test
    """
    
    def __init__(self):
        # =================================================================
        # PYTORCH/ROCM AOTRITON CONFIGURATION (Critical for RDNA3)
        # =================================================================
        
        # Enable AOTriton (Ahead-Of-Time compiled Triton kernels for ROCm)
        self.TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL: int = 1  # TESTING: Enable for Flash Attention
        # Options: 0 (disabled), 1 (enabled)
        # RDNA3: Set to 1 - enables precompiled attention kernels
        # This is a PyTorch/ROCm level flag that must be set BEFORE vLLM flags
        # NOTE: Testing with Flash Attention to see if performance improves
        
        # =================================================================
        # FLASH ATTENTION CONFIGURATION (Critical for RDNA3)
        # =================================================================
        
        # Enable standard Triton Flash Attention for comparison
        self.VLLM_USE_TRITON_FLASH_ATTN: int = 1  # BATCH TEST: Enabled to compare with baseline
        # Options: 0 (disabled), 1 (enabled)
        # RDNA3: RDNA3 has custom autotune configs for Flash Attention  
        # Baseline (DISABLED): 30.96 tok/s with 49.52x concurrency available
        # Testing if Flash Attention improves batch throughput
        
        # =================================================================
        # AITER (ATTENTION KERNEL TRITON OPS) CONFIGURATION
        # =================================================================
        # Aiter provides precompiled attention kernels that work with RDNA3
        
        # Master switch for aiter operations
        self.VLLM_ROCM_USE_AITER: bool = False
        # Options: True, False
        # RDNA3: Set False for RMSNorm models (Mistral, Qwen3) due to type hints issue
        # Note: opt-125m works with True because it uses LayerNorm, not RMSNorm
        
        # Enable aiter for Multi-Head Attention
        self.VLLM_ROCM_USE_AITER_MHA: bool = False  # TEST: Disable to check if MHA aiter causes repetition
        # Options: True, False
        # RDNA3: Set False for RMSNorm models (Qwen3 MoE) for consistency
        
        # Enable paged attention using aiter kernels
        self.VLLM_ROCM_USE_AITER_PAGED_ATTN: bool = False
        # Options: True, False
        # RDNA3: Try True - may improve performance with aotriton
        
        # Enable Triton unified attention for V1 in ROCm
        self.VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION: bool = False
        # Options: True, False
        # RDNA3: Try True - experimental unified attention path
        
        # Enable aiter's Triton Rope kernel
        self.VLLM_ROCM_USE_TRITON_ROPE: bool = False
        # Options: True, False
        # RDNA3: Try True - may improve RoPE performance
        
        # =================================================================
        # ROCBLAS / HIPBLASLT CONFIGURATION (CRITICAL FOR RDNA3)
        # =================================================================
        
        # Disable hipBLASLt to avoid RDNA3 compatibility issues
        self.ROCBLAS_USE_HIPBLASLT: int = 0
        # Options: 0 (disabled), 1 (enabled)
        # CRITICAL for RDNA3: hipBLASLt lacks gfx1100 tuning database
        # Issues with default (1):
        #   - "Failed validator: GCN_ARCH_NAME" warnings
        #   - Falls back to slow generic GEMM kernels
        #   - Poor GPU utilization (260W instead of 330W)
        #   - Low performance (4% GPU busy during batch inference)
        # Setting to 0 forces stable rocBLAS backend
        # Expected: Better performance, no warnings, higher GPU utilization
        
        # =================================================================
        # OTHER ROCM-SPECIFIC SETTINGS
        # =================================================================
        
        # Enable custom paged attention kernel for MI3* cards
        self.VLLM_ROCM_CUSTOM_PAGED_ATTN: bool = True
        # Options: True, False
        # RDNA3: Keep True (default, optimized for newer ROCm cards)
        
        # =================================================================
        # GENERAL VLLM SETTINGS
        # =================================================================
        
        # Worker multiprocessing method
        self.VLLM_WORKER_MULTIPROC_METHOD: str = "spawn"
        # Options: "spawn", "fork", "forkserver"
        # Recommendation: "spawn" for stability
        
        # Logging level
        self.VLLM_LOGGING_LEVEL: str = "INFO"
        # Options: "DEBUG", "INFO", "WARNING", "ERROR"
        # Use DEBUG for detailed troubleshooting
    
    def apply(self):
        """Apply all environment variables to the current process."""
        print("=" * 80)
        print("APPLYING RDNA3-SPECIFIC ENVIRONMENT VARIABLES")
        print("=" * 80)
        
        # PyTorch/ROCm AOTriton (MUST BE SET FIRST)
        os.environ["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = str(self.TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL)
        print(f"✓ TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL = {self.TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL}")
        print("  → Enables precompiled attention kernels at PyTorch level")
        print()
        
        # Flash Attention
        os.environ["VLLM_USE_TRITON_FLASH_ATTN"] = str(self.VLLM_USE_TRITON_FLASH_ATTN)
        print(f"✓ VLLM_USE_TRITON_FLASH_ATTN = {self.VLLM_USE_TRITON_FLASH_ATTN}")
        
        # Aiter Configuration
        os.environ["VLLM_ROCM_USE_AITER"] = str(self.VLLM_ROCM_USE_AITER).lower()
        print(f"✓ VLLM_ROCM_USE_AITER = {self.VLLM_ROCM_USE_AITER}")
        
        os.environ["VLLM_ROCM_USE_AITER_MHA"] = str(self.VLLM_ROCM_USE_AITER_MHA).lower()
        print(f"✓ VLLM_ROCM_USE_AITER_MHA = {self.VLLM_ROCM_USE_AITER_MHA}")
        
        os.environ["VLLM_ROCM_USE_AITER_PAGED_ATTN"] = str(self.VLLM_ROCM_USE_AITER_PAGED_ATTN).lower()
        print(f"✓ VLLM_ROCM_USE_AITER_PAGED_ATTN = {self.VLLM_ROCM_USE_AITER_PAGED_ATTN}")
        
        os.environ["VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION"] = str(self.VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION).lower()
        print(f"✓ VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION = {self.VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION}")
        
        os.environ["VLLM_ROCM_USE_TRITON_ROPE"] = str(self.VLLM_ROCM_USE_TRITON_ROPE).lower()
        print(f"✓ VLLM_ROCM_USE_TRITON_ROPE = {self.VLLM_ROCM_USE_TRITON_ROPE}")
        
        # ROCBLAS / hipBLASLt Settings
        os.environ["ROCBLAS_USE_HIPBLASLT"] = str(self.ROCBLAS_USE_HIPBLASLT)
        print(f"✓ ROCBLAS_USE_HIPBLASLT = {self.ROCBLAS_USE_HIPBLASLT} (Force rocBLAS, avoid hipBLASLt RDNA3 issues)")
        
        # Other ROCm Settings
        os.environ["VLLM_ROCM_CUSTOM_PAGED_ATTN"] = str(self.VLLM_ROCM_CUSTOM_PAGED_ATTN).lower()
        print(f"✓ VLLM_ROCM_CUSTOM_PAGED_ATTN = {self.VLLM_ROCM_CUSTOM_PAGED_ATTN}")
        
        # General Settings
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = self.VLLM_WORKER_MULTIPROC_METHOD
        print(f"✓ VLLM_WORKER_MULTIPROC_METHOD = {self.VLLM_WORKER_MULTIPROC_METHOD}")
        
        os.environ["VLLM_LOGGING_LEVEL"] = self.VLLM_LOGGING_LEVEL
        print(f"✓ VLLM_LOGGING_LEVEL = {self.VLLM_LOGGING_LEVEL}")
        
        print("=" * 80)
        print()


# =============================================================================
# PART 2: ENGINE CONFIGURATION
# =============================================================================
# This section will be expanded with ALL vLLM engine parameters
# We're building this step-by-step

class EngineConfig:
    """
    Comprehensive vLLM engine configuration.
    
    All parameters are organized by category for easy learning and tuning.
    Each parameter includes:
    - Type annotation
    - Default value
    - Detailed description
    - Recommendations for RDNA3
    """
    
    def __init__(self):
        # =====================================================================
        # CATEGORY 1: MODEL LOADING & BASIC CONFIGURATION
        # =====================================================================
        
        # Model identifier or path
        self.model: str = "unsloth/gpt-oss-20b-GGUF"
        # Description: HuggingFace model ID OR path to local model OR path to local GGUF file
        # Examples: 
        #   - "meta-llama/Llama-3.2-3B-Instruct"
        #   - "/path/to/local/model"
        #   - "/path/to/model.gguf" (for local GGUF file)
        #   - Testing: GPT-OSS 20B with GGUF Q4_K_M quantization
        # RDNA3: Testing GPT-OSS 20B (original uses MXFP4 which is CUDA-only)
        # Using: unsloth/gpt-oss-20b-GGUF with Q4_K_M quantization
        # Note: Original openai/gpt-oss-20b is MXFP4 (doesn't work on ROCm)
        # Previous tests: 
        #   - Qwen3-Coder-30B GGUF Q4_K_M: 98 tok/s single, 31.17 tok/s batch (8 requests)
        #   - Qwen3-Coder-30B GPTQ-4bit: 137.72 tok/s batch (4.4x faster than GGUF!)
        #   - Qwen3-8B AutoRound GPTQ: 488.33 tok/s batch (best so far!)
        
        # Tokenizer
        self.tokenizer: Optional[str] = "unsloth/gpt-oss-20b-GGUF"
        # Description: Tokenizer name or path. If None, uses same as model
        # Example: "meta-llama/Llama-3.2-3B-Instruct"
        # Note: For GPTQ models, tokenizer is included in the model repo
        
        # HuggingFace config path (for GGUF models)
        self.hf_config_path: Optional[str] = "openai/gpt-oss-20b"
        # Description: Path to HuggingFace model config (CRITICAL for GGUF)
        # - For GGUF models, this loads the proper model architecture config
        # - Use the base model repo (not -GGUF repo)
        # - Fixes repetition issues with GGUF models (GitHub issue #24025)
        # - For GPT-OSS: Use openai/gpt-oss-20b as config source
        # Example: "Qwen/Qwen3-30B-A3B-Instruct-2507"
        
        # Tokenizer mode
        self.tokenizer_mode: str = "auto"
        # Options: "auto", "slow", "mistral"
        # Description: Tokenizer loading mode
        # - "auto": Automatic selection
        # - "slow": Use slow tokenizer (more compatible)
        # - "mistral": Mistral-specific tokenizer
        
        # Tokenizer revision
        self.tokenizer_revision: Optional[str] = None
        # Description: Specific tokenizer revision (branch/tag/commit) on HuggingFace
        # Example: "main", "v1.0", specific commit hash
        
        # Model revision
        self.revision: Optional[str] = None
        # Description: Model revision (branch/tag/commit) on HuggingFace
        # Example: "main", "v1.0", specific commit hash
        
        # Trust remote code
        self.trust_remote_code: bool = False
        # Description: Allow executing custom Python code from model repository
        # WARNING: Only enable for trusted models
        # Required for: Some custom architectures (e.g., Phi, Qwen, Gemma 3 multimodal)
        # Note: Gemma 3 multimodal has corrupted preprocessor_config.json in unsloth GGUF repo
        
        # Download directory
        self.download_dir: Optional[str] = None
        # Description: Directory for downloading/loading model weights
        # Default: Uses HuggingFace cache directory
        # Example: "/app/models"
        
        # Load format
        self.load_format: str = "gguf"
        # Options: "auto", "pt", "safetensors", "npcache", "dummy", "tensorizer", "bitsandbytes", "gguf"
        # Description: Model weight loading format
        # - "auto": Automatically detect format (use for GPTQ/AWQ)
        # - "gguf": For GGUF quantized models
        # - "pt": PyTorch .bin files
        # - "safetensors": Safetensors format (safer, faster)
        # - "tensorizer": Optimized loading with tensorizer
        # - "bitsandbytes": BitsAndBytes quantization
        # RDNA3: Using "gguf" for GPT-OSS 20B (Q4_K_M quantization)
        # Testing: GPT-OSS 20B GGUF vs Qwen3 GGUF performance
        
        # Config format
        self.config_format: str = "auto"
        # Options: "auto", "hf"
        # Description: Model config file format
        
        # Random seed
        self.seed: int = 0
        # Description: Random seed for reproducibility
        # Set to specific value for deterministic behavior
        
        # Data type
        self.dtype: str = "auto"
        # Options: "auto", "float16", "bfloat16", "float32"
        # Description: Model weight and activation data type
        # - "auto": Automatically select based on model config
        # - "float16": FP16 (good memory/speed balance)
        # - "bfloat16": BF16 (better numerical stability)
        # - "float32": FP32 (highest precision, most memory)
        # RDNA3 Recommendation: "float16" or "bfloat16"
        
        # =====================================================================
        # CATEGORY 2: MEMORY MANAGEMENT
        # =====================================================================
        
        # GPU memory utilization
        self.gpu_memory_utilization: float = 0.90
        # Range: 0.0 - 1.0
        # Description: Fraction of GPU memory to use for model weights
        # - Higher values: More KV cache, more sequences
        # - Lower values: More headroom, more stable
        # RDNA3 (24GB): Try 0.85-0.95 depending on model size
        
        # Maximum model length
        self.max_model_len: Optional[int] = 1024
        # Description: Maximum model context length (prompt + output)
        # - None: Auto-derived from model config
        # BATCH TEST: Set to 1024 (prompt ~12 tokens + 512 max_tokens = ~524 total)
        # This allows much higher concurrency than 8192
        # Previous: 8192 allowed 6.31x concurrency (wasted memory for 512 token responses)
        # Expected: 1024 should allow ~50x concurrency, perfect for 8 concurrent requests
        # - None: Auto-derived from model config
        # - Specify value: Override model's maximum
        # Example: 4096, 8192, 16384
        # Note: Higher values require more memory
        # RDNA3 24GB: Limited to ~8K-16K for 30B Q4 models due to KV cache requirements
        
        # Block size
        self.block_size: int = 16
        # Description: Token block size for contiguous memory (PagedAttention)
        # - Typically 16 or 32
        # - Affects KV cache granularity
        # - Usually keep default unless testing
        
        # Swap space
        self.swap_space: int = 4
        # Description: CPU swap space size in GiB for GPU-CPU memory swapping
        # - Allows handling sequences that don't fit in GPU memory
        # - Performance penalty when swapping occurs
        # RDNA3: 4-8 GiB recommended
        
        # CPU offloading
        self.cpu_offload_gb: float = 0
        # Description: GiB of model weights to offload to CPU per GPU
        # - Reduces GPU memory usage
        # - Significant performance penalty
        # - Use only if model doesn't fit in GPU memory
        # RDNA3 (24GB): Typically not needed for small/medium models
        
        # Enable prefix caching
        self.enable_prefix_caching: bool = False
        # Description: Enable automatic prefix caching (vLLM caching)
        # - Caches common prompt prefixes
        # - Speeds up repeated prefix patterns
        # - Uses additional memory
        # Use case: Chat applications with system prompts
        
        # Disable sliding window
        self.disable_sliding_window: bool = False
        # Description: Disable sliding window attention
        # - Some models use sliding window (e.g., Mistral)
        # - Disabling caps to model's max context length
        
        # KV cache data type
        self.kv_cache_dtype: str = "fp8_e4m3"
        # Options: "auto", "fp8", "fp8_e5m2", "fp8_e4m3"
        # Description: KV cache quantization data type
        # - "auto": Use same as model dtype (currently: FP16/BF16 = ~2.34 GiB)
        # - "fp8_e4m3": FP8 with 4-bit exponent, 3-bit mantissa (RECOMMENDED for memory savings)
        # - "fp8_e5m2": FP8 with 5-bit exponent, 2-bit mantissa (better range)
        # - Lower precision = 50% memory savings vs FP16
        # RDNA3: Testing "fp8_e4m3" to reduce KV cache from 2.34 GiB
        # Expected: ~1.17 GiB KV cache, minimal quality loss, potential speed improvement
        
        # =====================================================================
        # CATEGORY 3: PERFORMANCE & PARALLELISM
        # =====================================================================
        
        # Tensor parallelism size
        self.tensor_parallel_size: int = 1
        # Description: Number of GPUs to use for tensor parallelism
        # - Splits model weights across multiple GPUs
        # - Use when model doesn't fit on single GPU
        # - Must be divisible by number of attention heads
        # Example: 2, 4, 8
        # RDNA3 (single GPU): Keep at 1
        
        # Pipeline parallelism size
        self.pipeline_parallel_size: int = 1
        # Description: Number of pipeline stages
        # - Splits model layers across GPUs
        # - Can combine with tensor parallelism
        # - Total GPUs = tensor_parallel_size * pipeline_parallel_size
        # RDNA3 (single GPU): Keep at 1
        
        # Maximum number of sequences
        self.max_num_seqs: int = 256
        # Description: Maximum number of sequences per iteration
        # - Higher = more throughput, more memory
        # - Lower = less memory, lower throughput
        # - Typical range: 128-512
        # RDNA3: Start with 256, adjust based on model size
        
        # Maximum number of batched tokens
        self.max_num_batched_tokens: Optional[int] = None
        # Description: Maximum tokens to process in a single batch
        # - None: Auto-calculated based on max_num_seqs
        # - Manual: Set to control memory usage
        # - Higher = more throughput, more memory
        # RDNA3: Leave as None initially
        
        # Scheduler delay factor
        self.scheduler_delay_factor: float = 0.0
        # Range: 0.0 - 1.0
        # Description: Delay factor for scheduler
        # - 0.0: No delay (default)
        # - Higher: More delay, can improve batching
        # - Rarely needs tuning
        
        # Enable chunked prefill
        self.enable_chunked_prefill: Optional[bool] = None
        # Description: Enable chunked prefill for long prompts
        # - None: Auto-decide based on configuration
        # - True: Split long prefills into chunks
        # - False: Process entire prefill at once
        # - Helps with long context lengths
        # RDNA3: Try True for long contexts (>4K tokens)
        
        # Number of scheduler steps
        self.num_scheduler_steps: int = 1
        # Description: Number of scheduler steps per iteration
        # - Typically 1
        # - Multi-step scheduling is advanced feature
        
        # Max parallel loading workers
        self.max_parallel_loading_workers: Optional[int] = None
        # Description: Maximum workers for parallel model loading
        # - None: Use default
        # - Higher: Faster model loading, more CPU/memory
        # RDNA3: Leave as None
        
        # Enable expert parallelism (for MoE models)
        self.enable_expert_parallel: bool = False
        # Description: Enable expert parallelism for Mixture-of-Experts models
        # - False: Standard execution (default)
        # - True: Parallelize expert execution across GPUs
        # - Requires multiple GPUs or tensor parallelism
        # RDNA3 (single GPU): Keep False, but may improve throughput if using multiple GPUs
        
        # Expert placement strategy (for MoE models)
        self.expert_placement_strategy: str = "linear"
        # Options: "linear", "round_robin"
        # Description: How to place experts across devices in MoE models
        # - "linear": Place experts sequentially on devices
        # - "round_robin": Distribute experts evenly
        # - Only relevant when enable_expert_parallel=True
        # RDNA3: Keep "linear" unless using multiple GPUs
        
        # Number of redundant experts (for MoE models)
        self.num_redundant_experts: int = 0
        # Description: Number of redundant experts for MoE models
        # - 0: No redundancy (default)
        # - Higher: More fault tolerance, more memory
        # - Advanced feature for distributed MoE
        # RDNA3: Keep 0
        
        # Disable custom all-reduce
        self.disable_custom_all_reduce: bool = False
        # Description: Disable custom all-reduce kernel, use NCCL
        # - False: Use custom kernel (faster)
        # - True: Use NCCL (more compatible)
        # - Only relevant for multi-GPU
        # RDNA3 (single GPU): Leave as False
        
        # Enforce eager execution
        self.enforce_eager: bool = False
        # Description: Disable CUDA graphs, use eager execution
        # - False: Use CUDA graphs (faster, recommended)
        # - True: Use eager mode (debugging, compatibility)
        # - CUDA graphs = optimization for repeated operations
        # TESTING RESULT: CUDA graphs provide 2.4x speedup (98 vs 41 tok/s) - KEEP ENABLED!
        
        # Max context length to capture
        self.max_context_len_to_capture: Optional[int] = None
        # Description: Maximum context length for CUDA graph capture
        # - None: Use default (8192)
        # - Lower: Less memory for graphs, more flexibility
        # - Only affects CUDA graph optimization
        
        # =====================================================================
        # CATEGORY 4: QUANTIZATION
        # =====================================================================
        
        # Quantization method
        self.quantization: Optional[str] = None
        # Options: None, "awq", "gptq", "gptq_marlin", "gptq_bitblas", "gptq_marlin_24", 
        #          "squeezellm", "fp8", "gguf", "compressed-tensors", "bitsandbytes"
        # Description: Quantization method for model weights
        # - None: Auto-detect from model config (then kernel selected by env var or default)
        # - "gptq": Default GPTQ (buggy, shows warning)
        # - "gptq_marlin": Optimized Marlin GPTQ kernel (RECOMMENDED for GPTQ models)
        # - "gptq_bitblas": BitBLAS GPTQ kernel
        # - "awq": Activation-aware Weight Quantization
        # - "gguf": GGUF format (auto-detected for .gguf files)
        # RDNA3: Cannot override to gptq_marlin due to validation error
        # Use environment variable VLLM_GPTQ_MARLIN=1 or model must be saved as marlin format
        # Current: Using default (auto-detect), getting buggy gptq_gemm kernel warning
        # - "awq": Activation-aware Weight Quantization
        # - "gptq": GPTQ quantization
        # - "gguf": GGUF format (RECOMMENDED for RDNA3)
        # - "fp8": FP8 quantization
        # - "bitsandbytes": BitsAndBytes (4bit/8bit)
        # RDNA3: Use "gguf" for best performance with quantized models
        # NOTE: When using GGUF files, you can leave this as None and set load_format="gguf"
        
        # RoPE scaling
        self.rope_scaling: Optional[Dict[str, Any]] = None
        # Description: RoPE (Rotary Position Embedding) scaling configuration
        # - None: No scaling
        # - Dict: Custom scaling (for extended context)
        # Example: {"type": "linear", "factor": 2.0}
        # Advanced: Used for context length extension
        
        # RoPE theta
        self.rope_theta: Optional[float] = None
        # Description: RoPE theta parameter
        # - None: Use model default
        # - Custom value: Override model's theta
        # - Affects positional encoding
        # Advanced parameter, rarely needs changing
        
        # =====================================================================
        # CATEGORY 5: ADVANCED FEATURES
        # =====================================================================
        
        # Speculative decoding config
        self.speculative_config: Optional[Dict[str, Any]] = None
        # Description: Configuration for speculative decoding
        # - None: Disabled
        # - Dict: Enable with draft model
        # Example: {
        #     "method": "eagle",
        #     "model": "path/to/draft-model",
        #     "num_speculative_tokens": 5
        # }
        # Advanced: Speeds up generation with draft model
        
        # Multi-modal processor cache (GB)
        self.mm_processor_cache_gb: float = 2.0
        # Description: Multi-modal processor cache size in GB
        # - 0: Disable cache
        # - Higher: Better performance for vision/audio models
        # - Only relevant for multi-modal models
        # RDNA3: Keep default unless using multi-modal
        
        # Multi-modal processor cache type
        self.mm_processor_cache_type: str = "local"
        # Options: "local", "shm"
        # Description: Multi-modal cache type
        # - "local": Local cache
        # - "shm": Shared memory (for tensor parallelism)
        # RDNA3 (single GPU): Keep "local"
        
        # Limit multi-modal per prompt
        self.limit_mm_per_prompt: Optional[Dict[str, int]] = None
        # Description: Limit multi-modal inputs per prompt
        # Example: {"image": 1, "audio": 1}
        # Only relevant for multi-modal models
        
        # Model loader extra config
        self.model_loader_extra_config: Optional[Dict[str, Any]] = None
        # Description: Extra configuration for model loader
        # - Tensorizer settings
        # - Custom loading parameters
        # Advanced use cases only
        
        # =====================================================================
        # CATEGORY 6: COMPILATION & OPTIMIZATION
        # =====================================================================
        
        # Compilation config
        self.compilation_config: Optional[Dict[str, Any]] = None
        # Description: PyTorch compilation configuration
        # - None: Use defaults
        # - Dict: Custom compilation settings
        # Example: {
        #     "mode": 3,  # Compilation level
        #     "cudagraph_mode": "FULL_AND_PIECEWISE"
        # }
        # Advanced: Controls torch.compile behavior
        
        # Distributed executor backend
        self.distributed_executor_backend: Optional[str] = None
        # Options: None, "ray", "mp" (multiprocessing)
        # Description: Backend for distributed execution
        # - None: Auto-select
        # - "ray": Ray for distribution
        # - "mp": Multiprocessing
        # RDNA3 (single GPU): Leave as None
        
        # Worker use Ray
        self.worker_use_ray: bool = False
        # Description: Use Ray for workers
        # - Only relevant for distributed setups
        # RDNA3 (single GPU): Keep False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary for LLM initialization."""
        config = {}
        
        # Category 1: Model Loading & Basic Configuration
        if self.model:
            config["model"] = self.model
        if self.tokenizer is not None:
            config["tokenizer"] = self.tokenizer
        if self.hf_config_path is not None:
            config["hf_config_path"] = self.hf_config_path
        if self.tokenizer_mode != "auto":
            config["tokenizer_mode"] = self.tokenizer_mode
        if self.tokenizer_revision is not None:
            config["tokenizer_revision"] = self.tokenizer_revision
        if self.revision is not None:
            config["revision"] = self.revision
        if self.trust_remote_code:
            config["trust_remote_code"] = self.trust_remote_code
        if self.download_dir is not None:
            config["download_dir"] = self.download_dir
        if self.load_format != "auto":
            config["load_format"] = self.load_format
        if self.seed != 0:
            config["seed"] = self.seed
        if self.dtype != "auto":
            config["dtype"] = self.dtype
        
        # Category 2: Memory Management
        config["gpu_memory_utilization"] = self.gpu_memory_utilization
        if self.max_model_len is not None:
            config["max_model_len"] = self.max_model_len
        if self.block_size != 16:
            config["block_size"] = self.block_size
        if self.swap_space != 4:
            config["swap_space"] = self.swap_space
        if self.cpu_offload_gb > 0:
            config["cpu_offload_gb"] = self.cpu_offload_gb
        if self.enable_prefix_caching:
            config["enable_prefix_caching"] = self.enable_prefix_caching
        if self.disable_sliding_window:
            config["disable_sliding_window"] = self.disable_sliding_window
        if self.kv_cache_dtype != "auto":
            config["kv_cache_dtype"] = self.kv_cache_dtype
        
        # Category 3: Performance & Parallelism
        if self.tensor_parallel_size > 1:
            config["tensor_parallel_size"] = self.tensor_parallel_size
        if self.pipeline_parallel_size > 1:
            config["pipeline_parallel_size"] = self.pipeline_parallel_size
        if self.max_num_seqs != 256:
            config["max_num_seqs"] = self.max_num_seqs
        if self.max_num_batched_tokens is not None:
            config["max_num_batched_tokens"] = self.max_num_batched_tokens
        if self.scheduler_delay_factor > 0:
            config["scheduler_delay_factor"] = self.scheduler_delay_factor
        if self.enable_chunked_prefill is not None:
            config["enable_chunked_prefill"] = self.enable_chunked_prefill
        if self.num_scheduler_steps != 1:
            config["num_scheduler_steps"] = self.num_scheduler_steps
        if self.max_parallel_loading_workers is not None:
            config["max_parallel_loading_workers"] = self.max_parallel_loading_workers
        if self.enable_expert_parallel:
            config["enable_expert_parallel"] = self.enable_expert_parallel
        if self.expert_placement_strategy != "linear":
            config["expert_placement_strategy"] = self.expert_placement_strategy
        if self.num_redundant_experts > 0:
            config["num_redundant_experts"] = self.num_redundant_experts
        if self.disable_custom_all_reduce:
            config["disable_custom_all_reduce"] = self.disable_custom_all_reduce
        if self.enforce_eager:
            config["enforce_eager"] = self.enforce_eager
        if self.max_context_len_to_capture is not None:
            config["max_context_len_to_capture"] = self.max_context_len_to_capture
        
        # Category 4: Quantization
        if self.quantization is not None:
            config["quantization"] = self.quantization
        if self.rope_scaling is not None:
            config["rope_scaling"] = self.rope_scaling
        if self.rope_theta is not None:
            config["rope_theta"] = self.rope_theta
        
        # Category 5: Advanced Features
        if self.speculative_config is not None:
            config["speculative_config"] = self.speculative_config
        if self.mm_processor_cache_gb != 2.0:
            config["mm_processor_cache_gb"] = self.mm_processor_cache_gb
        if self.mm_processor_cache_type != "local":
            config["mm_processor_cache_type"] = self.mm_processor_cache_type
        if self.limit_mm_per_prompt is not None:
            config["limit_mm_per_prompt"] = self.limit_mm_per_prompt
        if self.model_loader_extra_config is not None:
            config["model_loader_extra_config"] = self.model_loader_extra_config
        
        # Category 6: Compilation & Optimization
        if self.compilation_config is not None:
            config["compilation_config"] = self.compilation_config
        if self.distributed_executor_backend is not None:
            config["distributed_executor_backend"] = self.distributed_executor_backend
        if self.worker_use_ray:
            config["worker_use_ray"] = self.worker_use_ray
        
        return config
    
    def display(self):
        """Display current configuration in readable format."""
        print("=" * 80)
        print("VLLM ENGINE CONFIGURATION")
        print("=" * 80)
        print()
        
        print("CATEGORY 1: MODEL LOADING & BASIC CONFIGURATION")
        print("-" * 80)
        print(f"  model                : {self.model}")
        print(f"  tokenizer            : {self.tokenizer}")
        print(f"  hf_config_path       : {self.hf_config_path}")
        print(f"  tokenizer_mode       : {self.tokenizer_mode}")
        print(f"  tokenizer_revision   : {self.tokenizer_revision}")
        print(f"  revision             : {self.revision}")
        print(f"  trust_remote_code    : {self.trust_remote_code}")
        print(f"  download_dir         : {self.download_dir}")
        print(f"  load_format          : {self.load_format}")
        print(f"  config_format        : {self.config_format}")
        print(f"  seed                 : {self.seed}")
        print(f"  dtype                : {self.dtype}")
        print()
        
        print("CATEGORY 2: MEMORY MANAGEMENT")
        print("-" * 80)
        print(f"  gpu_memory_utilization : {self.gpu_memory_utilization}")
        print(f"  max_model_len          : {self.max_model_len}")
        print(f"  block_size             : {self.block_size}")
        print(f"  swap_space             : {self.swap_space} GiB")
        print(f"  cpu_offload_gb         : {self.cpu_offload_gb} GiB")
        print(f"  enable_prefix_caching  : {self.enable_prefix_caching}")
        print(f"  disable_sliding_window : {self.disable_sliding_window}")
        print(f"  kv_cache_dtype         : {self.kv_cache_dtype}")
        print()
        
        print("CATEGORY 3: PERFORMANCE & PARALLELISM")
        print("-" * 80)
        print(f"  tensor_parallel_size         : {self.tensor_parallel_size}")
        print(f"  pipeline_parallel_size       : {self.pipeline_parallel_size}")
        print(f"  max_num_seqs                 : {self.max_num_seqs}")
        print(f"  max_num_batched_tokens       : {self.max_num_batched_tokens}")
        print(f"  scheduler_delay_factor       : {self.scheduler_delay_factor}")
        print(f"  enable_chunked_prefill       : {self.enable_chunked_prefill}")
        print(f"  num_scheduler_steps          : {self.num_scheduler_steps}")
        print(f"  max_parallel_loading_workers : {self.max_parallel_loading_workers}")
        print(f"  enable_expert_parallel       : {self.enable_expert_parallel} (MoE)")
        print(f"  expert_placement_strategy    : {self.expert_placement_strategy} (MoE)")
        print(f"  num_redundant_experts        : {self.num_redundant_experts} (MoE)")
        print(f"  disable_custom_all_reduce    : {self.disable_custom_all_reduce}")
        print(f"  enforce_eager                : {self.enforce_eager}")
        print(f"  max_context_len_to_capture   : {self.max_context_len_to_capture}")
        print()
        
        print("CATEGORY 4: QUANTIZATION")
        print("-" * 80)
        print(f"  quantization         : {self.quantization}")
        print(f"  rope_scaling         : {self.rope_scaling}")
        print(f"  rope_theta           : {self.rope_theta}")
        print()
        
        print("CATEGORY 5: ADVANCED FEATURES")
        print("-" * 80)
        print(f"  speculative_config        : {self.speculative_config}")
        print(f"  mm_processor_cache_gb     : {self.mm_processor_cache_gb} GiB")
        print(f"  mm_processor_cache_type   : {self.mm_processor_cache_type}")
        print(f"  limit_mm_per_prompt       : {self.limit_mm_per_prompt}")
        print(f"  model_loader_extra_config : {self.model_loader_extra_config}")
        print()
        
        print("CATEGORY 6: COMPILATION & OPTIMIZATION")
        print("-" * 80)
        print(f"  compilation_config            : {self.compilation_config}")
        print(f"  distributed_executor_backend  : {self.distributed_executor_backend}")
        print(f"  worker_use_ray                : {self.worker_use_ray}")
        print()
        
        print("=" * 80)
        print()


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function."""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 78 + "║")
    print("║" + "  vLLM ENGINE CONFIGURATION & BENCHMARK SCRIPT".center(78) + "║")
    print("║" + "  Optimized for AMD Radeon RX 7900 XTX (RDNA3)".center(78) + "║")
    print("║" + " " * 78 + "║")
    print("╚" + "=" * 78 + "╝")
    print("\n")
    
    # Step 1: Configure and apply RDNA3-specific environment variables
    print("STEP 1: Configuring RDNA3-Specific Environment Variables")
    print("-" * 80)
    env_config = RDNA3EnvironmentConfig()
    env_config.apply()
    print()
    
    # Step 2: Configure engine parameters
    print("STEP 2: Configuring vLLM Engine Parameters")
    print("-" * 80)
    engine_config = EngineConfig()
    
    # Display current configuration
    engine_config.display()
    
    # Step 3: Initialize engine
    print("STEP 3: Engine Initialization")
    print("-" * 80)
    print(f"Initializing vLLM engine with model: {engine_config.model}")
    print("This may take a moment on first run (downloading/loading model)...")
    print()
    
    try:
        # Convert config to dict for LLM initialization
        llm_config = engine_config.to_dict()
        
        # Initialize the LLM engine
        print("Creating LLM instance...")
        llm = LLM(**llm_config)
        print("✓ Engine initialized successfully!")
        print()
        
        # Step 4: Test inference
        print("=" * 80)
        print("STEP 4: Test Inference")
        print("-" * 80)
        print()
        
        # Define sampling parameters - single long generation for realistic benchmarking
        sampling_params = SamplingParams(
            temperature=0.8,
            top_p=0.95,
            max_tokens=512,  # Shorter for batch testing
        )
        
        # Batch test prompts - testing Flash Attention benefits
        # Flash Attention shows improvements with concurrent requests
        batch_prompts = [
            "Explain how neural networks learn through backpropagation in simple terms.",
            "What are the key differences between supervised and unsupervised learning?",
            "Describe the architecture of a convolutional neural network (CNN).",
            "How does gradient descent optimization work in machine learning?",
            "Explain the concept of overfitting and how to prevent it.",
            "What is transfer learning and why is it useful in deep learning?",
            "Describe how attention mechanisms work in transformer models.",
            "What are the main challenges in training large language models?",
        ]
        
        batch_size = len(batch_prompts)
        
        print("Sampling Parameters:")
        print(f"  - Temperature: {sampling_params.temperature}")
        print(f"  - Top-p: {sampling_params.top_p}")
        print(f"  - Max tokens: {sampling_params.max_tokens}")
        print(f"  - Batch size: {batch_size} concurrent requests")
        print()
        
        print("Test Configuration:")
        print(f"  - Flash Attention: {'DISABLED' if os.environ.get('VLLM_USE_TRITON_FLASH_ATTN') == '0' else 'ENABLED'}")
        print(f"  - Testing: Batch throughput with {batch_size} requests")
        print()
        
        print("Batch prompts:")
        for i, prompt in enumerate(batch_prompts, 1):
            print(f"  {i}. {prompt}")
        print()
        
        print(f"Generating {batch_size} responses concurrently...")
        print("(This tests Flash Attention's ability to handle multiple requests)")
        print()
        
        import time
        start_time = time.time()
        
        # Generate - batch inference to test Flash Attention benefits
        outputs = llm.generate(batch_prompts, sampling_params)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Display results
        print("=" * 80)
        print("BATCH INFERENCE RESULTS")
        print("=" * 80)
        print()
        
        # Calculate aggregate metrics
        total_prompt_tokens = sum(len(output.prompt_token_ids) for output in outputs)
        total_generated_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
        
        print(f"BATCH PERFORMANCE METRICS:")
        print(f"  Total time: {total_time:.2f} seconds")
        print(f"  Batch size: {batch_size} requests")
        print(f"  Total prompt tokens: {total_prompt_tokens}")
        print(f"  Total generated tokens: {total_generated_tokens}")
        print(f"  Overall throughput: {total_generated_tokens / total_time:.2f} tokens/second")
        print(f"  Throughput per request: {(total_generated_tokens / total_time) / batch_size:.2f} tokens/second")
        print()
        
        # Show individual outputs (abbreviated)
        print("INDIVIDUAL OUTPUTS (first 100 chars each):")
        print("-" * 80)
        for i, output in enumerate(outputs, 1):
            prompt = output.prompt
            generated_text = output.outputs[0].text
            num_tokens = len(output.outputs[0].token_ids)
            
            print(f"\n{i}. Prompt: {prompt[:80]}...")
            print(f"   Tokens: {num_tokens}")
            print(f"   Response: {generated_text[:100]}...")
        print("-" * 80)
        print()
        
        # vLLM internal metrics (if available from first output)
        if hasattr(outputs[0], 'metrics') and outputs[0].metrics is not None:
            metrics = outputs[0].metrics
            print("VLLM INTERNAL METRICS (first request):")
            
            # Time to First Token (TTFT)
            if metrics.first_scheduled_time is not None and metrics.first_token_time is not None:
                ttft = metrics.first_token_time - metrics.first_scheduled_time
                print(f"  Time to First Token (TTFT): {ttft:.4f} seconds ({ttft*1000:.2f} ms)")
            
            # Time per Output Token (TPOT)
            first_output_tokens = len(outputs[0].outputs[0].token_ids)
            if metrics.first_token_time is not None and metrics.finished_time is not None:
                decode_time = metrics.finished_time - metrics.first_token_time
                if first_output_tokens > 1:
                    tpot = decode_time / (first_output_tokens - 1)  # Exclude first token
                    print(f"  Time per Output Token (TPOT): {tpot:.4f} seconds ({tpot*1000:.2f} ms/token)")
                    print(f"  Decode throughput: {(first_output_tokens - 1) / decode_time:.2f} tokens/second")
            
            # Total generation time (from vLLM perspective)
            if metrics.first_scheduled_time is not None and metrics.finished_time is not None:
                total_vllm_time = metrics.finished_time - metrics.first_scheduled_time
                print(f"  Total vLLM time: {total_vllm_time:.4f} seconds")
                throughput = first_output_tokens / total_vllm_time
                print(f"  vLLM throughput: {throughput:.2f} tokens/second")
            
            # Queue time
            if metrics.time_in_queue is not None:
                print(f"  Time in queue: {metrics.time_in_queue:.4f} seconds")
            
            # Model execution times
            if metrics.model_forward_time is not None:
                print(f"  Model forward time: {metrics.model_forward_time:.4f} seconds")
            if metrics.model_execute_time is not None:
                print(f"  Model execute time: {metrics.model_execute_time:.4f} seconds")
            
            print()
        else:
            print("Note: vLLM internal metrics not available")
            print()        
        print("=" * 80)
        print("✓ TEST COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print()
        print("Next Steps:")
        print("  1. Modify engine parameters in EngineConfig class")
        print("  2. Test with your own models (including GGUF)")
        print("  3. Adjust RDNA3 environment variables for optimization")
        print("  4. Run benchmarks with different configurations")
        print()
        print("Note: You may see a 'destroy_process_group() not called' warning after exit.")
        print("      This is a known cosmetic issue in vLLM V1 engine and can be safely ignored.")
        print("      It does not indicate memory leaks or incorrect behavior.")
        print()
        
    except Exception as e:
        print("=" * 80)
        print("❌ ERROR DURING ENGINE INITIALIZATION OR INFERENCE")
        print("=" * 80)
        print()
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Message: {str(e)}")
        print()
        print("Troubleshooting:")
        print("  1. Check if model is available (may need download)")
        print("  2. Verify GPU memory is sufficient")
        print("  3. Check RDNA3 environment variables are set correctly")
        print("  4. Try with enforce_eager=True if seeing CUDA graph issues")
        print("  5. Review error message above for specific issues")
        print()
        print("Configuration that was used:")
        engine_config.display()
        
        # Re-raise for debugging
        raise
    finally:
        # Cleanup: Properly destroy distributed process groups to avoid resource leaks
        # This addresses the warning: "destroy_process_group() was not called before program exit"
        try:
            if 'llm' in locals():
                # Try to shutdown the engine's executor (which handles worker processes)
                if hasattr(llm, 'llm_engine') and hasattr(llm.llm_engine, 'model_executor'):
                    executor = llm.llm_engine.model_executor
                    if hasattr(executor, 'shutdown'):
                        executor.shutdown()
                        print("\n✓ Cleaned up worker processes and distributed groups")
                
                # Delete the LLM instance
                del llm
                
        except Exception as cleanup_error:
            # Don't fail if cleanup has issues - the warning is harmless
            print(f"\nNote: Cleanup had minor issues (safe to ignore): {cleanup_error}")


if __name__ == "__main__":
    main()
