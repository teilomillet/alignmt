"""
Weight loading and comparison utilities.
"""

from .loader import (
    load_model_and_tokenizer,
    extract_model_weights,
    load_qwen_weights,
    load_deepseek_weights,
    iterate_model_layers,
    load_model_layer,
    get_layer_names,
)

from .compare import (
    analyze_weight_differences,
    find_most_different_layers,
    get_matching_keys,
    compute_weight_statistics,
)

from .crosscoder import Crosscoder
from .trainer import CrosscoderTrainer

__all__ = [
    # Loader functions
    "load_model_and_tokenizer",
    "extract_model_weights",
    "load_qwen_weights",
    "load_deepseek_weights",
    "iterate_model_layers",
    "load_model_layer",
    "get_layer_names",
    
    # Comparison functions
    "analyze_weight_differences",
    "find_most_different_layers",
    "get_matching_keys",
    "compute_weight_statistics",
    
    # Crosscoder
    "Crosscoder",
    "CrosscoderTrainer",
] 