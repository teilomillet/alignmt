"""
Weight loading and preparation utilities for model comparison.
"""

from pathlib import Path
from typing import Dict, Tuple, Optional, Union, Iterator, List
import logging

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer
import gc

# Configure logging
logger = logging.getLogger(__name__)

def get_layer_names(model_name: str) -> List[str]:
    """Get the names of weight layers in order."""
    if "Qwen" in model_name:
        return [f"model.layers.{i}.mlp.gate_proj.weight" for i in range(28)]
    elif "DeepSeek" in model_name:
        return [f"model.layers.{i}.mlp.gate_proj.weight" for i in range(28)]
    else:
        raise ValueError(f"Unsupported model: {model_name}")

def load_model_layer(
    model_name: str,
    layer_name: str,
    dtype: torch.dtype = torch.float32,
    device: str = "cuda",
    cache_dir: Optional[str] = None,
) -> torch.Tensor:
    """
    Load a single layer from a model.
    
    Args:
        model_name: HuggingFace model name
        layer_name: Name of the layer to load
        dtype: Data type to load
        device: Device to load onto
        cache_dir: Cache directory
        
    Returns:
        Tensor containing the layer weights
    """
    try:
        # Load model with minimal memory usage
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            trust_remote_code=True,
            torch_dtype=dtype,
            device_map={"": device},
            max_memory={device: "4GiB"},
            offload_folder="offload",
            low_cpu_mem_usage=True
        )
        
        # Extract just the layer we want
        state_dict = {}
        for name, param in model.named_parameters():
            if layer_name in name:
                # Transpose the weights to get correct dimensions
                state_dict[name] = param.detach().clone().t()
                break
                
        # Clean up
        del model
        gc.collect()
        torch.cuda.empty_cache()
        
        if not state_dict:
            raise ValueError(f"Layer {layer_name} not found in model")
            
        return next(iter(state_dict.values()))
        
    except Exception as e:
        logger.error(f"Failed to load layer {layer_name}: {str(e)}")
        raise

def iterate_model_layers(
    model_name: str,
    dtype: torch.dtype = torch.float32,
    device: str = "cuda",
    cache_dir: Optional[str] = None,
) -> Iterator[Tuple[str, torch.Tensor]]:
    """
    Iterate through model layers one at a time.
    
    Args:
        model_name: HuggingFace model name
        dtype: Data type to load
        device: Device to load onto
        cache_dir: Cache directory
        
    Yields:
        Tuples of (layer_name, weights)
    """
    layer_names = get_layer_names(model_name)
    
    for layer_name in layer_names:
        weights = load_model_layer(
            model_name,
            layer_name,
            dtype=dtype,
            device=device,
            cache_dir=cache_dir
        )
        yield layer_name, weights
        
        # Clean up
        del weights
        gc.collect()
        torch.cuda.empty_cache()

def load_model_and_tokenizer(
    model_name: str,
    use_bf16: bool = True,
    device_map: Optional[Union[str, Dict]] = "auto",
    cache_dir: Optional[Union[str, Path]] = None,
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    Pure function to load a model and its tokenizer.
    
    Args:
        model_name: HuggingFace model identifier
        use_bf16: Whether to load in bfloat16 precision
        device_map: Device mapping strategy for model loading
        cache_dir: Directory to cache downloaded models
    
    Returns:
        Tuple of (model, tokenizer)
        
    Raises:
        ValueError: If model loading fails
        OSError: If network or file system errors occur
    """
    try:
        dtype = torch.bfloat16 if use_bf16 else torch.float32
        logger.info(f"Loading model {model_name} with dtype {dtype}")
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=device_map,
            cache_dir=cache_dir,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            trust_remote_code=True
        )
        
        logger.info(f"Successfully loaded {model_name}")
        return model, tokenizer
        
    except Exception as e:
        logger.error(f"Failed to load model {model_name}: {str(e)}")
        raise

def extract_model_weights(
    model: PreTrainedModel,
    detach: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Pure function to extract weights from a model into a flat dictionary.
    
    Args:
        model: The loaded transformer model
        detach: Whether to detach tensors from computation graph
    
    Returns:
        Dictionary mapping weight names to their tensor values
    """
    try:
        state_dict = model.state_dict()
        
        if detach:
            return {
                name: param.detach().clone()
                for name, param in state_dict.items()
            }
        return {
            name: param.clone()
            for name, param in state_dict.items()
        }
        
    except Exception as e:
        logger.error(f"Failed to extract weights: {str(e)}")
        raise

def load_qwen_weights(
    cache_dir: Optional[str] = None,
    use_bf16: bool = True
) -> Dict[str, torch.Tensor]:
    """Load Qwen weights layer by layer."""
    model_name = "Qwen/Qwen2-1.5B"
    dtype = torch.bfloat16 if use_bf16 else torch.float32
    
    weights = {}
    for name, tensor in iterate_model_layers(
        model_name,
        dtype=dtype,
        cache_dir=cache_dir
    ):
        weights[name] = tensor
    return weights

def load_deepseek_weights(
    cache_dir: Optional[str] = None,
    use_bf16: bool = True
) -> Dict[str, torch.Tensor]:
    """Load DeepSeek weights layer by layer."""
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    dtype = torch.bfloat16 if use_bf16 else torch.float32
    
    weights = {}
    for name, tensor in iterate_model_layers(
        model_name,
        dtype=dtype,
        cache_dir=cache_dir
    ):
        weights[name] = tensor
    return weights 