"""
Weight loading and preparation utilities for model comparison.
"""

from pathlib import Path
from typing import Dict, Tuple, Optional, Union, Iterator, List
import logging
import os
import shutil

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

def remove_model_directory(model_name: str, cache_dir: Optional[str] = None):
    """
    Remove a model's directory to free up disk space.
    
    Args:
        model_name: HuggingFace model name
        cache_dir: Cache directory
    """
    # Convert model name to directory format
    model_dir_name = model_name.replace("/", "--")
    
    # Determine cache directory
    if cache_dir is None:
        cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    
    # Path to model directory
    model_path = os.path.join(cache_dir, f"models--{model_dir_name}")
    
    if os.path.exists(model_path):
        logger.info(f"Removing model directory for {model_name} to free up space")
        try:
            shutil.rmtree(model_path)
            logger.info(f"Successfully removed {model_path}")
        except Exception as e:
            logger.warning(f"Failed to remove {model_path}: {str(e)}")

def load_model_layer(
    model_name: str,
    layer_name: str,
    dtype: torch.dtype = torch.float32,
    device: str = "cuda",
    cache_dir: Optional[str] = None,
) -> Dict[str, torch.Tensor]:
    """
    Load a single layer from a model.
    
    Args:
        model_name: HuggingFace model name
        layer_name: Name of the layer to load
        dtype: Data type to load
        device: Device to load onto
        cache_dir: Cache directory
        
    Returns:
        Dictionary mapping parameter names to their tensor values
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
        
        # Extract all matching layer parameters
        state_dict = {}
        for name, param in model.named_parameters():
            if layer_name in name:
                # Store parameter with its full name for clarity
                state_dict[name] = param.detach().clone()
                
                # Handle different parameter types
                if 'weight' in name:
                    # Transpose weight matrices
                    state_dict[name] = state_dict[name].t()
                elif 'bias' in name:
                    # Keep biases as is
                    pass
                elif 'scale' in name or 'norm' in name:
                    # Handle layer norm parameters
                    pass
                
        # Clean up
        del model
        gc.collect()
        torch.cuda.empty_cache()
        
        if not state_dict:
            raise ValueError(f"Layer {layer_name} not found in model")
            
        return state_dict
        
    except OSError as e:
        # Check if it's a disk space error
        if "No space left on device" in str(e):
            logger.warning(f"Disk space error encountered. Removing model directory and retrying.")
            
            # Remove the model directory to free up space
            remove_model_directory(model_name, cache_dir)
            
            # Raise the error to allow the caller to retry
            logger.error(f"Please retry the operation: {str(e)}")
        
        # Re-raise the exception
        logger.error(f"Failed to load layer {layer_name}: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Failed to load layer {layer_name}: {str(e)}")
        raise

def iterate_model_layers(
    model_name: str,
    dtype: torch.dtype = torch.float32,
    device: str = "cuda",
    cache_dir: Optional[str] = None,
) -> Iterator[Tuple[str, Dict[str, torch.Tensor]]]:
    """
    Iterate through model layers one at a time.
    
    Args:
        model_name: HuggingFace model name
        dtype: Data type to load
        device: Device to load onto
        cache_dir: Cache directory
        
    Yields:
        Tuples of (layer_name, parameter_dict)
    """
    layer_names = get_layer_names(model_name)
    
    for layer_name in layer_names:
        try:
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
        except OSError as e:
            if "No space left on device" in str(e):
                # If we've already tried to remove the model directory, but still have disk issues
                logger.error(f"Persistent disk space issues. Please free up disk space manually.")
                raise
            else:
                raise

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
        
    except OSError as e:
        # Check if it's a disk space error
        if "No space left on device" in str(e):
            logger.warning(f"Disk space error encountered. Removing model directory.")
            remove_model_directory(model_name, cache_dir)
            logger.error(f"Please retry after freeing up space: {str(e)}")
        else:
            logger.error(f"Failed to load model {model_name}: {str(e)}")
        raise
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