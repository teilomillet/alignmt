"""
Crosscoder implementation for model diffing.

This module implements the crosscoder approach described in:
https://transformer-circuits.pub/2024/crosscoders/index.html

The implementation uses functional programming principles for clarity and composability.
Each function is pure and handles a specific part of the crosscoding process.
"""

from typing import Dict, List, Tuple, Optional, Callable, Union
import torch
import torch.nn.functional as F
from pathlib import Path

from ..loader.load import load_model_layer, get_layer_names

def compute_cosine_similarity(
    tensor1: torch.Tensor,
    tensor2: torch.Tensor,
) -> torch.Tensor:
    """
    Compute cosine similarity between two tensors.
    
    Args:
        tensor1: First tensor of shape (D1, D2)
        tensor2: Second tensor of shape (D1, D2)
        
    Returns:
        Cosine similarity matrix
    """
    # Reshape tensors if needed
    if tensor1.dim() == 1:
        tensor1 = tensor1.unsqueeze(0)
    if tensor2.dim() == 1:
        tensor2 = tensor2.unsqueeze(0)
        
    # Normalize tensors along last dimension
    tensor1_normalized = F.normalize(tensor1, p=2, dim=-1)
    tensor2_normalized = F.normalize(tensor2, p=2, dim=-1)
    
    # Compute similarity
    return torch.mm(tensor1_normalized, tensor2_normalized.t())

def create_crosscoder_mapping(
    source_tensor: torch.Tensor,
    target_tensor: torch.Tensor,
    temperature: float = 1.0,
    threshold: float = 0.9999,  # Threshold for considering tensors identical
) -> torch.Tensor:
    """
    Create a crosscoder mapping between source and target tensors.
    
    Args:
        source_tensor: Source tensor of shape (N, D)
        target_tensor: Target tensor of shape (M, D)
        temperature: Temperature for softmax scaling
        threshold: Cosine similarity threshold for identical features
        
    Returns:
        Mapping matrix of shape (N, M)
    """
    # Compute similarities
    similarities = compute_cosine_similarity(source_tensor, target_tensor)
    
    # Check if tensors are effectively identical
    if torch.all(torch.abs(similarities - torch.eye(*similarities.shape, device=similarities.device)) < (1 - threshold)):
        return torch.eye(*similarities.shape, device=similarities.device)
    
    # Apply temperature scaling
    scaled_similarities = similarities / temperature
    
    # Convert to probabilities
    return F.softmax(scaled_similarities, dim=-1)

def compute_parameter_difference(
    tensor1: torch.Tensor,
    tensor2: torch.Tensor,
    return_tensor: bool = True
) -> Union[torch.Tensor, float]:
    """
    Compute difference between two tensors.
    
    Args:
        tensor1: First tensor
        tensor2: Second tensor
        return_tensor: Whether to return the full difference tensor
        
    Returns:
        If return_tensor is True:
            The difference tensor (tensor1 - tensor2)
        If return_tensor is False:
            A scalar metric of the difference:
            - 0.0 for identical tensors (including both zero)
            - 1.0 for tensors that differ only by scaling
            - 2.0 for tensors that cannot be mapped to each other
    """
    # Handle zero tensors
    if torch.all(tensor1 == 0) and torch.all(tensor2 == 0):
        return torch.zeros_like(tensor1) if return_tensor else 0.0
        
    if return_tensor:
        return tensor1 - tensor2
        
    # Check if tensors are identical first
    if torch.allclose(tensor1, tensor2, rtol=1e-5, atol=1e-8):
        return 0.0
        
    # Compute normalized difference for scalar metric
    norm1 = torch.norm(tensor1)
    norm2 = torch.norm(tensor2)
    
    # Handle zero norms
    if norm1 == 0 or norm2 == 0:
        return 2.0
        
    # Normalize tensors
    normalized1 = tensor1 / norm1
    normalized2 = tensor2 / norm2
    
    # Check if tensors differ only by scaling (including sign changes)
    # This means their normalized versions should be identical or negatives
    if torch.allclose(normalized1, normalized2, rtol=1e-5, atol=1e-8) or \
       torch.allclose(normalized1, -normalized2, rtol=1e-5, atol=1e-8):
        return 1.0
        
    # Tensors cannot be mapped to each other
    return 2.0

def crosscode_layer_params(
    source_params: Dict[str, torch.Tensor],
    target_params: Dict[str, torch.Tensor],
    mapping_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
) -> Dict[str, torch.Tensor]:
    """
    Apply crosscoding between corresponding layer parameters.
    
    Args:
        source_params: Dictionary of source parameter tensors
        target_params: Dictionary of target parameter tensors
        mapping_fn: Optional custom mapping function
        
    Returns:
        Dictionary of crosscoded parameters
    """
    if mapping_fn is None:
        mapping_fn = create_crosscoder_mapping
        
    result = {}
    
    # Process each parameter type
    for param_name in source_params:
        if param_name not in target_params:
            continue
            
        source_tensor = source_params[param_name]
        target_tensor = target_params[param_name]
        
        # Validate dimensions based on parameter type
        if any(t in param_name for t in ['layernorm', 'bias', 'norm', 'scale']):
            # Layer norm and other 1D parameters
            if source_tensor.dim() != 1 or target_tensor.dim() != 1:
                raise ValueError(f"Parameter {param_name} must be a vector")
                
            # Copy these parameters directly
            result[param_name] = target_tensor.clone()
            
        elif 'mlp' in param_name and 'weight' in param_name:
            # MLP weight matrices
            if source_tensor.dim() != 2 or target_tensor.dim() != 2:
                raise ValueError(f"MLP weight tensor {param_name} must be a matrix")
                
            # Apply crosscoding
            mapping = mapping_fn(source_tensor, target_tensor)
            result[param_name] = torch.mm(mapping, target_tensor)
            
        elif 'weight' in param_name:
            # Other weight tensors (attention, etc)
            if source_tensor.dim() != 2 or target_tensor.dim() != 2:
                raise ValueError(f"Weight tensor {param_name} must be a matrix")
                
            # Copy these parameters directly
            result[param_name] = target_tensor.clone()
            
        else:
            # Unknown parameters - copy as is
            result[param_name] = target_tensor.clone()
            
    return result

def diff_layer_params(
    source_params: Dict[str, torch.Tensor],
    target_params: Dict[str, torch.Tensor],
    crosscode: bool = True,
) -> Dict[str, Dict[str, Union[torch.Tensor, float]]]:
    """
    Compute differences between layer parameters.
    
    Args:
        source_params: Dictionary of source parameter tensors
        target_params: Dictionary of target parameter tensors
        crosscode: Whether to apply crosscoding
        
    Returns:
        Dictionary containing:
        - 'differences': Parameter differences as scalar metrics
        - 'similarities': Absolute cosine similarities between parameters (0 to 1)
        - 'crosscoded': Crosscoded parameters
    """
    results = {
        'differences': {},
        'similarities': {},
        'crosscoded': {}
    }
    
    if crosscode:
        # Apply crosscoding
        results['crosscoded'] = crosscode_layer_params(source_params, target_params)
        
        # Compute differences with crosscoded parameters
        for name in source_params:
            if name in results['crosscoded']:
                results['differences'][name] = compute_parameter_difference(
                    source_params[name],
                    results['crosscoded'][name],
                    return_tensor=False  # Return scalar metric instead of tensor
                )
    else:
        # Direct differences
        for name in source_params:
            if name in target_params:
                results['differences'][name] = compute_parameter_difference(
                    source_params[name],
                    target_params[name],
                    return_tensor=False  # Return scalar metric instead of tensor
                )
    
    # Compute similarities for all parameters
    for name in source_params:
        if name in target_params:
            # Reshape tensors if needed
            source = source_params[name]
            target = target_params[name]
            if source.dim() == 1:
                source = source.unsqueeze(0)
            if target.dim() == 1:
                target = target.unsqueeze(0)
                
            # Compute cosine similarity
            similarity = compute_cosine_similarity(source, target)
            
            # For identical tensors, similarity should be 1.0
            if torch.allclose(source, target, rtol=1e-5, atol=1e-8):
                results['similarities'][name] = 1.0
            else:
                # Take absolute max similarity and clamp to [0, 1] to handle numerical precision
                similarity = float(torch.abs(similarity).max().item())
                results['similarities'][name] = max(0.0, min(1.0, similarity))
            
    return results

def analyze_layer_changes(
    source: Union[str, Dict[str, torch.Tensor]],
    target: Union[str, Dict[str, torch.Tensor]],
    layer_name: Optional[str] = None,
    cache_dir: Optional[str] = None,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Analyze changes between corresponding layers in two models.
    
    Args:
        source: Either model name or dictionary of source parameters
        target: Either model name or dictionary of target parameters
        layer_name: Name of layer to analyze (required if using model names)
        cache_dir: Optional cache directory
        device: Device to use
        dtype: Data type to use
        
    Returns:
        Dictionary containing analysis results for each parameter
    """
    # Handle direct parameter input
    if isinstance(source, dict) and isinstance(target, dict):
        source_params = source
        target_params = target
    else:
        # Load layers from models
        if layer_name is None:
            raise ValueError("layer_name is required when using model names")
            
        source_params = load_model_layer(
            source,
            layer_name,
            dtype=dtype,
            device=device,
            cache_dir=cache_dir
        )
        target_params = load_model_layer(
            target,
            layer_name,
            dtype=dtype,
            device=device,
            cache_dir=cache_dir
        )
    
    # Analyze differences
    return diff_layer_params(source_params, target_params)

def analyze_model_changes(
    source_model: str,
    target_model: str,
    cache_dir: Optional[str] = None,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
) -> Dict[str, Dict[str, Dict[str, torch.Tensor]]]:
    """
    Analyze changes between two models layer by layer.
    
    Args:
        source_model: Name of source model
        target_model: Name of target model
        cache_dir: Optional cache directory
        device: Device to use
        dtype: Data type to use
        
    Returns:
        Dictionary mapping layer names to their analysis results
    """
    # Get layer names
    layer_names = get_layer_names(source_model)
    
    # Analyze each layer
    results = {}
    for layer_name in layer_names:
        results[layer_name] = analyze_layer_changes(
            source_model,
            target_model,
            layer_name,
            cache_dir=cache_dir,
            device=device,
            dtype=dtype
        )
        
        # Clean up memory
        torch.cuda.empty_cache()
        
    return results 