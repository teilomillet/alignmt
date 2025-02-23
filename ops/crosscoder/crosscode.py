"""
Crosscoder implementation for model diffing.

This module implements the crosscoder approach described in:
https://transformer-circuits.pub/2024/crosscoders/index.html
with updates from:
https://transformer-circuits.pub/2025/january-update/index.html

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
    non_linear: bool = False,
    attention_patterns: bool = False,
) -> torch.Tensor:
    """
    Compute cosine similarity between two tensors with support for non-linear transformations
    and attention patterns.
    
    Args:
        tensor1: First tensor of shape (D1, D2)
        tensor2: Second tensor of shape (D1, D2)
        non_linear: Whether to apply non-linear transformation before comparison
        attention_patterns: Whether tensors represent attention patterns
        
    Returns:
        Cosine similarity matrix
    """
    # Reshape tensors if needed
    if tensor1.dim() == 1:
        tensor1 = tensor1.unsqueeze(0)
    if tensor2.dim() == 1:
        tensor2 = tensor2.unsqueeze(0)
    
    if non_linear:
        # Apply non-linear transformation (ReLU followed by normalization)
        tensor1 = F.relu(tensor1)
        tensor2 = F.relu(tensor2)
        
    if attention_patterns:
        # For attention patterns, we need to handle the head dimension specially
        # Reshape: (batch, heads, seq_len, seq_len) -> (batch * heads, seq_len * seq_len)
        if tensor1.dim() == 4:
            b, h, s1, s2 = tensor1.shape
            tensor1 = tensor1.view(b * h, s1 * s2)
        if tensor2.dim() == 4:
            b, h, s1, s2 = tensor2.shape
            tensor2 = tensor2.view(b * h, s1 * s2)
    
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
    residual_weight: float = 0.1,  # Weight for residual connections
    non_linear: bool = False,  # Whether to use non-linear similarity
    attention_patterns: bool = False,  # Whether these are attention patterns
) -> torch.Tensor:
    """
    Create a crosscoder mapping between source and target tensors.
    
    Args:
        source_tensor: Source tensor of shape (N, D)
        target_tensor: Target tensor of shape (M, D)
        temperature: Temperature for softmax scaling
        threshold: Cosine similarity threshold for identical features
        residual_weight: Weight for residual connections in similarity computation
        non_linear: Whether to use non-linear similarity metrics
        attention_patterns: Whether these are attention patterns
        
    Returns:
        Mapping matrix of shape (N, M)
    """
    # Compute direct similarities
    similarities = compute_cosine_similarity(
        source_tensor, 
        target_tensor,
        non_linear=non_linear,
        attention_patterns=attention_patterns
    )
    
    if residual_weight > 0:
        # Compute residual similarities (similarity after subtracting mean)
        source_residual = source_tensor - source_tensor.mean(dim=0, keepdim=True)
        target_residual = target_tensor - target_tensor.mean(dim=0, keepdim=True)
        
        residual_similarities = compute_cosine_similarity(
            source_residual,
            target_residual,
            non_linear=non_linear,
            attention_patterns=attention_patterns
        )
        
        # Combine direct and residual similarities
        similarities = (1 - residual_weight) * similarities + residual_weight * residual_similarities
    
    # Check if tensors are effectively identical
    if similarities.size(0) == similarities.size(1):  # Square matrix
        diag_sim = torch.diagonal(similarities)
        if torch.all(diag_sim > threshold):
            # Use very low temperature for nearly identical tensors
            temperature = 0.01
    
    # Apply temperature scaling
    scaled_similarities = similarities / temperature
    
    # Convert to probabilities with stability improvements
    max_sim = scaled_similarities.max(dim=-1, keepdim=True)[0]
    exp_similarities = torch.exp(scaled_similarities - max_sim)
    return exp_similarities / exp_similarities.sum(dim=-1, keepdim=True)

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
    residual_weight: float = 0.1,
    non_linear: bool = False,
) -> Dict[str, torch.Tensor]:
    """
    Apply crosscoding between corresponding layer parameters.
    
    Args:
        source_params: Dictionary of source parameter tensors
        target_params: Dictionary of target parameter tensors
        mapping_fn: Optional custom mapping function
        residual_weight: Weight for residual connections
        non_linear: Whether to use non-linear similarity metrics
        
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
            
        elif 'attention' in param_name and 'weight' in param_name:
            # Handle attention weights specially
            if source_tensor.dim() != 2 or target_tensor.dim() != 2:
                raise ValueError(f"Attention weight tensor {param_name} must be a matrix")
                
            # For attention, we need to consider head structure
            # Reshape: (hidden_size, num_heads * head_size) -> (num_heads, head_size, hidden_size)
            hidden_size = source_tensor.size(0)
            num_heads = source_tensor.size(1) // (hidden_size // 32)  # Infer num_heads from dimensions
            head_size = hidden_size // num_heads
            
            # Reshape tensors to expose head structure
            source_reshaped = source_tensor.t().reshape(num_heads, head_size, hidden_size)
            target_reshaped = target_tensor.t().reshape(num_heads, head_size, hidden_size)
            
            # Apply crosscoding per attention head
            mapping = mapping_fn(
                source_reshaped.reshape(num_heads, -1),
                target_reshaped.reshape(num_heads, -1),
                residual_weight=residual_weight,
                non_linear=non_linear,
                attention_patterns=True
            )
            
            # Apply mapping and reshape back
            mapped = torch.mm(mapping, target_reshaped.reshape(num_heads, -1))
            result[param_name] = mapped.reshape(num_heads, head_size, hidden_size).permute(2, 0, 1).reshape(hidden_size, -1)
            
        elif 'mlp' in param_name and 'weight' in param_name:
            # MLP weight matrices
            if source_tensor.dim() != 2 or target_tensor.dim() != 2:
                raise ValueError(f"MLP weight tensor {param_name} must be a matrix")
                
            # Apply crosscoding with residual connections and non-linear metrics
            mapping = mapping_fn(
                source_tensor,
                target_tensor,
                residual_weight=residual_weight,
                non_linear=non_linear
            )
            result[param_name] = torch.mm(mapping, target_tensor)
            
        elif 'weight' in param_name:
            # Other weight tensors
            if source_tensor.dim() != 2 or target_tensor.dim() != 2:
                raise ValueError(f"Weight tensor {param_name} must be a matrix")
                
            # Apply standard crosscoding
            mapping = mapping_fn(source_tensor, target_tensor)
            result[param_name] = torch.mm(mapping, target_tensor)
            
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