"""
Activation Differences Analysis Module.

This module provides functions to compute and analyze differences between
base and target model activations and outputs.
"""

import torch
import numpy as np
import logging
from typing import Dict

# Configure logging
logger = logging.getLogger(__name__)

def compute_activation_differences(
    base_activations: Dict,
    target_activations: Dict,
    layer_name: str
) -> Dict:
    """
    Compute differences between base and target model activations.
    
    Args:
        base_activations: Dictionary of base model activations
        target_activations: Dictionary of target model activations
        layer_name: Name of the layer to analyze
        
    Returns:
        Dictionary with activation differences
    """
    logger.info(f"Computing activation differences for layer: {layer_name}")
    
    differences = {}
    
    for prompt in base_activations:
        if prompt not in target_activations:
            continue
            
        if 'activations' not in base_activations[prompt] or 'activations' not in target_activations[prompt]:
            continue
            
        if layer_name not in base_activations[prompt]['activations'] or layer_name not in target_activations[prompt]['activations']:
            continue
        
        # Extract activations for this prompt and layer
        base_act = base_activations[prompt]['activations'][layer_name]
        target_act = target_activations[prompt]['activations'][layer_name]
        
        # Create tensors if not already
        if not isinstance(base_act, torch.Tensor):
            base_act = torch.tensor(base_act)
        if not isinstance(target_act, torch.Tensor):
            target_act = torch.tensor(target_act)
        
        # Convert to float32 if needed to handle bfloat16
        if base_act.dtype != torch.float32:
            base_act = base_act.to(torch.float32)
        if target_act.dtype != torch.float32:
            target_act = target_act.to(torch.float32)
        
        # Mean pool across sequence length if present
        if len(base_act.shape) > 2:
            base_act = base_act.mean(dim=1)
        if len(target_act.shape) > 2:
            target_act = target_act.mean(dim=1)
        
        # Flatten batch dimension if present
        if len(base_act.shape) > 1:
            base_act = base_act.reshape(-1)
        if len(target_act.shape) > 1:
            target_act = target_act.reshape(-1)
        
        # Normalize to unit length for similarity computation
        base_norm = torch.norm(base_act, p=2)
        target_norm = torch.norm(target_act, p=2)
        
        if base_norm > 0 and target_norm > 0:
            base_act_normalized = base_act / base_norm
            target_act_normalized = target_act / target_norm
            
            # Compute cosine similarity
            similarity = torch.dot(base_act_normalized, target_act_normalized).item()
            
            # Compute Euclidean distance and normalize by dimensions
            difference = torch.norm(base_act_normalized - target_act_normalized, p=2).item() / np.sqrt(base_act.numel())
            
            # Store results
            differences[prompt] = {
                'similarity': similarity,
                'difference': difference,
                'base_output': base_activations[prompt].get('text', ''),
                'target_output': target_activations[prompt].get('text', ''),
                'layer': layer_name
            }
    
    logger.info(f"Computed activation differences for {len(differences)} prompts")
    return differences

def analyze_output_differences(
    base_output: str,
    target_output: str
) -> Dict:
    """
    Analyze differences between base and target model outputs.
    
    Args:
        base_output: Output text from base model
        target_output: Output text from target model
        
    Returns:
        Dictionary with output difference analysis
    """
    # Compute simple text length difference
    base_length = len(base_output.split())
    target_length = len(target_output.split())
    length_difference = target_length - base_length
    
    # Identify common and unique words
    base_words = set(base_output.lower().split())
    target_words = set(target_output.lower().split())
    
    common_words = base_words.intersection(target_words)
    base_unique = base_words - common_words
    target_unique = target_words - common_words
    
    # Compute a basic lexical similarity score
    total_words = len(base_words.union(target_words))
    
    if total_words > 0:
        lexical_similarity = len(common_words) / total_words
    else:
        lexical_similarity = 0.0
    
    # Look for specific patterns in differences
    patterns = {
        'reasoning': any('reason' in w or 'explain' in w or 'because' in w for w in target_unique) and not any('reason' in w or 'explain' in w or 'because' in w for w in base_unique),
        'step_by_step': any('step' in w or 'first' in w or 'second' in w or 'finally' in w for w in target_unique) and not any('step' in w or 'first' in w or 'second' in w or 'finally' in w for w in base_unique),
        'mathematical': any('calculate' in w or 'equation' in w or 'solve' in w for w in target_unique) and not any('calculate' in w or 'equation' in w or 'solve' in w for w in base_unique),
        'detail': length_difference > 20 and lexical_similarity < 0.5,
        'concise': length_difference < -20 and lexical_similarity < 0.5
    }
    
    return {
        'length_difference': length_difference,
        'lexical_similarity': lexical_similarity,
        'common_word_count': len(common_words),
        'base_unique_count': len(base_unique),
        'target_unique_count': len(target_unique),
        'base_unique': list(base_unique)[:10],  # Limit for readability
        'target_unique': list(target_unique)[:10],  # Limit for readability
        'patterns': patterns
    } 