"""
Weight comparison and analysis utilities.
"""

from typing import Dict, List, Tuple, Set
import logging
from pathlib import Path
import json

import torch
import numpy as np
from torch import Tensor

logger = logging.getLogger(__name__)

def get_matching_keys(
    weights1: Dict[str, Tensor],
    weights2: Dict[str, Tensor]
) -> Tuple[Set[str], Set[str], Set[str]]:
    """
    Find matching and unique keys between two weight dictionaries.
    
    Args:
        weights1: First weight dictionary
        weights2: Second weight dictionary
        
    Returns:
        Tuple of (matching_keys, only_in_weights1, only_in_weights2)
    """
    keys1 = set(weights1.keys())
    keys2 = set(weights2.keys())
    
    matching = keys1 & keys2
    only_1 = keys1 - keys2
    only_2 = keys2 - keys1
    
    if only_1:
        logger.info(f"Keys only in first weights: {len(only_1)}")
    if only_2:
        logger.info(f"Keys only in second weights: {len(only_2)}")
        
    return matching, only_1, only_2

def compute_weight_statistics(
    weights1: Dict[str, Tensor],
    weights2: Dict[str, Tensor],
    matching_keys: Set[str]
) -> Dict[str, Dict[str, float]]:
    """
    Compute basic statistics between matching weights.
    
    Args:
        weights1: First weight dictionary
        weights2: Second weight dictionary
        matching_keys: Set of keys present in both dictionaries
        
    Returns:
        Dictionary of statistics per layer
    """
    stats = {}
    
    for key in matching_keys:
        w1 = weights1[key].float().cpu()
        w2 = weights2[key].float().cpu()
        
        if w1.shape != w2.shape:
            logger.warning(f"Shape mismatch for {key}: {w1.shape} vs {w2.shape}")
            continue
            
        # Compute basic statistics
        diff = w1 - w2
        abs_diff = torch.abs(diff)
        
        stats[key] = {
            "mean_diff": float(torch.mean(diff).item()),
            "std_diff": float(torch.std(diff).item()),
            "max_abs_diff": float(torch.max(abs_diff).item()),
            "mean_abs_diff": float(torch.mean(abs_diff).item()),
            "cosine_similarity": float(torch.nn.functional.cosine_similarity(
                w1.flatten(), w2.flatten(), dim=0
            ).item())
        }
        
    return stats

def analyze_weight_differences(
    weights1: Dict[str, Tensor],
    weights2: Dict[str, Tensor],
    output_file: Path = None
) -> Dict:
    """
    Analyze differences between two sets of weights.
    
    Args:
        weights1: First weight dictionary
        weights2: Second weight dictionary
        output_file: Optional path to save results
        
    Returns:
        Dictionary containing analysis results
    """
    try:
        matching, only_1, only_2 = get_matching_keys(weights1, weights2)
        
        if not matching:
            logger.error("No matching keys found between weights")
            return {}
            
        logger.info(f"Analyzing {len(matching)} matching layers...")
        
        # Compute statistics for matching weights
        stats = compute_weight_statistics(weights1, weights2, matching)
        
        # Prepare summary
        summary = {
            "matching_layers": len(matching),
            "only_in_weights1": len(only_1),
            "only_in_weights2": len(only_2),
            "layer_statistics": stats
        }
        
        # Save results if requested
        if output_file:
            output_file = Path(output_file)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w') as f:
                json.dump(summary, f, indent=2)
            logger.info(f"Saved analysis results to {output_file}")
            
        return summary
        
    except Exception as e:
        logger.error(f"Failed to analyze weights: {str(e)}")
        raise

def find_most_different_layers(
    stats: Dict[str, Dict[str, float]],
    metric: str = "cosine_similarity",
    top_k: int = 5
) -> List[Tuple[str, float]]:
    """
    Find the layers with the most differences according to a metric.
    
    Args:
        stats: Layer statistics from analyze_weight_differences
        metric: Metric to sort by
        top_k: Number of layers to return
        
    Returns:
        List of (layer_name, metric_value) tuples
    """
    if metric not in next(iter(stats.values())):
        raise ValueError(f"Invalid metric: {metric}")
        
    sorted_layers = sorted(
        [(k, v[metric]) for k, v in stats.items()],
        key=lambda x: x[1],
        reverse=metric != "cosine_similarity"
    )
    
    return sorted_layers[:top_k] 