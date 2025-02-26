"""
Feature Extraction Module.

This module provides functions to extract distinctive features from
activation differences between base and target models.
"""

import numpy as np
import logging
from typing import Dict

# Configure logging
logger = logging.getLogger(__name__)

def extract_distinctive_features(
    activation_differences: Dict,
    threshold: float = 0.1,
    min_prompts: int = 5
) -> Dict:
    """
    Extract distinctive features from activation differences.
    
    Args:
        activation_differences: Dictionary of activation differences by prompt
        threshold: Minimum difference threshold to consider a feature distinctive
        min_prompts: Minimum number of prompts where a feature must be distinctive
        
    Returns:
        Dictionary of distinctive features with their characteristics
    """
    logger.info(f"Extracting distinctive features with threshold {threshold} and min_prompts {min_prompts}")
    
    if not activation_differences:
        logger.warning("No activation differences provided for feature extraction")
        return {}
    
    # Aggregate similarities and differences across prompts
    all_similarities = [data['similarity'] for data in activation_differences.values()]
    all_differences = [data['difference'] for data in activation_differences.values()]
    
    # Identify prompts with significant differences
    significant_prompts = {
        prompt: data for prompt, data in activation_differences.items() 
        if data['difference'] > threshold
    }
    
    # Extract the layer name from the first entry (should be consistent)
    layer_name = next(iter(activation_differences.values()))['layer'] if activation_differences else "unknown"
    
    # Compute basic statistics
    avg_similarity = np.mean(all_similarities) if all_similarities else 0
    avg_difference = np.mean(all_differences) if all_differences else 0
    
    # Group prompts by similarity characteristics
    similarity_groups = {}
    for prompt, data in activation_differences.items():
        # Discretize similarity values into bins
        sim_bin = round(data['similarity'] * 10) / 10
        if sim_bin not in similarity_groups:
            similarity_groups[sim_bin] = []
        similarity_groups[sim_bin].append(prompt)
    
    # Create distinctive feature summary
    feature_info = {
        'layer': layer_name,
        'avg_similarity': avg_similarity,
        'avg_difference': avg_difference,
        'significant_prompt_count': len(significant_prompts),
        'total_prompt_count': len(activation_differences),
        'similarity_distribution': {str(k): len(v) for k, v in similarity_groups.items()},
        'significant_prompts': list(significant_prompts.keys())[:20]  # Limit for readability
    }
    
    # Only consider this a distinctive feature if enough prompts show significant differences
    if len(significant_prompts) >= min_prompts:
        feature_info['is_distinctive'] = True
        logger.info(f"Identified distinctive feature in layer {layer_name} with {len(significant_prompts)} significant prompts")
    else:
        feature_info['is_distinctive'] = False
        logger.info(f"Layer {layer_name} did not meet the criteria for a distinctive feature")
    
    return feature_info 