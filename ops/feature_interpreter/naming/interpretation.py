"""
Feature Interpretation Module.

This module provides functions to interpret the meaning of feature differences
between base and target models.
"""

import numpy as np
import logging
from typing import Dict
from collections import Counter

# Configure logging
logger = logging.getLogger(__name__)

def interpret_feature_differences(
    activation_differences: Dict,
    output_analyses: Dict = None,
    threshold: float = 0.1
) -> Dict:
    """
    Interpret the meaning of feature differences by analyzing patterns in outputs.
    
    Args:
        activation_differences: Dictionary of activation differences by prompt
        output_analyses: Dictionary of output difference analyses by prompt (optional)
        threshold: Minimum difference threshold to consider
        
    Returns:
        Dictionary of feature interpretations and patterns
    """
    logger.info("Interpreting feature differences from output patterns")
    
    if not activation_differences:
        logger.warning("Missing activation differences for feature interpretation")
        return {"features": [], "interpretation": "No activation differences provided"}
    
    # Handle empty output_analyses
    if not output_analyses:
        logger.warning("No output analyses provided, generating basic interpretation from activation data only")
        return _generate_basic_interpretation(activation_differences, threshold)
    
    # Original implementation continues below for when output_analyses is available
    # Identify prompts with significant differences
    significant_prompts = {
        prompt: act_diff for prompt, act_diff in activation_differences.items() 
        if act_diff['difference'] > threshold and prompt in output_analyses
    }
    
    if not significant_prompts:
        logger.warning(f"No significant differences found above threshold {threshold}")
        return {}
    
    # Collect patterns from output analyses for significant prompts
    pattern_counts = Counter()
    length_differences = []
    lexical_similarities = []
    
    for prompt in significant_prompts:
        if prompt not in output_analyses:
            continue
            
        # Count pattern occurrences
        patterns = output_analyses[prompt]['patterns']
        for pattern, is_present in patterns.items():
            if is_present:
                pattern_counts[pattern] += 1
        
        # Collect metrics
        length_differences.append(output_analyses[prompt]['length_difference'])
        lexical_similarities.append(output_analyses[prompt]['lexical_similarity'])
    
    # Extract unique words more common in target than base
    target_unique_words = Counter()
    base_unique_words = Counter()
    
    for prompt in significant_prompts:
        if prompt not in output_analyses:
            continue
            
        for word in output_analyses[prompt].get('target_unique', []):
            target_unique_words[word] += 1
            
        for word in output_analyses[prompt].get('base_unique', []):
            base_unique_words[word] += 1
    
    # Compute prevalence of each pattern
    total_prompts = len(significant_prompts)
    pattern_prevalence = {
        pattern: count / total_prompts 
        for pattern, count in pattern_counts.items()
    }
    
    # Determine primary and secondary feature types
    primary_pattern = max(pattern_prevalence.items(), key=lambda x: x[1])[0] if pattern_prevalence else "unknown"
    
    # Calculate average metrics
    avg_length_diff = np.mean(length_differences) if length_differences else 0
    avg_lexical_sim = np.mean(lexical_similarities) if lexical_similarities else 0
    
    # Create feature interpretation
    interpretation = {
        'primary_pattern': primary_pattern,
        'pattern_prevalence': pattern_prevalence,
        'avg_length_difference': avg_length_diff,
        'avg_lexical_similarity': avg_lexical_sim,
        'common_target_words': [word for word, count in target_unique_words.most_common(10)],
        'common_base_words': [word for word, count in base_unique_words.most_common(10)],
        'prompt_count': total_prompts
    }
    
    # Generate a textual description
    description = f"Feature characterized by {primary_pattern}"
    if avg_length_diff > 20:
        description += " with significantly longer outputs"
    elif avg_length_diff < -20:
        description += " with significantly shorter outputs"
        
    if avg_lexical_sim < 0.3:
        description += " and substantial vocabulary differences"
    
    interpretation['description'] = description
    
    logger.info(f"Interpreted feature as: {description}")
    return interpretation

def _generate_basic_interpretation(activation_differences: Dict, threshold: float = 0.1) -> Dict:
    """
    Generate a basic interpretation using only activation differences when output analyses are missing.
    
    Args:
        activation_differences: Dictionary of activation differences by prompt
        threshold: Minimum difference threshold to consider
        
    Returns:
        Basic interpretation dictionary
    """
    # Identify prompts with significant differences
    significant_prompts = {
        prompt: act_diff for prompt, act_diff in activation_differences.items() 
        if act_diff['difference'] > threshold
    }
    
    # Log the number of prompts found above threshold
    logger.warning(f"Only {len(significant_prompts)} prompts found above threshold {threshold}")
    
    # If no significant prompts found, use the top 10 by difference value as a fallback
    if not significant_prompts:
        significant_prompts = dict(sorted(
            activation_differences.items(),
            key=lambda item: item[1]['difference'],
            reverse=True
        )[:10])
        logger.warning(f"Still only {len(significant_prompts)} prompts found, using top differences instead")
        
    if not significant_prompts:
        logger.warning("No differences found at all")
        return {"features": [], "interpretation": "No significant activation differences identified"}
    
    # Count occurrences by layer
    layer_counts = Counter()
    avg_differences = []
    
    for prompt, data in significant_prompts.items():
        layer = data.get('layer', 'unknown')
        layer_counts[layer] += 1
        avg_differences.append(data['difference'])
    
    # Generate basic interpretation
    interpretation = {
        "features": [{
            "name": f"feature_in_{layer}" if layer != 'unknown' else "unknown_feature",
            "description": f"Feature identified by activation differences in {len(significant_prompts)} prompts",
            "layer": layer,
            "avg_difference": sum(avg_differences) / len(avg_differences) if avg_differences else 0,
            "significance": len(significant_prompts) / len(activation_differences) if activation_differences else 0,
            "prompts": list(significant_prompts.keys())[:10]  # Limit to 10 prompts for readability
        } for layer, count in layer_counts.most_common(3)],  # Top 3 layers with differences
        "interpretation": "Basic interpretation based only on activation differences without output analysis"
    }
    
    return interpretation 