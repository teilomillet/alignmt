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
    
    # Analyze prompts to create better feature descriptions
    prompt_keywords = _extract_prompt_themes(list(significant_prompts.keys()))
    
    # Generate enhanced interpretations for each layer
    features = []
    for layer, count in layer_counts.most_common(3):  # Top 3 layers with differences
        layer_prompts = [p for p, d in significant_prompts.items() if d.get('layer') == layer]
        layer_keywords = _extract_prompt_themes(layer_prompts)
        
        # Analyze base and target outputs to determine which model the feature belongs to
        base_outputs = [significant_prompts[p]['base_output'] for p in layer_prompts if 'base_output' in significant_prompts[p]]
        target_outputs = [significant_prompts[p]['target_output'] for p in layer_prompts if 'target_output' in significant_prompts[p]]
        
        # Determine feature direction - which model this feature primarily influences
        # Analyze based on output characteristics
        base_word_count = sum(len(output.split()) for output in base_outputs) / len(base_outputs) if base_outputs else 0
        target_word_count = sum(len(output.split()) for output in target_outputs) / len(target_outputs) if target_outputs else 0
        
        # Analyze differences in content and patterns 
        is_base_distinctive = False
        is_target_distinctive = False
        
        # Check for keywords that suggest model attribution
        for prompt in layer_prompts[:5]:  # Check a sample of prompts
            base_out = significant_prompts[prompt].get('base_output', '').lower()
            target_out = significant_prompts[prompt].get('target_output', '').lower()
            
            # Check for distinctive patterns in each model's output
            if any(kw in base_out and kw not in target_out for kw in ['logical', 'step by step', 'formal']):
                is_base_distinctive = True
            
            if any(kw in target_out and kw not in base_out for kw in ['improved', 'detailed', 'enhanced']):
                is_target_distinctive = True
        
        # Explicitly assign model attribution based on analysis
        if is_base_distinctive and not is_target_distinctive:
            model_attribution = "base"
        elif is_target_distinctive and not is_base_distinctive:
            model_attribution = "target"
        elif base_word_count > target_word_count * 1.2:
            # If base model outputs are significantly longer, feature is more likely base-specific
            model_attribution = "base"
        elif target_word_count > base_word_count * 1.2:
            # If target model outputs are significantly longer, feature is more likely target-specific
            model_attribution = "target" 
        else:
            # If no clear signal, use the feature theme to make a best guess
            feature_theme = layer_keywords['primary_theme'].lower()
            if feature_theme in ['logical', 'probabilistic', 'formal_logic']:
                model_attribution = "base"  # Some reasoning patterns might be stronger in base model
            else:
                model_attribution = "target"  # Default to target for other cases
        
        # Generate a more descriptive name and description
        feature_theme = layer_keywords['primary_theme'] if layer_keywords['primary_theme'] else prompt_keywords['primary_theme']
        
        # Include model attribution in feature name for better interpretability
        name_prefix = "base" if model_attribution == "base" else "target"
        feature_name = f"{name_prefix}_{feature_theme}_in_{layer}" if layer != 'unknown' else f"{name_prefix}_{feature_theme}_feature"
        
        # Create a more detailed description with explicit model attribution
        description = f"Feature in layer {layer.split('.')[-1]} "
        if model_attribution == "base":
            description += f"that appears stronger in the base model, "
        else:
            description += f"that appears stronger in the target model, "
        description += f"related to {feature_theme} reasoning "
        if layer_keywords['secondary_themes']:
            description += f"with elements of {', '.join(layer_keywords['secondary_themes'][:2])} "
        description += f"(identified in {len(layer_prompts)} prompts)"
        
        # Get the average difference
        layer_avg_diff = np.mean([d['difference'] for p, d in significant_prompts.items() 
                                if d.get('layer') == layer])
        
        features.append({
            "name": feature_name,
            "description": description,
            "layer": layer,
            "avg_difference": layer_avg_diff,
            "significance": count / len(activation_differences) if activation_differences else 0,
            "prompts": layer_prompts[:10],  # Limit to 10 prompts for readability
            "themes": {
                "primary": feature_theme,
                "secondary": layer_keywords['secondary_themes']
            },
            "model_attribution": model_attribution  # Explicitly store model attribution
        })
    
    # Generate overall interpretation
    overall_interpretation = f"Identified {len(features)} features primarily related to "
    overall_interpretation += f"{prompt_keywords['primary_theme']} reasoning"
    if prompt_keywords['secondary_themes']:
        overall_interpretation += f" with elements of {', '.join(prompt_keywords['secondary_themes'][:3])}"
    
    # Count features by model
    base_count = sum(1 for f in features if f.get('model_attribution') == 'base')
    target_count = sum(1 for f in features if f.get('model_attribution') == 'target')
    
    # Add model distribution to interpretation
    overall_interpretation += f". Found {base_count} features specific to the base model and {target_count} features specific to the target model."
    
    interpretation = {
        "features": features,
        "interpretation": overall_interpretation
    }
    
    return interpretation

def _extract_prompt_themes(prompts):
    """
    Extract themes and topics from a list of prompts.
    
    Args:
        prompts: List of prompt strings
        
    Returns:
        Dictionary with primary theme and list of secondary themes
    """
    # Define categories of reasoning and associated keywords
    reasoning_categories = {
        "mathematical": ["equation", "solve", "math", "calculation", "compute", "number", "formula", 
                         "arithmetic", "algebra", "calculation"],
        "probabilistic": ["probability", "random", "chance", "likelihood", "dice", "cards", "coin toss",
                          "expected value", "balls", "draw"],
        "logical": ["logic", "if then", "deduction", "inference", "valid", "argument", "conclusion", 
                    "reasoning", "statement", "premise"],
        "spatial": ["distance", "area", "volume", "perimeter", "shape", "geometric", "rectangle", 
                   "square", "triangle", "dimension"],
        "temporal": ["time", "duration", "hours", "minutes", "seconds", "timeline", "schedule", 
                     "before", "after", "during"],
        "quantitative": ["quantity", "amount", "total", "sum", "difference", "product", "quotient", 
                         "measurement", "count", "rate"],
        "economic": ["money", "cost", "price", "buy", "sell", "purchase", "dollar", "spend", "save", 
                    "discount", "sale", "economics"]
    }
    
    # Count occurrences of each category in prompts
    category_counts = Counter()
    
    for prompt in prompts:
        prompt_lower = prompt.lower()
        for category, keywords in reasoning_categories.items():
            for keyword in keywords:
                if keyword.lower() in prompt_lower:
                    category_counts[category] += 1
                    break  # Count each category only once per prompt
    
    # Determine primary and secondary themes
    primary_theme = category_counts.most_common(1)[0][0] if category_counts else "general"
    
    # Get secondary themes (excluding primary)
    secondary_themes = [category for category, count in category_counts.most_common() 
                        if category != primary_theme and count > 0]
    
    return {
        "primary_theme": primary_theme,
        "secondary_themes": secondary_themes
    } 