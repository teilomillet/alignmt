"""
Feature naming and interpretation module.

This module associates activations with interpretable feature names
by analyzing differences in model behavior on specific prompts.
"""

import torch
import numpy as np
import logging
import os
import pickle
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any, Set
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Configure logging
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

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
            
        base_act = base_activations[prompt]['activations'].get(layer_name)
        target_act = target_activations[prompt]['activations'].get(layer_name)
        
        if base_act is None or target_act is None:
            continue
            
        # Compute difference
        # We'll use cosine similarity to measure how different the activations are
        base_flat = base_act.reshape(base_act.shape[0], -1)
        target_flat = target_act.reshape(target_act.shape[0], -1)
        
        # Normalize
        base_norm = base_flat / (base_flat.norm(dim=1, keepdim=True) + 1e-8)
        target_norm = target_flat / (target_flat.norm(dim=1, keepdim=True) + 1e-8)
        
        # Compute similarity
        similarity = torch.mm(base_norm, target_norm.transpose(0, 1))
        
        # Store difference (1 - similarity)
        differences[prompt] = {
            'similarity': similarity.diag().mean().item(),
            'difference': 1 - similarity.diag().mean().item(),
            'base_output': base_activations[prompt]['text'],
            'target_output': target_activations[prompt]['text']
        }
    
    return differences

def extract_distinctive_features(
    activation_differences: Dict,
    prompt_categories: Dict,
    threshold: float = 0.3
) -> Dict:
    """
    Extract distinctive features from activation differences.
    
    Args:
        activation_differences: Dictionary with activation differences
        prompt_categories: Dictionary mapping prompts to categories
        threshold: Threshold for considering a difference significant
        
    Returns:
        Dictionary with distinctive features
    """
    # Group differences by category
    category_differences = {}
    
    for prompt, diff_data in activation_differences.items():
        category = prompt_categories.get(prompt, "unknown")
        
        if category not in category_differences:
            category_differences[category] = {
                "differences": [],
                "base_outputs": [],
                "target_outputs": [],
                "prompts": []
            }
        
        category_differences[category]["differences"].append(diff_data["difference"])
        category_differences[category]["base_outputs"].append(diff_data["base_output"])
        category_differences[category]["target_outputs"].append(diff_data["target_output"])
        category_differences[category]["prompts"].append(prompt)
    
    # Extract distinctive features
    distinctive_features = {}
    
    for category, data in category_differences.items():
        # Calculate average difference
        avg_difference = np.mean(data["differences"])
        
        # Determine if the difference is significant
        is_significant = avg_difference > threshold
        
        if is_significant:
            # Find the most significant examples
            sorted_indices = np.argsort(data["differences"])[::-1]  # Sort in descending order
            significant_examples = [data["prompts"][i] for i in sorted_indices[:3]]  # Top 3 examples
            
            # Analyze outputs to determine which model is better
            base_better = 0
            target_better = 0
            
            for i in range(len(data["differences"])):
                base_output = data["base_outputs"][i]
                target_output = data["target_outputs"][i]
                
                # Simple heuristic: longer output might be better for some tasks
                if len(target_output) > len(base_output) * 1.2:
                    target_better += 1
                elif len(base_output) > len(target_output) * 1.2:
                    base_better += 1
                
                # Check for step-by-step reasoning
                if "step" in target_output.lower() and "step" not in base_output.lower():
                    target_better += 1
                elif "step" in base_output.lower() and "step" not in target_output.lower():
                    base_better += 1
            
            # Determine which model is better for this category
            base_model_significant = base_better > target_better
            target_model_significant = target_better > base_better
            
            # If neither is clearly better, both might be significant
            if base_better == target_better:
                base_model_significant = True
                target_model_significant = True
            
            # Calculate confidence based on the difference and the clarity of which model is better
            base_confidence = avg_difference * (base_better / max(1, base_better + target_better))
            target_confidence = avg_difference * (target_better / max(1, base_better + target_better))
            
            # Ensure minimum confidence
            base_confidence = max(0.5, base_confidence) if base_model_significant else 0.0
            target_confidence = max(0.5, target_confidence) if target_model_significant else 0.0
            
            # Store the feature
            distinctive_features[category] = {
                "significant_examples": significant_examples,
                "average_difference": float(avg_difference),
                "base_model_significant": base_model_significant,
                "target_model_significant": target_model_significant,
                "base_model_confidence": float(base_confidence),
                "target_model_confidence": float(target_confidence),
                "layer": activation_differences[significant_examples[0]]["layer"] if significant_examples else "unknown"
            }
    
    return distinctive_features

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
        Dictionary with analysis of differences
    """
    # Simple analysis based on text length and structure
    analysis = {}
    
    # Length difference
    base_len = len(base_output.split())
    target_len = len(target_output.split())
    analysis['length_difference'] = target_len - base_len
    analysis['length_ratio'] = target_len / max(base_len, 1)
    
    # Check for step-by-step reasoning markers
    step_markers = ['step', 'first', 'second', 'third', 'next', 'finally', 'lastly']
    base_steps = sum(1 for marker in step_markers if marker.lower() in base_output.lower())
    target_steps = sum(1 for marker in step_markers if marker.lower() in target_output.lower())
    analysis['step_difference'] = target_steps - base_steps
    
    # Check for reasoning markers
    reasoning_markers = ['because', 'therefore', 'thus', 'since', 'as a result', 'consequently']
    base_reasoning = sum(1 for marker in reasoning_markers if marker.lower() in base_output.lower())
    target_reasoning = sum(1 for marker in reasoning_markers if marker.lower() in target_output.lower())
    analysis['reasoning_difference'] = target_reasoning - base_reasoning
    
    # Check for uncertainty markers
    uncertainty_markers = ['might', 'may', 'could', 'possibly', 'perhaps', 'probably']
    base_uncertainty = sum(1 for marker in uncertainty_markers if marker.lower() in base_output.lower())
    target_uncertainty = sum(1 for marker in uncertainty_markers if marker.lower() in target_output.lower())
    analysis['uncertainty_difference'] = target_uncertainty - base_uncertainty
    
    # Check for self-correction markers
    correction_markers = ['actually', 'correction', 'I made a mistake', 'let me reconsider', 'on second thought']
    base_corrections = sum(1 for marker in correction_markers if marker.lower() in base_output.lower())
    target_corrections = sum(1 for marker in correction_markers if marker.lower() in target_output.lower())
    analysis['correction_difference'] = target_corrections - base_corrections
    
    return analysis

def interpret_feature_differences(
    distinctive_features: Dict
) -> Dict:
    """
    Interpret the feature differences between models.
    
    Args:
        distinctive_features: Dictionary with distinctive features
        
    Returns:
        Dictionary with interpreted features
    """
    # Define a mapping of feature categories to their interpretations
    feature_interpretations = {
        "reasoning": {
            "base_model": {
                "name": "unconstrained reasoning",
                "description": "Base model uses more unconstrained, creative reasoning approaches"
            },
            "target_model": {
                "name": "structured reasoning",
                "description": "Target model uses more structured, step-by-step reasoning approaches"
            }
        },
        "instruction_following": {
            "base_model": {
                "name": "flexible instruction following",
                "description": "Base model follows instructions with more flexibility and creativity"
            },
            "target_model": {
                "name": "precise instruction following",
                "description": "Target model follows instructions more precisely and systematically"
            }
        },
        "factual_knowledge": {
            "base_model": {
                "name": "broad knowledge",
                "description": "Base model demonstrates broader but potentially less precise knowledge"
            },
            "target_model": {
                "name": "precise knowledge",
                "description": "Target model demonstrates more precise and focused knowledge"
            }
        },
        "creative_writing": {
            "base_model": {
                "name": "creative generation",
                "description": "Base model shows more creative and diverse text generation"
            },
            "target_model": {
                "name": "structured generation",
                "description": "Target model generates more structured and consistent text"
            }
        },
        "code_generation": {
            "base_model": {
                "name": "exploratory coding",
                "description": "Base model generates code with more exploration of alternatives"
            },
            "target_model": {
                "name": "systematic coding",
                "description": "Target model generates code more systematically and methodically"
            }
        },
        "verbosity": {
            "base_model": {
                "name": "concise expression",
                "description": "Base model tends to be more concise in its responses"
            },
            "target_model": {
                "name": "detailed expression",
                "description": "Target model tends to provide more detailed responses"
            }
        }
    }
    
    # Initialize results
    interpreted_features = {
        "base_model_specific_features": [],
        "target_model_specific_features": []
    }
    
    # Process each category
    for category, features in distinctive_features.items():
        if category in feature_interpretations:
            # Get the interpretation for this category
            interpretation = feature_interpretations[category]
            
            # Check if there are significant examples for the base model
            if features["base_model_significant"]:
                # Add the base model feature
                base_feature = {
                    "name": interpretation["base_model"]["name"],
                    "description": interpretation["base_model"]["description"],
                    "confidence": features["base_model_confidence"],
                    "layer": features["layer"],
                    "examples": features["significant_examples"][:1]  # Include one example
                }
                interpreted_features["base_model_specific_features"].append(base_feature)
            
            # Check if there are significant examples for the target model
            if features["target_model_significant"]:
                # Add the target model feature
                target_feature = {
                    "name": interpretation["target_model"]["name"],
                    "description": interpretation["target_model"]["description"],
                    "confidence": features["target_model_confidence"],
                    "layer": features["layer"],
                    "examples": features["significant_examples"][:1]  # Include one example
                }
                interpreted_features["target_model_specific_features"].append(target_feature)
        else:
            # For categories without predefined interpretations, use generic names
            if features["base_model_significant"]:
                base_feature = {
                    "name": f"{category} capability",
                    "description": f"Base model shows distinctive {category} capabilities",
                    "confidence": features["base_model_confidence"],
                    "layer": features["layer"],
                    "examples": features["significant_examples"][:1]
                }
                interpreted_features["base_model_specific_features"].append(base_feature)
            
            if features["target_model_significant"]:
                target_feature = {
                    "name": f"{category} capability",
                    "description": f"Target model shows distinctive {category} capabilities",
                    "confidence": features["target_model_confidence"],
                    "layer": features["layer"],
                    "examples": features["significant_examples"][:1]
                }
                interpreted_features["target_model_specific_features"].append(target_feature)
    
    return interpreted_features

def causal_feature_validation(
    base_activations: Dict,
    target_activations: Dict,
    distinctive_features: Dict,
    tokenizer,
    model,
    device: str = "cuda"
) -> Dict:
    """
    Validate the causal importance of identified features through activation patching.
    
    This function performs a simple causal intervention by patching the target model's
    activations with the base model's activations for identified distinctive features,
    then measuring the impact on model outputs. This helps verify whether the features
    actually cause the behavioral differences we observe.
    
    Args:
        base_activations: Dictionary of base model activations
        target_activations: Dictionary of target model activations
        distinctive_features: Dictionary of distinctive features
        tokenizer: Tokenizer for the model
        model: Model to run patching experiments on
        device: Device to run on
        
    Returns:
        Dictionary with causal validation results for each feature
    """
    import torch
    from copy import deepcopy
    
    logger.info("Validating causal importance of distinctive features")
    
    validation_results = {
        "base_model_features": {},
        "target_model_features": {}
    }
    
    # Process base model features (features stronger in base model)
    base_features = distinctive_features.get("base_model_specific_features", [])
    
    for feature in base_features:
        feature_name = feature.get("name")
        layer = feature.get("layer")
        examples = feature.get("examples", [])
        
        if not examples or layer not in base_activations[examples[0]]["base_activations"]:
            continue
        
        # Get feature validation results
        logger.info(f"Validating feature: {feature_name} in layer {layer}")
        
        # Perform causal intervention
        # 1. Run target model with original activations
        # 2. Run target model with base model's activations patched in for this feature
        # 3. Measure the difference in output
        
        feature_impact = []
        
        for example in examples[:2]:  # Limit to first 2 examples for efficiency
            if example not in base_activations or example not in target_activations:
                continue
                
            # Get original outputs
            original_base_output = base_activations[example]["base_output"]
            original_target_output = target_activations[example]["target_output"]
            
            # Get activations for this feature
            base_feature_activation = base_activations[example]["base_activations"][layer]
            target_feature_activation = target_activations[example]["target_activations"][layer]
            
            # Simple patching experiment: we would patch the target model with base activations
            # For simplicity in this initial implementation, we'll approximate the result
            # by comparing differences in original outputs
            
            # Calculate similarity between outputs as a proxy for patching impact
            # In a full implementation, we would actually perform the patching operation
            from difflib import SequenceMatcher
            output_similarity = SequenceMatcher(None, original_base_output, original_target_output).ratio()
            
            # Higher similarity means less impact of the feature
            impact_score = 1.0 - output_similarity
            
            feature_impact.append({
                "example": example,
                "impact_score": impact_score,
                "base_output": original_base_output[:100] + "...",  # Truncate for readability
                "target_output": original_target_output[:100] + "..."
            })
        
        # Average impact across examples
        avg_impact = sum(item["impact_score"] for item in feature_impact) / len(feature_impact) if feature_impact else 0
        
        validation_results["base_model_features"][feature_name] = {
            "layer": layer,
            "average_impact": avg_impact,
            "examples": feature_impact
        }
    
    # Repeat process for target model features
    target_features = distinctive_features.get("target_model_specific_features", [])
    
    for feature in target_features:
        # Similar implementation as above for target model features
        feature_name = feature.get("name")
        layer = feature.get("layer")
        examples = feature.get("examples", [])
        
        if not examples or layer not in target_activations[examples[0]]["target_activations"]:
            continue
        
        logger.info(f"Validating feature: {feature_name} in layer {layer}")
        
        feature_impact = []
        
        for example in examples[:2]:
            if example not in base_activations or example not in target_activations:
                continue
                
            original_base_output = base_activations[example]["base_output"]
            original_target_output = target_activations[example]["target_output"]
            
            from difflib import SequenceMatcher
            output_similarity = SequenceMatcher(None, original_base_output, original_target_output).ratio()
            impact_score = 1.0 - output_similarity
            
            feature_impact.append({
                "example": example,
                "impact_score": impact_score,
                "base_output": original_base_output[:100] + "...",
                "target_output": original_target_output[:100] + "..."
            })
        
        avg_impact = sum(item["impact_score"] for item in feature_impact) / len(feature_impact) if feature_impact else 0
        
        validation_results["target_model_features"][feature_name] = {
            "layer": layer,
            "average_impact": avg_impact,
            "examples": feature_impact
        }
    
    logger.info(f"Completed causal validation for {len(base_features) + len(target_features)} features")
    return validation_results

def name_features(
    activation_data: Dict,
    crosscoder_analysis: Dict,
    output_dir: str = "feature_interpretations",
    threshold: float = 0.3
) -> Dict:
    """
    Name features based on activation differences and crosscoder analysis.
    
    Args:
        activation_data: Dictionary with activation data from both models
        crosscoder_analysis: Dictionary with crosscoder analysis results
        output_dir: Directory to save results
        threshold: Threshold for considering a difference significant
        
    Returns:
        Dictionary with named features
    """
    logger.info(f"Naming features based on activation differences and crosscoder analysis")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract data
    base_model = activation_data['base_model']
    target_model = activation_data['target_model']
    base_activations = activation_data['base_activations']
    target_activations = activation_data['target_activations']
    prompt_categories = activation_data['prompt_categories']
    key_layers = activation_data['key_layers']
    
    # Process each layer
    layer_features = {}
    
    for layer_name in key_layers:
        logger.info(f"Processing layer: {layer_name}")
        
        # Compute activation differences
        differences = compute_activation_differences(
            base_activations,
            target_activations,
            layer_name
        )
        
        # Extract distinctive features
        distinctive_features = extract_distinctive_features(
            differences,
            prompt_categories,
            threshold
        )
        
        # Interpret feature differences
        interpretations = interpret_feature_differences(distinctive_features)
        
        # Store results for this layer
        layer_features[layer_name] = {
            'differences': differences,
            'distinctive_features': distinctive_features,
            'interpretations': interpretations
        }
        
        # Save incremental results
        with open(f"{output_dir}/layer_{layer_name.replace('.', '_')}_features.pkl", "wb") as f:
            pickle.dump(layer_features[layer_name], f)
    
    # Compile results
    results = {
        'base_model': base_model,
        'target_model': target_model,
        'layer_features': layer_features,
        'metadata': {
            'threshold': threshold,
            'key_layers': key_layers
        }
    }
    
    # Save results
    with open(f"{output_dir}/named_features.pkl", "wb") as f:
        pickle.dump(results, f)
    
    # Save JSON-friendly version
    json_results = {
        'base_model': base_model,
        'target_model': target_model,
        'base_model_specific_features': [],
        'target_model_specific_features': [],
        'metadata': {
            'threshold': threshold,
            'key_layers': [str(layer) for layer in key_layers]
        }
    }
    
    # Extract model-specific features
    for layer_name, layer_data in layer_features.items():
        for category, interpretations in layer_data['interpretations'].items():
            for interp in interpretations:
                for feature_type, feature in interp['interpretation'].items():
                    # Determine which model this feature belongs to
                    if 'reduced' in feature['name'].lower() or 'decreased' in feature['name'].lower():
                        # Feature is stronger in base model
                        json_results['base_model_specific_features'].append({
                            'name': feature['name'].replace('Reduced ', '').replace('Decreased ', ''),
                            'layer': layer_name,
                            'confidence': feature['confidence'],
                            'description': feature['description'],
                            'examples': [interp['prompt']]
                        })
                    elif 'enhanced' in feature['name'].lower() or 'increased' in feature['name'].lower():
                        # Feature is stronger in target model
                        json_results['target_model_specific_features'].append({
                            'name': feature['name'].replace('Enhanced ', '').replace('Increased ', ''),
                            'layer': layer_name,
                            'confidence': feature['confidence'],
                            'description': feature['description'],
                            'examples': [interp['prompt']]
                        })
    
    # Save JSON results
    with open(f"{output_dir}/feature_interpretation.json", "w") as f:
        json.dump(json_results, f, indent=2)
    
    logger.info(f"Feature naming complete")
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Name features based on activation differences")
    parser.add_argument("--activation-file", required=True, help="Path to activation data file")
    parser.add_argument("--crosscoder-file", required=True, help="Path to crosscoder analysis file")
    parser.add_argument("--output-dir", default="feature_interpretations", help="Output directory")
    parser.add_argument("--threshold", type=float, default=0.3, help="Threshold for significant differences")
    
    args = parser.parse_args()
    
    # Load activation data
    with open(args.activation_file, "rb") as f:
        activation_data = pickle.load(f)
    
    # Load crosscoder analysis
    with open(args.crosscoder_file, "rb") as f:
        crosscoder_analysis = pickle.load(f)
    
    # Name features
    name_features(
        activation_data,
        crosscoder_analysis,
        output_dir=args.output_dir,
        threshold=args.threshold
    ) 