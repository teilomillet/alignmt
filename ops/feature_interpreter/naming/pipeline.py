"""
Feature Naming Pipeline Module.

This module provides the main pipeline for naming and interpreting features
by analyzing activation differences between base and target models.
"""

import os
import json
import pickle
import logging
from typing import Dict, Optional
import time

from .differences import compute_activation_differences, analyze_output_differences
from .extraction import extract_distinctive_features
from .interpretation import interpret_feature_differences
from .validation import causal_feature_validation

# Configure logging
logger = logging.getLogger(__name__)

def name_features(
    activation_data: Dict,
    crosscoder_analysis: Optional[Dict] = None,
    output_dir: str = "./output",
    difference_threshold: float = 0.1,
    min_prompts: int = 5,
    validation_threshold: float = 0.7
) -> Dict:
    """
    Main pipeline for naming features based on activation differences.
    
    Args:
        activation_data: Dictionary with base and target model activations
        crosscoder_analysis: Optional dictionary with crosscoder analysis
        output_dir: Directory to save results
        difference_threshold: Threshold for considering activation differences significant
        min_prompts: Minimum number of prompts needed for a distinctive feature
        validation_threshold: Threshold for causal validation
        
    Returns:
        Dictionary with named features and their characteristics
    """
    start_time = time.time()
    logger.info("Starting feature naming pipeline")
    
    # Make sure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Check for required activation data
    if 'base' not in activation_data or 'target' not in activation_data:
        logger.error("Missing base or target activation data")
        return {}
    
    base_activations = activation_data['base']
    target_activations = activation_data['target']
    
    # Process each layer where activations are available
    all_layers = set()
    for prompt in base_activations:
        if 'activations' in base_activations[prompt]:
            all_layers.update(base_activations[prompt]['activations'].keys())
    
    logger.info(f"Found {len(all_layers)} layers to analyze")
    
    # Initialize results
    activation_differences_by_layer = {}
    output_analyses = {}
    distinctive_features = {}
    feature_interpretations = {}
    
    # Step 1: Compute activation differences for each layer
    for layer_name in all_layers:
        logger.info(f"Processing layer: {layer_name}")
        
        # Compute activation differences
        differences = compute_activation_differences(
            base_activations, 
            target_activations, 
            layer_name
        )
        
        if differences:
            activation_differences_by_layer[layer_name] = differences
            
            # Analyze output differences for this layer's prompts
            for prompt, diff_data in differences.items():
                if prompt not in output_analyses:
                    output_analyses[prompt] = analyze_output_differences(
                        diff_data.get('base_output', ''),
                        diff_data.get('target_output', '')
                    )
    
    # Step 2: Extract distinctive features
    for layer_name, differences in activation_differences_by_layer.items():
        feature_info = extract_distinctive_features(
            differences,
            threshold=difference_threshold,
            min_prompts=min_prompts
        )
        
        if feature_info.get('is_distinctive', False):
            feature_id = f"feature_{layer_name}"
            distinctive_features[feature_id] = feature_info
    
    # Step 3: Interpret features
    for feature_id, feature_info in distinctive_features.items():
        layer_name = feature_info['layer']
        
        if layer_name in activation_differences_by_layer:
            interpretation = interpret_feature_differences(
                activation_differences_by_layer[layer_name],
                output_analyses,
                threshold=difference_threshold
            )
            
            # Add the original feature info to the interpretation
            interpretation.update(feature_info)
            feature_interpretations[feature_id] = interpretation
    
    # Step 4: Validate features
    validation_results = causal_feature_validation(
        feature_interpretations,
        crosscoder_analysis,
        validation_threshold=validation_threshold
    )
    
    # Step 5: Assemble final results
    named_features = {}
    
    for feature_id, interpretation in feature_interpretations.items():
        # Check if this feature was validated
        is_validated = False
        validation_info = {}
        
        if feature_id in validation_results:
            validation_info = validation_results[feature_id]
            is_validated = validation_info.get('is_validated', False)
        
        # Generate a name for the feature
        primary_pattern = interpretation.get('primary_pattern', 'unknown')
        layer_name = interpretation.get('layer', 'unknown')
        
        feature_name = f"{primary_pattern}_{layer_name}"
        
        # Create the named feature entry
        named_features[feature_id] = {
            'name': feature_name,
            'layer': layer_name,
            'description': interpretation.get('description', ''),
            'is_validated': is_validated,
            'validation': validation_info,
            'interpretation': interpretation
        }
    
    # Prepare the final results dictionary
    results = {
        'named_features': named_features,
        'processed_layers': len(all_layers),
        'distinctive_features_count': len(distinctive_features),
        'validated_features_count': sum(1 for f in named_features.values() if f.get('is_validated', False)),
        'metadata': {
            'difference_threshold': difference_threshold,
            'min_prompts': min_prompts,
            'validation_threshold': validation_threshold,
            'processing_time': time.time() - start_time
        }
    }
    
    # Save results to disk
    output_file_base = os.path.join(output_dir, 'named_features')
    
    # Save as JSON for human readability
    with open(f"{output_file_base}.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save as pickle for preserving object types
    with open(f"{output_file_base}.pkl", 'wb') as f:
        pickle.dump(results, f)
    
    logger.info(f"Feature naming complete. Found {len(named_features)} named features " +
                f"({results['validated_features_count']} validated)")
    logger.info(f"Results saved to {output_file_base}.json and {output_file_base}.pkl")
    
    return results 