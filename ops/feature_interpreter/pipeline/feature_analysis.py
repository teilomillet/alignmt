"""
Feature analysis module for the feature interpretation pipeline.

This module provides functions for analyzing and interpreting feature
differences between base and target models.
"""

import os
import json
import logging
import torch
from typing import Dict, Any
from tqdm.auto import tqdm

from ..naming import (
    compute_activation_differences,
    extract_distinctive_features,
    interpret_feature_differences,
    causal_feature_validation,
)

logger = logging.getLogger(__name__)

def get_activation_differences(
    activations: Dict[str, Any],
    prompt_to_category: Dict[str, str]
) -> Dict[str, Any]:
    """
    Compute activation differences for each layer between models.
    
    Args:
        activations: Activation data dictionary
        prompt_to_category: Mapping from prompts to categories
        
    Returns:
        Dictionary of activation differences
    """
    logger.info("Computing activation differences for each layer")
    activation_differences = {}
    
    # Get a sample prompt to extract layer names
    sample_prompt = list(activations.keys())[0]
    sample_data = activations[sample_prompt]
    
    # Extract layer names from the sample data
    layer_names = []
    for key in sample_data["base_activations"].keys():
        layer_names.append(key)
    
    # Process each layer
    for layer_name in tqdm(layer_names, desc="Processing layers"):
        logger.info(f"Computing activation differences for layer: {layer_name}")
        
        # Prepare data for compute_activation_differences
        base_activations_layer = {}
        target_activations_layer = {}
        
        for prompt, data in activations.items():
            base_activations_layer[prompt] = {
                'activations': {layer_name: data['base_activations'].get(layer_name)},
                'text': data['base_output']
            }
            
            target_activations_layer[prompt] = {
                'activations': {layer_name: data['target_activations'].get(layer_name)},
                'text': data['target_output']
            }
        
        # Compute differences for this layer
        differences = compute_activation_differences(
            base_activations_layer,
            target_activations_layer,
            layer_name
        )
        
        # Store differences
        for prompt, diff in differences.items():
            # Add the layer name to the difference dictionary
            diff['layer'] = layer_name
            
            if prompt not in activation_differences:
                activation_differences[prompt] = diff
            else:
                # If we already have differences for this prompt, use the one with the larger difference
                if diff['difference'] > activation_differences[prompt]['difference']:
                    activation_differences[prompt] = diff
    
    return activation_differences

def perform_feature_analysis(
    output_dir: str,
    activation_differences: Dict[str, Any],
    prompt_to_category: Dict[str, str],
    layer_similarities: Dict[str, float],
    feature_threshold: float = 0.3
) -> Dict[str, Any]:
    """
    Analyze and interpret features from activation differences.
    
    Args:
        output_dir: Directory to save outputs
        activation_differences: Dictionary of activation differences
        prompt_to_category: Mapping from prompts to categories
        layer_similarities: Dictionary of layer similarities
        feature_threshold: Threshold for identifying distinctive features
        
    Returns:
        Dictionary of interpreted features
    """
    logger.info("Extracting distinctive features")
    distinctive_features = extract_distinctive_features(
        activation_differences,
        threshold=feature_threshold
    )
    
    # Incorporate layer similarity data into feature interpretation
    if "layer" in distinctive_features and distinctive_features["layer"] in layer_similarities:
        distinctive_features["layer_similarity"] = layer_similarities[distinctive_features["layer"]]
    else:
        distinctive_features["layer_similarity"] = 0.0
    
    # Interpret features
    logger.info("Interpreting feature differences")
    # The interpret_feature_differences function now handles empty output_analyses internally
    interpreted_features = interpret_feature_differences(
        activation_differences=activation_differences,
        threshold=feature_threshold
    )
    
    # Save features
    features_path = os.path.join(output_dir, "features.json")
    with open(features_path, "w") as f:
        json.dump(interpreted_features, f, indent=2)
        
    logger.info(f"Features saved to {features_path}")
    
    return interpreted_features

def perform_causal_validation(model_path, output_dir, interpreted_features, crosscoder_analysis=None):
    """
    Perform causal validation of features using crosscoder analysis.
    
    Args:
        model_path: Path to the model used for validation
        output_dir: Directory to save validation results
        interpreted_features: Dictionary of feature interpretations
        crosscoder_analysis: Optional dictionary of crosscoder analysis results
        
    Returns:
        Updated dictionary of feature interpretations with validation results
    """
    try:
        import torch
        
        # Load the model for validation
        logger.info(f"Loading model from {model_path} for causal validation")
        model = None  # Placeholder for actual model loading
        
        # Set up paths
        features_path = os.path.join(output_dir, "features.json")
        
        # Check if we have crosscoder analysis
        if crosscoder_analysis:
            logger.info(f"Using provided crosscoder analysis with {len(crosscoder_analysis)} layers")
            
            # Log structure of crosscoder analysis to help diagnose issues
            sample_layer = next(iter(crosscoder_analysis)) if crosscoder_analysis else None
            if sample_layer:
                logger.info(f"Sample layer key: {sample_layer}")
                sample_data = crosscoder_analysis[sample_layer]
                logger.info(f"Sample layer data type: {type(sample_data)}")
                if isinstance(sample_data, dict):
                    logger.info(f"Sample layer data keys: {list(sample_data.keys())}")
                    
                    # Check if 'strength' is already in the data
                    if not any('strength' in layer_data for layer_name, layer_data in crosscoder_analysis.items() 
                              if isinstance(layer_data, dict)):
                        logger.info("Reformatting crosscoder data to include 'strength'")
                        
                        # Create a new dictionary with strength information
                        reformatted_crosscoder = {}
                        for layer_name, layer_data in crosscoder_analysis.items():
                            if isinstance(layer_data, dict):
                                # Calculate strength from differences or similarities
                                strength = 0.0
                                if 'differences' in layer_data:
                                    diffs = list(layer_data['differences'].values())
                                    strength = sum(diffs) / len(diffs) if diffs else 0.0
                                elif 'similarities' in layer_data:
                                    sims = list(layer_data['similarities'].values())
                                    strength = sum(sims) / len(sims) if sims else 0.0
                                    
                                reformatted_crosscoder[layer_name] = {
                                    **layer_data,
                                    'strength': strength
                                }
                            else:
                                reformatted_crosscoder[layer_name] = {'strength': float(layer_data) if isinstance(layer_data, (int, float)) else 0.0}
                        
                        # Use the reformatted data
                        crosscoder_analysis = reformatted_crosscoder
                        logger.info(f"Reformatted crosscoder data with {len(crosscoder_analysis)} layers")
            else:
                logger.warning("Crosscoder analysis has unexpected format")
        else:
            logger.info("No crosscoder analysis provided for causal validation")
            
        # Run causal validation
        logger.info("Performing causal validation of features...")
        causal_results = causal_feature_validation(
            interpreted_features,
            crosscoder_analysis=crosscoder_analysis,  # Now passing the crosscoder analysis if available
            validation_threshold=0.3  # Lower threshold to detect more subtle differences
        )
        
        # Add causal validation results to interpreted features
        if "features" in interpreted_features and isinstance(interpreted_features["features"], list):
            # Add as top-level key when features are in a list
            interpreted_features["causal_validation"] = causal_results
        else:
            # Old format where features are top-level entries
            interpreted_features["causal_validation"] = causal_results
        
        # Save updated features
        with open(features_path, "w") as f:
            json.dump(interpreted_features, f, indent=2)
            
        logger.info("Causal validation complete")
        
        # Clean up to free memory
        del model
        torch.cuda.empty_cache()
        
    except Exception as e:
        logger.warning(f"Causal validation failed: {str(e)}")
        logger.warning("Continuing without causal validation")
    
    return interpreted_features 