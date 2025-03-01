"""
Feature Validation Module.

This module provides functions to validate the causal relationship
between identified features and model behavior differences.
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
import re

# Configure logging
logger = logging.getLogger(__name__)

def causal_feature_validation(
    feature_interpretations: Dict,
    crosscoder_analysis: Optional[Dict] = None,
    validation_threshold: float = 0.7
) -> Dict:
    """
    Validate the causal relationship of features using crosscoder analysis.
    
    Args:
        feature_interpretations: Dictionary of feature interpretations
        crosscoder_analysis: Optional dictionary of crosscoder analysis results
        validation_threshold: Threshold for considering a feature validated
        
    Returns:
        Dictionary of validation results for each feature
    """
    logger.info("Performing causal validation of features")
    
    validation_results = {}
    
    # Function to normalize layer names for matching
    def normalize_layer_name(name):
        # Handle various layer name formats
        if not name:
            return name
            
        # Try to extract just the layer number for more flexible matching
        match = re.search(r'layers?.(\d+)', name)
        if match:
            layer_num = match.group(1)
            # Return multiple possible formats for matching
            return [
                f"model.layers.{layer_num}",
                f"layers.{layer_num}",
                f"layer.{layer_num}",
                layer_num,
                name  # Also include original name
            ]
        return [name]  # Return original name if no pattern match
    
    # Function to find a layer in crosscoder analysis with flexible matching
    def find_layer_in_crosscoder(layer_name, crosscoder_data):
        if not crosscoder_data:
            return None
            
        # Try direct match first
        if layer_name in crosscoder_data:
            return layer_name
            
        # Try normalized matches
        normalized_names = normalize_layer_name(layer_name)
        for name in normalized_names:
            if name in crosscoder_data:
                logger.info(f"Found match for layer {layer_name} as {name} in crosscoder data")
                return name
        
        # Try prefix matching - this handles cases where layer_name is a prefix of crosscoder layer names
        # Example: layer_name = "model.layers.0" should match "model.layers.0.mlp.gate_proj.weight"
        prefix_matches = []
        for key in crosscoder_data.keys():
            if key.startswith(layer_name) or any(key.startswith(norm_name) for norm_name in normalized_names):
                prefix_matches.append(key)
                
        if prefix_matches:
            # Choose the shortest match as it's likely the most relevant
            best_match = min(prefix_matches, key=len)
            logger.info(f"Found prefix match for layer {layer_name} as {best_match} in crosscoder data")
            return best_match
                
        # Log available keys for debugging
        available_keys = list(crosscoder_data.keys())
        sample_keys = available_keys[:5] if len(available_keys) > 5 else available_keys
        logger.warning(f"Layer {layer_name} not found in crosscoder data. Available keys sample: {sample_keys}")
        return None
    
    # If no crosscoder analysis is available, return basic validation
    if not crosscoder_analysis:
        logger.warning("No crosscoder analysis provided for causal validation")
        if "features" in feature_interpretations and isinstance(feature_interpretations["features"], list):
            # Handle the case where features are in a list under the "features" key
            for feature in feature_interpretations["features"]:
                feature_id = feature.get("name", "unknown_feature")
                validation_results[feature_id] = {
                    'is_validated': False,
                    'validation_score': 0.0,
                    'reason': "No crosscoder analysis available"
                }
        else:
            # Handle the case where feature_interpretations is a dict of feature_id -> interpretation
            for feature_id, interpretation in feature_interpretations.items():
                validation_results[feature_id] = {
                    'is_validated': False,
                    'validation_score': 0.0,
                    'reason': "No crosscoder analysis available"
                }
        return validation_results
    
    # Process features based on their structure
    if "features" in feature_interpretations and isinstance(feature_interpretations["features"], list):
        # Handle the case where features are in a list under the "features" key
        for feature in feature_interpretations["features"]:
            feature_id = feature.get("name", "unknown_feature")
            layer_name = feature.get('layer', '')
            
            # Skip if layer information is missing
            if not layer_name:
                validation_results[feature_id] = {
                    'is_validated': False,
                    'validation_score': 0.0,
                    'reason': "Missing layer information"
                }
                continue
            
            # Find matching layer with flexible name matching
            matched_layer = find_layer_in_crosscoder(layer_name, crosscoder_analysis)
            
            # Check if this feature's layer is in the crosscoder analysis
            if not matched_layer:
                validation_results[feature_id] = {
                    'is_validated': False,
                    'validation_score': 0.0,
                    'reason': f"Layer {layer_name} not found in crosscoder analysis"
                }
                continue
            
            # Extract crosscoder connection strength
            layer_crosscoding = crosscoder_analysis[matched_layer]
            
            # Calculate a validation score based on crosscoder strength
            validation_score = 0.0
            
            if isinstance(layer_crosscoding, dict) and 'strength' in layer_crosscoding:
                validation_score = layer_crosscoding['strength']
            elif isinstance(layer_crosscoding, dict) and 'differences' in layer_crosscoding:
                # If we have differences, use the mean of differences
                differences = list(layer_crosscoding['differences'].values())
                validation_score = sum(differences) / len(differences) if differences else 0.0
            elif isinstance(layer_crosscoding, (int, float)):
                validation_score = float(layer_crosscoding)
            elif isinstance(layer_crosscoding, np.ndarray):
                validation_score = float(np.mean(layer_crosscoding))
            
            # Determine if feature is causally validated
            is_validated = validation_score >= validation_threshold
            
            # Determine model attribution based on validation score
            # Higher validation scores indicate stronger effect in target model
            model_attribution = "target" if validation_score >= 0.5 else "base"
            
            # Create validation result
            validation_results[feature_id] = {
                'is_validated': is_validated,
                'validation_score': validation_score,
                'threshold': validation_threshold,
                'layer': layer_name,
                'matched_layer': matched_layer,
                'model_attribution': model_attribution,
                'reason': f"Validation {'passed' if is_validated else 'failed'} with score {validation_score:.2f}"
            }
            
            # If the feature doesn't already have model_attribution, add it
            if "features" in feature_interpretations and isinstance(feature_interpretations["features"], list):
                for feature in feature_interpretations["features"]:
                    if feature.get("name") == feature_id and "model_attribution" not in feature:
                        feature["model_attribution"] = model_attribution
                        logger.info(f"Added model_attribution '{model_attribution}' to feature {feature_id} based on validation score {validation_score:.2f}")
            
            logger.info(f"Feature {feature_id} validation: {validation_results[feature_id]['reason']}")
    else:
        # Process each feature in the original dictionary format
        for feature_id, interpretation in feature_interpretations.items():
            layer_name = interpretation.get('layer', '')
            
            # Skip if layer information is missing
            if not layer_name:
                validation_results[feature_id] = {
                    'is_validated': False,
                    'validation_score': 0.0,
                    'reason': "Missing layer information"
                }
                continue
            
            # Find matching layer with flexible name matching
            matched_layer = find_layer_in_crosscoder(layer_name, crosscoder_analysis)
            
            # Check if this feature's layer is in the crosscoder analysis
            if not matched_layer:
                validation_results[feature_id] = {
                    'is_validated': False,
                    'validation_score': 0.0,
                    'reason': f"Layer {layer_name} not found in crosscoder analysis"
                }
                continue
            
            # Extract crosscoder connection strength
            layer_crosscoding = crosscoder_analysis[matched_layer]
            
            # Calculate a validation score based on crosscoder strength
            # Higher crosscoder values indicate stronger causal relationship
            validation_score = 0.0
            
            if isinstance(layer_crosscoding, dict) and 'strength' in layer_crosscoding:
                validation_score = layer_crosscoding['strength']
            elif isinstance(layer_crosscoding, dict) and 'differences' in layer_crosscoding:
                # If we have differences, use the mean of differences
                differences = list(layer_crosscoding['differences'].values())
                validation_score = sum(differences) / len(differences) if differences else 0.0
            elif isinstance(layer_crosscoding, (int, float)):
                validation_score = float(layer_crosscoding)
            elif isinstance(layer_crosscoding, np.ndarray):
                validation_score = float(np.mean(layer_crosscoding))
            
            # Determine if feature is causally validated
            is_validated = validation_score >= validation_threshold
            
            # Determine model attribution based on validation score
            # Higher validation scores indicate stronger effect in target model
            model_attribution = "target" if validation_score >= 0.5 else "base"
            
            # Create validation result
            validation_results[feature_id] = {
                'is_validated': is_validated,
                'validation_score': validation_score,
                'threshold': validation_threshold,
                'layer': layer_name,
                'matched_layer': matched_layer,
                'model_attribution': model_attribution,
                'reason': f"Validation {'passed' if is_validated else 'failed'} with score {validation_score:.2f}"
            }
            
            # If the feature doesn't already have model_attribution, add it
            if "features" in feature_interpretations and isinstance(feature_interpretations["features"], list):
                for feature in feature_interpretations["features"]:
                    if feature.get("name") == feature_id and "model_attribution" not in feature:
                        feature["model_attribution"] = model_attribution
                        logger.info(f"Added model_attribution '{model_attribution}' to feature {feature_id} based on validation score {validation_score:.2f}")
            
            logger.info(f"Feature {feature_id} validation: {validation_results[feature_id]['reason']}")
    
    return validation_results 