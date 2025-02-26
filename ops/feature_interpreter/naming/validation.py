"""
Feature Validation Module.

This module provides functions to validate the causal relationship
between identified features and model behavior differences.
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Union, Any

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
    
    # If no crosscoder analysis is available, return basic validation
    if not crosscoder_analysis:
        logger.warning("No crosscoder analysis provided for causal validation")
        for feature_id, interpretation in feature_interpretations.items():
            validation_results[feature_id] = {
                'is_validated': False,
                'validation_score': 0.0,
                'reason': "No crosscoder analysis available"
            }
        return validation_results
    
    # Process each feature for validation
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
        
        # Check if this feature's layer is in the crosscoder analysis
        if layer_name not in crosscoder_analysis:
            validation_results[feature_id] = {
                'is_validated': False,
                'validation_score': 0.0,
                'reason': f"Layer {layer_name} not found in crosscoder analysis"
            }
            continue
        
        # Extract crosscoder connection strength
        layer_crosscoding = crosscoder_analysis[layer_name]
        
        # Calculate a validation score based on crosscoder strength
        # Higher crosscoder values indicate stronger causal relationship
        validation_score = 0.0
        
        if isinstance(layer_crosscoding, dict) and 'strength' in layer_crosscoding:
            validation_score = layer_crosscoding['strength']
        elif isinstance(layer_crosscoding, (int, float)):
            validation_score = float(layer_crosscoding)
        elif isinstance(layer_crosscoding, np.ndarray):
            validation_score = float(np.mean(layer_crosscoding))
        
        # Determine if feature is causally validated
        is_validated = validation_score >= validation_threshold
        
        # Create validation result
        validation_results[feature_id] = {
            'is_validated': is_validated,
            'validation_score': validation_score,
            'threshold': validation_threshold,
            'layer': layer_name,
            'reason': f"Validation {'passed' if is_validated else 'failed'} with score {validation_score:.2f}"
        }
        
        logger.info(f"Feature {feature_id} validation: {validation_results[feature_id]['reason']}")
    
    return validation_results 