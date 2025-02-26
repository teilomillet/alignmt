"""
Feature Activation Analysis Module.

This module provides functionality for analyzing which features are
activated in base and target models.
"""

import numpy as np
import logging
from typing import Dict, Optional

# Configure logging
logger = logging.getLogger(__name__)

def identify_active_features(
    activations: Dict,
    threshold: float = 0.1,
    percentile_threshold: Optional[float] = 95
) -> Dict:
    """
    Identify which features are activated in each model based on 
    activation magnitudes.
    
    Args:
        activations: Dictionary with activation data for both models
        threshold: Absolute threshold for considering a feature as active
        percentile_threshold: Alternative percentile-based threshold
        
    Returns:
        Dictionary with active features in each model and overlap statistics
    """
    logger.info("Identifying active features in each model")
    
    if "base_activations" not in activations or "target_activations" not in activations:
        raise ValueError("Activations dictionary must contain 'base_activations' and 'target_activations'")
    
    base_acts = activations["base_activations"]
    target_acts = activations["target_activations"]
    
    # Ensure activations have compatible shapes
    if base_acts.shape != target_acts.shape:
        raise ValueError(f"Activation shapes don't match: {base_acts.shape} vs {target_acts.shape}")
    
    # Determine dynamic threshold if percentile_threshold is provided
    if percentile_threshold is not None:
        base_threshold = np.percentile(np.abs(base_acts), percentile_threshold)
        target_threshold = np.percentile(np.abs(target_acts), percentile_threshold)
        base_mask = np.max(np.abs(base_acts), axis=1) > base_threshold
        target_mask = np.max(np.abs(target_acts), axis=1) > target_threshold
    else:
        base_mask = np.max(np.abs(base_acts), axis=1) > threshold
        target_mask = np.max(np.abs(target_acts), axis=1) > threshold
    
    # Get indices of active features
    base_active = np.where(base_mask)[0].tolist()
    target_active = np.where(target_mask)[0].tolist()
    
    # Find overlap
    active_overlap = set(base_active).intersection(set(target_active))
    base_specific = set(base_active) - active_overlap
    target_specific = set(target_active) - active_overlap
    
    # Prepare output dictionary
    active_features = {
        "base_active": base_active,
        "target_active": target_active,
        "active_in_both": list(active_overlap),
        "base_specific_active": list(base_specific),
        "target_specific_active": list(target_specific),
        "stats": {
            "total_features": base_acts.shape[0],
            "base_active_count": len(base_active),
            "target_active_count": len(target_active),
            "overlap_count": len(active_overlap),
            "base_only_count": len(base_specific),
            "target_only_count": len(target_specific)
        }
    }
    
    return active_features

def calculate_feature_alignment(
    feature_data: Dict,
    output_path: Optional[str] = None
) -> Dict:
    """
    Calculate alignment between feature decoder weights in base and target models.
    
    Args:
        feature_data: Dictionary with feature data
        output_path: Optional path to save the alignment visualization
        
    Returns:
        Dictionary with alignment scores
    """
    logger.info("Calculating feature alignment")
    
    if "feature_decoders" not in feature_data:
        logger.warning("Feature decoder data not found. Cannot calculate alignment.")
        return {"alignment_scores": {}}
    
    feature_decoders = feature_data["feature_decoders"]
    alignment_scores = {}
    
    for feature_id, decoders in feature_decoders.items():
        if "base_decoder" not in decoders or "target_decoder" not in decoders:
            logger.warning(f"Incomplete decoder data for feature {feature_id}")
            continue
        
        base_decoder = np.array(decoders["base_decoder"])
        target_decoder = np.array(decoders["target_decoder"])
        
        # Calculate cosine similarity
        base_norm = np.linalg.norm(base_decoder)
        target_norm = np.linalg.norm(target_decoder)
        
        if base_norm > 0 and target_norm > 0:
            alignment = np.dot(base_decoder, target_decoder) / (base_norm * target_norm)
            alignment_scores[feature_id] = float(alignment)
        else:
            alignment_scores[feature_id] = 0.0
    
    # Calculate average alignment
    if alignment_scores:
        avg_alignment = sum(alignment_scores.values()) / len(alignment_scores)
        logger.info(f"Average feature alignment: {avg_alignment:.4f}")
    else:
        avg_alignment = 0.0
        logger.warning("No valid features for alignment calculation")
    
    # Create visualization if output path is provided
    if output_path and alignment_scores:
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Create sorted list of alignment scores
            sorted_scores = sorted(alignment_scores.values())
            
            # Plot alignment distribution
            plt.figure(figsize=(10, 6))
            sns.histplot(sorted_scores, kde=True)
            plt.axvline(avg_alignment, color='r', linestyle='--', label=f'Average: {avg_alignment:.4f}')
            plt.xlabel('Alignment Score (Cosine Similarity)')
            plt.ylabel('Frequency')
            plt.title('Distribution of Feature Alignment Scores')
            plt.legend()
            plt.savefig(output_path, dpi=300)
            plt.close()
            
            logger.info(f"Saved alignment visualization to {output_path}")
        except Exception as e:
            logger.error(f"Failed to create alignment visualization: {e}")
    
    return {
        "alignment_scores": alignment_scores,
        "average_alignment": avg_alignment
    } 