"""
Feature analysis visualization module.

This module provides functions for analyzing feature relationships
and categorizing features for visualization.
"""

import matplotlib.pyplot as plt
import numpy as np
import logging
import os
import json
from typing import Dict, Optional

# Configure logging
logger = logging.getLogger(__name__)

def categorize_features_by_norm(
    feature_data: Dict,
    norm_ratio_threshold: float = 1.5,
    output_path: Optional[str] = None
) -> Dict:
    """
    Categorize features into base-specific, target-specific, and shared features
    based on the relative norms of feature decoder weights.
    
    Args:
        feature_data: Dictionary with feature data including norms
        norm_ratio_threshold: Threshold for norm ratio to categorize features
        output_path: Optional path to save the categorization results
        
    Returns:
        Dictionary with categorized features
    """
    logger.info("Categorizing features based on decoder norms")
    
    # Check if the feature data contains decoder norms
    if "feature_norms" not in feature_data:
        logger.warning("Feature norms not found in data. Cannot categorize features.")
        return {
            "base_specific": [],
            "target_specific": [],
            "shared": []
        }
    
    feature_norms = feature_data["feature_norms"]
    categorized_features = {
        "base_specific": [],
        "target_specific": [],
        "shared": []
    }
    
    # Categorize features based on norm ratio
    for feature_id, norms in feature_norms.items():
        if "base_norm" not in norms or "target_norm" not in norms:
            logger.warning(f"Incomplete norm data for feature {feature_id}")
            continue
            
        base_norm = norms["base_norm"]
        target_norm = norms["target_norm"]
        
        # To avoid division by zero
        if base_norm < 1e-6 and target_norm < 1e-6:
            # Both norms are effectively zero, skip this feature
            continue
        elif base_norm < 1e-6:
            # Only base norm is zero, this is a target-specific feature
            categorized_features["target_specific"].append(feature_id)
        elif target_norm < 1e-6:
            # Only target norm is zero, this is a base-specific feature
            categorized_features["base_specific"].append(feature_id)
        else:
            # Calculate norm ratio
            base_to_target_ratio = base_norm / target_norm
            target_to_base_ratio = target_norm / base_norm
            
            if base_to_target_ratio >= norm_ratio_threshold:
                categorized_features["base_specific"].append(feature_id)
            elif target_to_base_ratio >= norm_ratio_threshold:
                categorized_features["target_specific"].append(feature_id)
            else:
                categorized_features["shared"].append(feature_id)
    
    # Log the results
    logger.info(f"Categorized {len(categorized_features['base_specific'])} base-specific features")
    logger.info(f"Categorized {len(categorized_features['target_specific'])} target-specific features")
    logger.info(f"Categorized {len(categorized_features['shared'])} shared features")
    
    # Save results if output path is provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(categorized_features, f, indent=2)
        logger.info(f"Saved categorized features to {output_path}")
    
    return categorized_features


def calculate_feature_alignment(
    feature_data: Dict,
    output_path: Optional[str] = None
) -> Dict:
    """
    Calculate alignment/correlation between shared features' decoder vectors
    in the base and target models.
    
    Args:
        feature_data: Dictionary with feature data including decoder vectors
        output_path: Optional path to save the alignment results plot
        
    Returns:
        Dictionary with alignment scores
    """
    logger.info("Calculating alignment between shared features' decoder vectors")
    
    # Check if the feature data contains decoder vectors
    if "feature_decoders" not in feature_data:
        logger.warning("Feature decoders not found in data. Cannot calculate alignment.")
        return {"alignment_scores": {}}
    
    feature_decoders = feature_data["feature_decoders"]
    alignment_scores = {}
    
    # Calculate alignment (cosine similarity) between decoder vectors
    for feature_id, decoders in feature_decoders.items():
        if "base_decoder" not in decoders or "target_decoder" not in decoders:
            logger.warning(f"Incomplete decoder data for feature {feature_id}")
            continue
            
        base_decoder = np.array(decoders["base_decoder"])
        target_decoder = np.array(decoders["target_decoder"])
        
        # Calculate cosine similarity
        norm_base = np.linalg.norm(base_decoder)
        norm_target = np.linalg.norm(target_decoder)
        
        # Avoid division by zero
        if norm_base < 1e-6 or norm_target < 1e-6:
            alignment = 0.0
        else:
            alignment = np.dot(base_decoder, target_decoder) / (norm_base * norm_target)
            
        alignment_scores[feature_id] = alignment
    
    # Create visualization if output path is provided
    if output_path and alignment_scores:
        plt.figure(figsize=(10, 6))
        alignments = list(alignment_scores.values())
        
        # Create histogram of alignment scores
        plt.hist(alignments, bins=50, alpha=0.75)
        plt.axvline(x=0, color='r', linestyle='--', alpha=0.5)
        
        # Add labels and title
        plt.xlabel('Alignment Score (Cosine Similarity)')
        plt.ylabel('Number of Features')
        plt.title('Distribution of Decoder Vector Alignment Between Models')
        
        # Add summary statistics
        neg_aligned = sum(1 for a in alignments if a < 0)
        low_aligned = sum(1 for a in alignments if 0 <= a < 0.5)
        high_aligned = sum(1 for a in alignments if 0.5 <= a <= 1.0)
        
        plt.figtext(0.15, 0.85, 
                   f"Negatively aligned: {neg_aligned} ({neg_aligned/len(alignments):.1%})\n"
                   f"Low alignment (0-0.5): {low_aligned} ({low_aligned/len(alignments):.1%})\n"
                   f"High alignment (0.5-1.0): {high_aligned} ({high_aligned/len(alignments):.1%})",
                   bbox=dict(facecolor='white', alpha=0.8))
        
        # Save figure
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
        
        logger.info(f"Saved decoder alignment visualization to {output_path}")
    
    return {"alignment_scores": alignment_scores} 