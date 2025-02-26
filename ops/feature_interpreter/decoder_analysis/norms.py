"""
Decoder Weight Norms Analysis Module.

This module provides functionality for analyzing the norms of feature decoder weights
between base and target models.
"""

import numpy as np
import logging
import json
import os
from typing import Dict, List, Optional

# Configure logging
logger = logging.getLogger(__name__)

def extract_feature_decoder_norms(
    base_decoder: np.ndarray,
    target_decoder: np.ndarray,
    feature_names: Optional[List[str]] = None
) -> Dict:
    """
    Extract and compare the norms of feature decoder weights in both models.
    
    Args:
        base_decoder: Base model decoder weights matrix
        target_decoder: Target model decoder weights matrix
        feature_names: Optional list of feature names
        
    Returns:
        Dictionary with feature norms and ratios
    """
    logger.info("Extracting feature decoder norms")
    
    # Ensure inputs have compatible shapes
    if base_decoder.shape != target_decoder.shape:
        raise ValueError(f"Decoder shapes don't match: {base_decoder.shape} vs {target_decoder.shape}")
    
    num_features = base_decoder.shape[0]
    
    # Create default feature names if not provided
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(num_features)]
    elif len(feature_names) != num_features:
        raise ValueError(f"Number of feature names ({len(feature_names)}) doesn't match number of features ({num_features})")
    
    # Calculate norms for each feature
    base_norms = np.linalg.norm(base_decoder, axis=1)
    target_norms = np.linalg.norm(target_decoder, axis=1)
    
    # Calculate norm ratios (target/base)
    norm_ratios = np.zeros_like(base_norms)
    # Avoid division by zero
    non_zero_mask = base_norms > 1e-6
    norm_ratios[non_zero_mask] = target_norms[non_zero_mask] / base_norms[non_zero_mask]
    # Set ratio to a large value when base_norm is close to zero
    norm_ratios[~non_zero_mask] = np.inf if target_norms[~non_zero_mask].any() > 1e-6 else 0
    
    # Prepare output dictionary
    feature_data = {
        "feature_norms": {},
        "feature_decoders": {}
    }
    
    for i in range(num_features):
        feature_id = feature_names[i]
        feature_data["feature_norms"][feature_id] = {
            "base_norm": float(base_norms[i]),
            "target_norm": float(target_norms[i]),
            "norm_ratio": float(norm_ratios[i])
        }
        feature_data["feature_decoders"][feature_id] = {
            "base_decoder": base_decoder[i].tolist(),
            "target_decoder": target_decoder[i].tolist()
        }
    
    return feature_data

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
    base_specific = []
    target_specific = []
    shared = []
    
    # Categorize features based on norm ratio
    for feature_id, norms in feature_norms.items():
        if "base_norm" not in norms or "target_norm" not in norms:
            logger.warning(f"Incomplete norm data for feature {feature_id}")
            continue
            
        base_norm = norms["base_norm"]
        target_norm = norms["target_norm"]
        ratio = norms.get("norm_ratio", target_norm / (base_norm + 1e-10))
        
        # To avoid division by zero
        if base_norm < 1e-6 and target_norm < 1e-6:
            # Both norms are effectively zero, skip this feature
            continue
        elif base_norm < 1e-6:
            # Only base norm is zero, this is a target-specific feature
            target_specific.append({
                "id": feature_id,
                "base_norm": base_norm,
                "target_norm": target_norm,
                "ratio": ratio if ratio != np.inf else 1000.0
            })
        elif target_norm < 1e-6:
            # Only target norm is zero, this is a base-specific feature
            base_specific.append({
                "id": feature_id,
                "base_norm": base_norm,
                "target_norm": target_norm,
                "ratio": ratio
            })
        else:
            # Calculate norm ratio
            base_to_target_ratio = base_norm / target_norm
            
            if base_to_target_ratio >= norm_ratio_threshold:
                base_specific.append({
                    "id": feature_id,
                    "base_norm": base_norm,
                    "target_norm": target_norm,
                    "ratio": ratio
                })
            elif ratio >= norm_ratio_threshold:
                target_specific.append({
                    "id": feature_id,
                    "base_norm": base_norm,
                    "target_norm": target_norm,
                    "ratio": ratio
                })
            else:
                shared.append({
                    "id": feature_id,
                    "base_norm": base_norm,
                    "target_norm": target_norm,
                    "ratio": ratio
                })
    
    # Create result dictionary
    categorized_features = {
        "base_specific": base_specific,
        "target_specific": target_specific,
        "shared": shared
    }
    
    # Log the results
    logger.info(f"Categorized {len(base_specific)} base-specific features")
    logger.info(f"Categorized {len(target_specific)} target-specific features")
    logger.info(f"Categorized {len(shared)} shared features")
    
    # Save results if output path is provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(categorized_features, f, indent=2)
        logger.info(f"Saved categorized features to {output_path}")
    
    return categorized_features 