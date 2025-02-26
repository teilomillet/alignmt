"""
Feature Response Analysis Module.

This module provides functionality for analyzing how features
respond to different prompts in base and target models.
"""

import numpy as np
import logging
import os
import json
from pathlib import Path
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logger = logging.getLogger(__name__)

def compare_feature_responses(
    base_activations: np.ndarray,
    target_activations: np.ndarray,
    prompt_labels: List[str],
    feature_names: Optional[List[str]] = None,
    output_dir: Optional[str] = None,
    top_n: int = 10
) -> Dict:
    """
    Compare how features respond to different prompts in both models.
    
    Args:
        base_activations: Base model activation matrix [features × prompts]
        target_activations: Target model activation matrix [features × prompts]
        prompt_labels: Labels for prompts
        feature_names: Optional list of feature names
        output_dir: Optional directory to save visualizations
        top_n: Number of top features to visualize
        
    Returns:
        Dictionary with feature response analysis
    """
    logger.info("Comparing feature responses to prompts")
    
    # Ensure inputs have compatible shapes
    if base_activations.shape != target_activations.shape:
        raise ValueError(f"Activation shapes don't match: {base_activations.shape} vs {target_activations.shape}")
    
    num_features, num_prompts = base_activations.shape
    
    if len(prompt_labels) != num_prompts:
        raise ValueError(f"Number of prompt labels ({len(prompt_labels)}) doesn't match number of prompts ({num_prompts})")
    
    # Create default feature names if not provided
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(num_features)]
    elif len(feature_names) != num_features:
        raise ValueError(f"Number of feature names ({len(feature_names)}) doesn't match number of features ({num_features})")
    
    # Calculate activation differences
    activation_diff = target_activations - base_activations
    
    # Find features with largest differences
    feature_diff_norms = np.linalg.norm(activation_diff, axis=1)
    top_diff_indices = np.argsort(feature_diff_norms)[-top_n:][::-1]
    
    # Prepare output dictionary
    response_analysis = {
        "feature_responses": {},
        "top_different_features": [],
        "prompt_specific_features": {}
    }
    
    # Extract top different features
    for idx in top_diff_indices:
        feature_id = feature_names[idx]
        response_analysis["top_different_features"].append({
            "id": feature_id,
            "difference_norm": float(feature_diff_norms[idx]),
            "base_activations": base_activations[idx].tolist(),
            "target_activations": target_activations[idx].tolist(),
            "activation_diff": activation_diff[idx].tolist()
        })
    
    # Find prompt-specific features (features that respond strongly to specific prompts)
    for p_idx, prompt in enumerate(prompt_labels):
        # Calculate difference in response to this prompt
        prompt_diff = target_activations[:, p_idx] - base_activations[:, p_idx]
        
        # Find top features with largest differences for this prompt
        top_prompt_indices = np.argsort(np.abs(prompt_diff))[-top_n:][::-1]
        
        prompt_features = []
        for idx in top_prompt_indices:
            feature_id = feature_names[idx]
            prompt_features.append({
                "id": feature_id,
                "base_activation": float(base_activations[idx, p_idx]),
                "target_activation": float(target_activations[idx, p_idx]),
                "activation_diff": float(prompt_diff[idx])
            })
        
        response_analysis["prompt_specific_features"][prompt] = prompt_features
    
    # Create visualizations if output directory is provided
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create heatmap of top different features
        plt.figure(figsize=(16, 10))
        
        # Prepare data for heatmap
        heatmap_data = []
        feature_labels = []
        
        for feature in response_analysis["top_different_features"]:
            feature_labels.append(feature["id"])
            heatmap_data.append(feature["activation_diff"])
        
        # Create heatmap
        ax = sns.heatmap(
            heatmap_data,
            annot=False,
            fmt=".2f",
            cmap="RdBu_r",
            center=0,
            yticklabels=feature_labels,
            xticklabels=prompt_labels
        )
        
        # Rotate x labels for better readability
        plt.xticks(rotation=45, ha="right")
        
        # Add labels and title
        plt.xlabel('Prompts')
        plt.ylabel('Features')
        plt.title('Top Features with Different Responses Between Models')
        
        # Save figure
        plt.tight_layout()
        plt.savefig(output_dir / 'feature_response_diff.png', dpi=300)
        plt.close()
        
        # Create feature response comparison plots for top features
        for i, feature in enumerate(response_analysis["top_different_features"][:5]):
            plt.figure(figsize=(12, 6))
            
            # Plot base and target activations
            x = np.arange(len(prompt_labels))
            width = 0.35
            
            plt.bar(x - width/2, feature["base_activations"], width, label='Base Model')
            plt.bar(x + width/2, feature["target_activations"], width, label='Target Model')
            
            # Add labels and title
            plt.xlabel('Prompts')
            plt.ylabel('Activation')
            plt.title(f'Feature {feature["id"]} Response Comparison')
            plt.xticks(x, prompt_labels, rotation=45, ha='right')
            plt.legend()
            
            # Save figure
            plt.tight_layout()
            plt.savefig(output_dir / f'feature_response_{feature["id"]}.png', dpi=300)
            plt.close()
        
        # Save response analysis as JSON
        with open(output_dir / 'feature_response_analysis.json', 'w') as f:
            json.dump(response_analysis, f, indent=2)
        
        logger.info(f"Saved feature response analysis to {output_dir}")
    
    return response_analysis 