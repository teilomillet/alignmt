"""
Basic visualization module.

This module provides fundamental visualization functions for feature interpretation.
"""

import matplotlib.pyplot as plt
import numpy as np
import logging
import re
from typing import Dict, Tuple

# Configure logging
logger = logging.getLogger(__name__)

def create_feature_distribution_plot(
    interpreted_features: Dict,
    output_path: str
) -> None:
    """
    Create a plot showing the distribution of features across layers.
    
    Args:
        interpreted_features: Dictionary with interpreted features
        output_path: Path to save the plot
    """
    # Extract features and layers
    base_features = interpreted_features.get("base_model_specific_features", [])
    target_features = interpreted_features.get("target_model_specific_features", [])
    
    # Collect layer information
    layer_features = {}
    
    # Process base model features
    for feature in base_features:
        layer = feature.get("layer", "unknown")
        if layer not in layer_features:
            layer_features[layer] = {"base": 0, "target": 0}
        layer_features[layer]["base"] += 1
    
    # Process target model features
    for feature in target_features:
        layer = feature.get("layer", "unknown")
        if layer not in layer_features:
            layer_features[layer] = {"base": 0, "target": 0}
        layer_features[layer]["target"] += 1
    
    # Sort layers by number if possible
    sorted_layers = []
    for layer in layer_features.keys():
        # Extract layer number if possible
        match = re.search(r'(\d+)', layer)
        if match:
            layer_num = int(match.group(1))
            sorted_layers.append((layer_num, layer))
        else:
            # If no number found, use a large number to place at the end
            sorted_layers.append((999, layer))
    
    sorted_layers.sort()
    sorted_layer_names = [layer[1] for layer in sorted_layers]
    
    # Create labels for layers
    layer_labels = []
    for layer in sorted_layer_names:
        # Extract layer number if possible
        match = re.search(r'(\d+)', layer)
        if match:
            layer_labels.append(f"Layer {match.group(1)}")
        else:
            layer_labels.append(layer)
    
    # Prepare data for plotting
    base_counts = [layer_features[layer]["base"] for layer in sorted_layer_names]
    target_counts = [layer_features[layer]["target"] for layer in sorted_layer_names]
    
    # Create plot
    plt.figure(figsize=(14, 8))
    
    # Set bar width
    bar_width = 0.35
    
    # Set the position of the bars on the x-axis
    r1 = np.arange(len(layer_labels))
    r2 = [x + bar_width for x in r1]
    
    # Create bars
    plt.bar(r1, base_counts, width=bar_width, label='Base Model', color='blue', alpha=0.7)
    plt.bar(r2, target_counts, width=bar_width, label='Target Model', color='red', alpha=0.7)
    
    # Add labels and title
    plt.xlabel('Layer')
    plt.ylabel('Number of Unique Features')
    plt.title('Distribution of Model-Specific Features Across Layers')
    plt.xticks([r + bar_width/2 for r in range(len(layer_labels))], layer_labels, rotation=45)
    plt.legend()
    
    # Ensure the plot is nicely laid out
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    logger.info(f"Feature distribution plot saved to {output_path}")


def create_feature_heatmap(
    feature_data: Dict,
    output_path: str,
    title: str = "Feature Confidence Heatmap",
    figsize: Tuple[int, int] = (12, 8)
) -> None:
    """
    Create a heatmap showing feature confidence across layers.
    
    Args:
        feature_data: Dictionary with feature data
        output_path: Path to save the plot
        title: Plot title
        figsize: Figure size
    """
    logger.info(f"Creating feature heatmap: {output_path}")
    
    # Extract data
    base_features = feature_data.get('base_model_specific_features', [])
    target_features = feature_data.get('target_model_specific_features', [])
    
    # Get unique feature names and layers
    base_names = set(f.get('name', 'Unknown') for f in base_features)
    target_names = set(f.get('name', 'Unknown') for f in target_features)
    all_names = sorted(base_names.union(target_names))
    
    # If there are too many features, filter to keep only the most confident ones
    max_features_to_display = 30
    if len(all_names) > max_features_to_display:
        logger.info(f"Limiting heatmap to top {max_features_to_display} features by confidence")
        
        # Extract all features with confidence
        all_features = []
        for f in base_features + target_features:
            if 'name' in f and 'confidence' in f:
                all_features.append((f['name'], f['confidence']))
        
        # Group by name and take max confidence
        feature_confidence = {}
        for name, conf in all_features:
            if name not in feature_confidence or conf > feature_confidence[name]:
                feature_confidence[name] = conf
        
        # Sort and take top features
        top_features = sorted(feature_confidence.items(), key=lambda x: x[1], reverse=True)[:max_features_to_display]
        all_names = [f[0] for f in top_features]
    
    all_layers = sorted(set(f.get('layer', 'unknown') for f in base_features + target_features if 'layer' in f))
    layer_labels = [layer.split('.')[-3] if '.' in layer else layer for layer in all_layers]
    
    # Create matrices for heatmap
    base_matrix = np.zeros((len(all_names), len(all_layers)))
    target_matrix = np.zeros((len(all_names), len(all_layers)))
    
    # Fill matrices with confidence values
    for feature in base_features:
        if 'name' in feature and feature['name'] in all_names and 'layer' in feature and feature['layer'] in all_layers:
            name_idx = all_names.index(feature['name'])
            layer_idx = all_layers.index(feature['layer'])
            base_matrix[name_idx, layer_idx] = feature.get('confidence', 0)
    
    for feature in target_features:
        if 'name' in feature and feature['name'] in all_names and 'layer' in feature and feature['layer'] in all_layers:
            name_idx = all_names.index(feature['name'])
            layer_idx = all_layers.index(feature['layer'])
            target_matrix[name_idx, layer_idx] = feature.get('confidence', 0)
    
    # Create plot with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot base model heatmap
    im1 = ax1.imshow(base_matrix, cmap='Blues', aspect='auto')
    ax1.set_title('Base Model Features', fontsize=12)
    ax1.set_xticks(np.arange(len(layer_labels)))
    ax1.set_yticks(np.arange(len(all_names)))
    ax1.set_xticklabels(layer_labels, rotation=45, ha='right')
    ax1.set_yticklabels(all_names)
    ax1.set_xlabel('Layer')
    
    # Plot target model heatmap
    im2 = ax2.imshow(target_matrix, cmap='Reds', aspect='auto')
    ax2.set_title('Target Model Features', fontsize=12)
    ax2.set_xticks(np.arange(len(layer_labels)))
    ax2.set_yticks(np.arange(len(all_names)))
    ax2.set_xticklabels(layer_labels, rotation=45, ha='right')
    ax2.set_yticklabels([])  # No need to repeat labels
    ax2.set_xlabel('Layer')
    
    # Add colorbars
    cbar1 = fig.colorbar(im1, ax=ax1)
    cbar1.set_label('Confidence')
    cbar2 = fig.colorbar(im2, ax=ax2)
    cbar2.set_label('Confidence')
    
    # Add overall title
    fig.suptitle(title, fontsize=16)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close() 