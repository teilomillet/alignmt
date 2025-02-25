"""
Feature visualization module.

This module creates Anthropic-style visualizations of feature differences
between base and target models.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import logging
import os
import pickle
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
import seaborn as sns
import re

# Configure logging
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

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
    
    # Create layer labels for x-axis
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
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    
    # Create a grouped bar chart
    x = np.arange(len(layer_labels))
    width = 0.35
    
    plt.bar(x - width/2, base_counts, width, label='Base Model')
    plt.bar(x + width/2, target_counts, width, label='Target Model')
    
    plt.xlabel('Layer')
    plt.ylabel('Number of Features')
    plt.title('Distribution of Distinctive Features Across Layers')
    plt.xticks(x, layer_labels, rotation=45, ha='right')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def create_anthropic_style_visualization(
    interpreted_features: Dict,
    layer_similarities: Dict,
    output_path: str
) -> None:
    """
    Create an Anthropic-style visualization showing feature distributions and layer similarities.
    
    Args:
        interpreted_features: Dictionary with interpreted features
        layer_similarities: Dictionary with layer similarities
        output_path: Path to save the visualization
    """
    # Extract features
    base_features = interpreted_features.get("base_model_specific_features", [])
    target_features = interpreted_features.get("target_model_specific_features", [])
    
    # Create a figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 10), gridspec_kw={'width_ratios': [1, 2]})
    
    # Check if there are layer similarities
    if not layer_similarities:
        logger.warning("No layer similarities found for visualization. Creating placeholder.")
        ax1.text(0.5, 0.5, "No layer similarities data available", 
                 horizontalalignment='center', verticalalignment='center',
                 transform=ax1.transAxes, fontsize=14)
        ax1.set_title('Layer Similarities (No Data)')
    else:
        # Sort layers by number if possible
        sorted_layers = []
        for layer in layer_similarities.keys():
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
        
        # Create layer labels for y-axis
        layer_labels = []
        for layer in sorted_layer_names:
            # Extract layer number if possible
            match = re.search(r'(\d+)', layer)
            if match:
                layer_labels.append(f"Layer {match.group(1)}")
            else:
                layer_labels.append(layer)
        
        # Plot 1: Layer similarities
        similarities = [layer_similarities[layer] for layer in sorted_layer_names]
        
        # Create a horizontal bar chart for similarities
        ax1.barh(layer_labels, similarities, color='lightgray')
        ax1.set_xlabel('Similarity')
        ax1.set_ylabel('Layer')
        ax1.set_title('Layer Similarities')
        ax1.set_xlim(0, 1)
        ax1.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Check if there are any features at all
    if not base_features and not target_features:
        logger.warning("No features found for visualization. Creating fallback plot.")
        ax2.text(0.5, 0.5, "No features found\n\nThis may indicate either:\n- Very similar models with few distinct features\n- Insufficient prompts to detect differences\n- Issue in feature extraction", 
                 horizontalalignment='center', verticalalignment='center',
                 transform=ax2.transAxes, fontsize=14)
        ax2.set_title('Feature Distribution Across Layers (No Features Detected)')
    else:
        # Get all layers from both base and target features
        all_layers = set()
        for feature in base_features + target_features:
            layer = feature.get("layer", "unknown")
            all_layers.add(layer)
        
        # Sort layers
        sorted_layers = []
        for layer in all_layers:
            match = re.search(r'(\d+)', layer)
            if match:
                layer_num = int(match.group(1))
                sorted_layers.append((layer_num, layer))
            else:
                sorted_layers.append((999, layer))
        
        sorted_layers.sort()
        sorted_layer_names = [layer[1] for layer in sorted_layers]
        
        # Create layer labels
        layer_labels = []
        for layer in sorted_layer_names:
            match = re.search(r'(\d+)', layer)
            if match:
                layer_labels.append(f"Layer {match.group(1)}")
            else:
                layer_labels.append(layer)
        
        # Initialize feature matrix
        feature_matrix = np.zeros((len(layer_labels), 2))
        
        # Count features per layer for base model
        for feature in base_features:
            layer = feature.get("layer", "unknown")
            if layer in sorted_layer_names:
                idx = sorted_layer_names.index(layer)
                feature_matrix[idx, 0] += feature.get("confidence", 1.0)
        
        # Count features per layer for target model
        for feature in target_features:
            layer = feature.get("layer", "unknown")
            if layer in sorted_layer_names:
                idx = sorted_layer_names.index(layer)
                feature_matrix[idx, 1] += feature.get("confidence", 1.0)
        
        # Check if the feature matrix is empty or has no non-zero values
        if feature_matrix.size == 0 or np.count_nonzero(feature_matrix) == 0:
            logger.warning("Feature matrix is empty or has no non-zero values. Creating fallback plot.")
            ax2.text(0.5, 0.5, "No significant features detected\n\nFeatures may be present but below confidence threshold", 
                     horizontalalignment='center', verticalalignment='center',
                     transform=ax2.transAxes, fontsize=14)
            ax2.set_title('Feature Distribution (No Significant Features)')
        else:
            # Normalize feature counts
            max_count = np.max(feature_matrix)
            if max_count > 0:  # Avoid division by zero
                feature_matrix = feature_matrix / max_count
            
            # Create a custom colormap (white to blue for base, white to red for target)
            cmap_base = plt.cm.Blues
            cmap_target = plt.cm.Reds
            
            # Plot heatmap for base model
            sns.heatmap(feature_matrix[:, 0:1], ax=ax2, cmap=cmap_base, 
                        cbar=False, linewidths=1, linecolor='white',
                        xticklabels=['Base Model'], yticklabels=layer_labels)
            
            # Plot heatmap for target model
            sns.heatmap(feature_matrix[:, 1:2], ax=ax2, cmap=cmap_target, 
                        cbar=True, linewidths=1, linecolor='white',
                        xticklabels=['Target Model'], yticklabels=[])
            
            ax2.set_title('Feature Distribution Across Layers')
            
            # Add feature annotations
            for i, layer in enumerate(sorted_layer_names):
                # Add base model features
                base_layer_features = [f for f in base_features if f.get("layer") == layer]
                if base_layer_features:
                    feature_names = [f.get("name", "Unknown") for f in base_layer_features]
                    ax2.text(-0.5, i + 0.5, ", ".join(feature_names[:2]), 
                             ha='right', va='center', fontsize=8, color='blue')
                
                # Add target model features
                target_layer_features = [f for f in target_features if f.get("layer") == layer]
                if target_layer_features:
                    feature_names = [f.get("name", "Unknown") for f in target_layer_features]
                    ax2.text(2.5, i + 0.5, ", ".join(feature_names[:2]), 
                             ha='left', va='center', fontsize=8, color='red')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

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
    base_features = feature_data['base_model_specific_features']
    target_features = feature_data['target_model_specific_features']
    
    # Get unique feature names and layers
    base_names = set(f['name'] for f in base_features)
    target_names = set(f['name'] for f in target_features)
    all_names = sorted(base_names.union(target_names))
    
    all_layers = sorted(set(f['layer'] for f in base_features + target_features))
    layer_labels = [layer.split('.')[-3] if '.' in layer else layer for layer in all_layers]
    
    # Create matrices for heatmap
    base_matrix = np.zeros((len(all_names), len(all_layers)))
    target_matrix = np.zeros((len(all_names), len(all_layers)))
    
    # Fill matrices with confidence values
    for feature in base_features:
        name_idx = all_names.index(feature['name'])
        layer_idx = all_layers.index(feature['layer'])
        base_matrix[name_idx, layer_idx] = feature['confidence']
    
    for feature in target_features:
        name_idx = all_names.index(feature['name'])
        layer_idx = all_layers.index(feature['layer'])
        target_matrix[name_idx, layer_idx] = feature['confidence']
    
    # Create plot with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot base model heatmap
    im1 = ax1.imshow(base_matrix, cmap='Blues', aspect='auto')
    ax1.set_title(f'Base Model Features', fontsize=12)
    ax1.set_xticks(np.arange(len(layer_labels)))
    ax1.set_yticks(np.arange(len(all_names)))
    ax1.set_xticklabels(layer_labels, rotation=45, ha='right')
    ax1.set_yticklabels(all_names)
    ax1.set_xlabel('Layer')
    
    # Plot target model heatmap
    im2 = ax2.imshow(target_matrix, cmap='Reds', aspect='auto')
    ax2.set_title(f'Target Model Features', fontsize=12)
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

def create_visualizations(
    feature_data: Dict,
    output_dir: str = "feature_visualizations"
) -> None:
    """
    Create all visualizations for feature interpretation.
    
    Args:
        feature_data: Dictionary with feature data
        output_dir: Directory to save visualizations
    """
    logger.info(f"Creating visualizations in directory: {output_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create feature distribution plot
    create_feature_distribution_plot(
        feature_data,
        os.path.join(output_dir, "feature_distribution.png"),
        title="Feature Distribution Between Models"
    )
    
    # Create Anthropic-style visualization
    create_anthropic_style_visualization(
        feature_data,
        os.path.join(output_dir, "feature_anthropic_style.png"),
        title="Feature-Level Model Differences"
    )
    
    # Create feature heatmap
    create_feature_heatmap(
        feature_data,
        os.path.join(output_dir, "feature_heatmap.png"),
        title="Feature Confidence Heatmap"
    )
    
    logger.info(f"Visualizations created successfully")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create visualizations for feature interpretation")
    parser.add_argument("--feature-file", required=True, help="Path to feature interpretation JSON file")
    parser.add_argument("--output-dir", default="feature_visualizations", help="Output directory")
    
    args = parser.parse_args()
    
    # Load feature data
    with open(args.feature_file, "r") as f:
        feature_data = json.load(f)
    
    # Create visualizations
    create_visualizations(feature_data, args.output_dir) 