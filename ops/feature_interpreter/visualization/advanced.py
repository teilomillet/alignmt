"""
Advanced visualization module.

This module provides complex visualization functions for feature interpretation.
"""

import matplotlib.pyplot as plt
import numpy as np
import logging
import re
from typing import Dict, Tuple
import seaborn as sns

# Configure logging
logger = logging.getLogger(__name__)

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
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Anthropic-style visualization saved to {output_path}")


def visualize_interpretable_features(
    feature_data: Dict,
    interpretable_features: Dict,
    output_path: str,
    figsize: Tuple[int, int] = (14, 10)
) -> None:
    """
    Create a visualization highlighting notable interpretable features
    from base and target models.
    
    Args:
        feature_data: Dictionary with feature data
        interpretable_features: Dictionary of interpretable features categorized by model and type
        output_path: Path to save the visualization
        figsize: Figure size
    """
    logger.info(f"Creating interpretable features visualization: {output_path}")
    
    # Create a figure with a grid for feature examples
    fig, ax = plt.subplots(figsize=figsize)
    
    # Remove axis
    ax.axis('off')
    
    # Set up the layout
    title = "Notable Interpretable Features: Base vs Target Model"
    fig.suptitle(title, fontsize=16, y=0.98)
    
    # Create a table-like structure
    feature_types = set()
    for model_features in interpretable_features.values():
        for feature_type in model_features.keys():
            feature_types.add(feature_type)
    
    feature_types = sorted(feature_types)
    
    # Set margins and spacing
    margin = 0.1
    row_height = (1.0 - 2 * margin) / (len(feature_types) + 1)  # +1 for header
    col_width = (1.0 - 2 * margin) / 3  # 3 columns (feature type, base, target)
    
    # Add headers
    plt.figtext(margin + col_width * 0.5, 1 - margin, "Feature Type", 
               ha='center', va='center', fontsize=12, fontweight='bold')
    plt.figtext(margin + col_width * 1.5, 1 - margin, "Base Model", 
               ha='center', va='center', fontsize=12, fontweight='bold')
    plt.figtext(margin + col_width * 2.5, 1 - margin, "Target Model", 
               ha='center', va='center', fontsize=12, fontweight='bold')
    
    # Add horizontal line after header
    plt.axhline(y=1 - margin - row_height * 0.5, xmin=margin, xmax=1 - margin, 
               color='black', linestyle='-', linewidth=1)
    
    # Add rows for each feature type
    for i, feature_type in enumerate(feature_types):
        # Calculate row position
        row_pos = 1 - margin - row_height * (i + 1.5)
        
        # Add feature type
        plt.figtext(margin + col_width * 0.5, row_pos, feature_type.replace('_', ' ').title(), 
                   ha='center', va='center', fontsize=11)
        
        # Add base model features
        base_features = interpretable_features.get('base', {}).get(feature_type, [])
        base_text = ''
        if base_features:
            for idx, feature in enumerate(base_features[:3]):  # Show up to 3 features
                desc = feature.get('name', 'Unnamed Feature')
                conf = feature.get('confidence', 0.0)
                base_text += f"• {desc} ({conf:.2f})\n"
        else:
            base_text = "None detected"
            
        plt.figtext(margin + col_width * 1.5, row_pos, base_text, 
                   ha='center', va='center', fontsize=10, linespacing=1.3)
        
        # Add target model features
        target_features = interpretable_features.get('target', {}).get(feature_type, [])
        target_text = ''
        if target_features:
            for idx, feature in enumerate(target_features[:3]):  # Show up to 3 features
                desc = feature.get('name', 'Unnamed Feature')
                conf = feature.get('confidence', 0.0)
                target_text += f"• {desc} ({conf:.2f})\n"
        else:
            target_text = "None detected"
            
        plt.figtext(margin + col_width * 2.5, row_pos, target_text, 
                   ha='center', va='center', fontsize=10, linespacing=1.3)
        
        # Add horizontal line after row
        if i < len(feature_types) - 1:
            plt.axhline(y=1 - margin - row_height * (i + 2), xmin=margin, xmax=1 - margin, 
                       color='gray', linestyle='--', linewidth=0.5)
    
    # Add footer with note
    plt.figtext(0.5, margin/2, 
               "Note: Features are shown with their confidence scores (0-1). Higher scores indicate stronger model differentiation.", 
               ha='center', va='center', fontsize=9, style='italic')
    
    # Save visualization
    plt.tight_layout(rect=[margin, margin, 1-margin, 1-margin])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Interpretable features visualization saved to {output_path}") 