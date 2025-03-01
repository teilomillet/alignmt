"""
Advanced visualization module.

This module provides complex visualization functions for feature interpretation.
"""

import matplotlib.pyplot as plt
import numpy as np
import logging
import re
from typing import Dict, Tuple, List
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

# Configure logging
logger = logging.getLogger(__name__)

def create_reasoning_category_visualization(
    interpreted_features: Dict,
    layer_similarities: Dict,
    output_path: str
) -> None:
    """
    Create a visualization showing feature distributions across reasoning categories and layers.
    
    Args:
        interpreted_features: Dictionary with interpreted features
        layer_similarities: Dictionary with layer similarities
        output_path: Path to save the visualization
    """
    # Extract features based on different possible formats
    if "features" in interpreted_features and isinstance(interpreted_features["features"], list):
        # New format: features are in a list under "features" key
        features_list = interpreted_features["features"]
        base_features = []
        target_features = []
        
        # Determine which features belong to base vs target based on their description
        for feature in features_list:
            if "description" in feature:
                description = feature["description"].lower()
                if "base" in description or "weakened" in description:
                    base_features.append(feature)
                elif "target" in description or "added" in description or "enhanced" in description:
                    target_features.append(feature)
                else:
                    # If not clear, put in base features by default
                    base_features.append(feature)
    else:
        # Old format: separate keys for base and target features
        base_features = interpreted_features.get("base_model_specific_features", [])
        target_features = interpreted_features.get("target_model_specific_features", [])
    
    # Create a figure with three panels
    fig = plt.figure(figsize=(18, 12))
    
    # Define grid for the figure
    gs = plt.GridSpec(4, 6, figure=fig)
    
    # First panel (top left): Layer similarities
    ax1 = fig.add_subplot(gs[0:2, 0:2])
    
    # Second panel (spans two rows): Feature distribution by reasoning category
    ax2 = fig.add_subplot(gs[0:4, 2:6])
    
    # Third panel (bottom left): Legend and summary
    ax3 = fig.add_subplot(gs[2:4, 0:2])
    
    # Plot layer similarities
    _plot_layer_similarities(ax1, layer_similarities)
    
    # Check if we have any features
    if not base_features and not target_features:
        _plot_no_features_message(ax2)
        _plot_legend_and_summary(ax3, {}, {})
    else:
        # Group features by reasoning category
        base_by_category = _group_by_reasoning_category(base_features)
        target_by_category = _group_by_reasoning_category(target_features)
        
        # Plot feature distribution by reasoning category
        _plot_features_by_reasoning(ax2, base_by_category, target_by_category, layer_similarities)
        
        # Plot legend and summary
        _plot_legend_and_summary(ax3, base_by_category, target_by_category)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Reasoning category visualization saved to {output_path}")

def _plot_layer_similarities(ax, layer_similarities: Dict) -> None:
    """Plot layer similarities in a horizontal bar chart."""
    if not layer_similarities:
        ax.text(0.5, 0.5, "No layer similarities data available", 
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, fontsize=12)
        ax.set_title('Layer Similarities (No Data)')
        return
    
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
    
    # Plot layer similarities
    similarities = [layer_similarities[layer] for layer in sorted_layer_names]
    
    # Add color gradient based on similarity value
    colors = plt.cm.coolwarm(np.array(similarities))
    
    # Create a horizontal bar chart for similarities
    bars = ax.barh(layer_labels, similarities, color=colors)
    ax.set_xlabel('Similarity')
    ax.set_ylabel('Layer')
    ax.set_title('Layer Similarities')
    ax.set_xlim(0, 1)
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Add a horizontal line at the average similarity
    avg_similarity = np.mean(similarities)
    ax.axvline(x=avg_similarity, color='green', linestyle='--', 
               label=f'Avg: {avg_similarity:.2f}')
    ax.legend(loc='upper right', fontsize=8)

def _plot_no_features_message(ax) -> None:
    """Plot a message when no features are available."""
    ax.text(0.5, 0.5, "No features found\n\nThis may indicate either:\n- Very similar models with few distinct features\n- Insufficient prompts to detect differences\n- Issue in feature extraction", 
             horizontalalignment='center', verticalalignment='center',
             transform=ax.transAxes, fontsize=14)
    ax.set_title('Feature Distribution by Reasoning Category (No Features Detected)')
    ax.axis('off')

def _group_by_reasoning_category(features: List[Dict]) -> Dict[str, List[Dict]]:
    """Group features by reasoning category."""
    # Initialize categories
    categories = {
        "logical": [],
        "mathematical": [],
        "probabilistic": [],
        "spatial": [],
        "temporal": [],
        "counterfactual": [],
        "analogical": [],
        "causal": [],
        "ethical": [],
        "creative": [],
        "other": []
    }
    
    # Group features by reasoning category
    for feature in features:
        assigned = False
        
        # Check for themes in the new format
        if "themes" in feature and isinstance(feature["themes"], dict):
            primary_theme = feature["themes"].get("primary", "").lower()
            if primary_theme and primary_theme in categories:
                categories[primary_theme].append(feature)
                assigned = True
            
        # Try to infer from the name or description if themes not available
        if not assigned:
            name = feature.get("name", "").lower()
            description = feature.get("description", "").lower()
            
            # Check name and description for category keywords
            for category in categories.keys():
                if category != "other" and (category in name or category in description):
                    categories[category].append(feature)
                    assigned = True
                    break
            
            # If still not assigned, put in "other"
            if not assigned:
                categories["other"].append(feature)
    
    # Remove empty categories
    return {k: v for k, v in categories.items() if v}

def _plot_features_by_reasoning(
    ax, 
    base_by_category: Dict, 
    target_by_category: Dict, 
    layer_similarities: Dict
) -> None:
    """Plot features organized by reasoning category and layers."""
    # Get all unique categories
    all_categories = set(list(base_by_category.keys()) + list(target_by_category.keys()))
    
    # Get all unique layers
    all_layers = set()
    for features in base_by_category.values():
        for f in features:
            if "layer" in f:
                all_layers.add(f["layer"])
    
    for features in target_by_category.values():
        for f in features:
            if "layer" in f:
                all_layers.add(f["layer"])
    
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
    
    # Create layer labels for y-axis
    layer_labels = []
    for layer in sorted_layer_names:
        match = re.search(r'(\d+)', layer)
        if match:
            layer_labels.append(f"Layer {match.group(1)}")
        else:
            layer_labels.append(layer)
    
    # Sort categories alphabetically
    sorted_categories = sorted(all_categories)
    
    # Create a matrix for the heatmap
    matrix = np.zeros((len(sorted_layer_names), len(sorted_categories) * 2))
    
    # Define custom colormaps for base and target models
    base_colors = [(1, 1, 1), (0, 0, 0.8)]  # White to blue
    target_colors = [(1, 1, 1), (0.8, 0, 0)]  # White to red
    base_cmap = LinearSegmentedColormap.from_list("base_cmap", base_colors)
    target_cmap = LinearSegmentedColormap.from_list("target_cmap", target_colors)
    
    # Fill matrix with feature counts and confidence
    for c_idx, category in enumerate(sorted_categories):
        # Base model features (left side of the category)
        for feature in base_by_category.get(category, []):
            layer = feature.get("layer", "unknown")
            if layer in sorted_layer_names:
                l_idx = sorted_layer_names.index(layer)
                confidence = feature.get("confidence", feature.get("avg_difference", 1.0))
                matrix[l_idx, c_idx * 2] += confidence
        
        # Target model features (right side of the category)
        for feature in target_by_category.get(category, []):
            layer = feature.get("layer", "unknown")
            if layer in sorted_layer_names:
                l_idx = sorted_layer_names.index(layer)
                confidence = feature.get("confidence", feature.get("avg_difference", 1.0))
                matrix[l_idx, c_idx * 2 + 1] += confidence
    
    # Normalize matrix for better visualization
    max_val = np.max(matrix) if np.max(matrix) > 0 else 1.0
    matrix = matrix / max_val
    
    # Create the heatmap
    sns.heatmap(matrix, cmap="viridis", ax=ax, cbar=False, 
                linewidths=0.5, linecolor='white')
    
    # Customize the plot
    ax.set_yticks(np.arange(len(layer_labels)) + 0.5)
    ax.set_yticklabels(layer_labels)
    
    # Set x-ticks at the middle of each category pair
    xticks = np.arange(1, len(sorted_categories) * 2, 2)
    ax.set_xticks(xticks)
    ax.set_xticklabels(sorted_categories, rotation=45, ha='right')
    
    # Add markers for base (B) and target (T) models
    for c_idx, category in enumerate(sorted_categories):
        # Add visual indicators for model types
        for l_idx, layer in enumerate(sorted_layer_names):
            # Base model features
            if matrix[l_idx, c_idx * 2] > 0:
                ax.text(c_idx * 2 + 0.25, l_idx + 0.5, "B", 
                       color='white' if matrix[l_idx, c_idx * 2] > 0.5 else 'black',
                       ha='center', va='center', fontsize=8, fontweight='bold')
            
            # Target model features
            if matrix[l_idx, c_idx * 2 + 1] > 0:
                ax.text(c_idx * 2 + 1 + 0.25, l_idx + 0.5, "T", 
                       color='white' if matrix[l_idx, c_idx * 2 + 1] > 0.5 else 'black',
                       ha='center', va='center', fontsize=8, fontweight='bold')
    
    ax.set_title('Feature Distribution by Reasoning Category and Layer')

def _plot_legend_and_summary(
    ax, 
    base_by_category: Dict, 
    target_by_category: Dict
) -> None:
    """Plot a legend and summary information."""
    # Turn off axis
    ax.axis('off')
    
    # Calculate summary statistics
    base_count = sum(len(features) for features in base_by_category.values())
    target_count = sum(len(features) for features in target_by_category.values())
    
    # Get counts by category
    base_category_counts = {category: len(features) for category, features in base_by_category.items()}
    target_category_counts = {category: len(features) for category, features in target_by_category.items()}
    
    # Create summary text
    summary = f"Feature Summary:\n\n"
    summary += f"Base Model Features: {base_count}\n"
    summary += f"Target Model Features: {target_count}\n\n"
    
    # Add category breakdown if we have features
    if base_category_counts or target_category_counts:
        all_categories = set(list(base_category_counts.keys()) + list(target_category_counts.keys()))
        summary += "Category Breakdown:\n"
        
        for category in sorted(all_categories):
            base_cat_count = base_category_counts.get(category, 0)
            target_cat_count = target_category_counts.get(category, 0)
            summary += f"• {category.capitalize()}: {base_cat_count} base, {target_cat_count} target\n"
    
    # Add legend explanation
    summary += "\nLegend:\n"
    summary += "• 'B': Base model feature\n"
    summary += "• 'T': Target model feature\n"
    summary += "• Color intensity shows confidence/significance"
    
    # Add the text to the axis
    ax.text(0.02, 0.98, summary, 
           transform=ax.transAxes, 
           verticalalignment='top',
           horizontalalignment='left',
           fontsize=10)




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