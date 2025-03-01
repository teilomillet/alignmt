"""
Basic visualization module.

This module provides fundamental visualization functions for feature interpretation.
"""

import matplotlib.pyplot as plt
import numpy as np
import logging
import re
from typing import Dict, Tuple, List, Optional
import torch
import os
import seaborn as sns

from ..decoder_analysis.norms import extract_feature_decoder_norms, categorize_features_by_norm

# Configure logging
logger = logging.getLogger(__name__)

def create_feature_distribution_plot(
    interpreted_features: Dict,
    output_path: str,
    feature_data: Optional[Dict] = None,
    crosscoder_data: Optional[Dict] = None,
    include_advanced_visualizations: bool = True,
    feature_threshold: float = 0.01
) -> None:
    """
    Create a plot showing the distribution of features across layers with advanced
    visualizations inspired by Transformer Circuits methodology.
    
    Args:
        interpreted_features: Dictionary with interpreted features
        output_path: Path to save the plot
        feature_data: Optional dictionary with feature decoder norm data
        crosscoder_data: Optional dictionary with cosine similarity data between models
        include_advanced_visualizations: Whether to include advanced visualizations
        feature_threshold: Threshold for feature confidence
    """
    # Extract features using the appropriate format
    # Check if we're using the new format with a 'features' list
    if "features" in interpreted_features and isinstance(interpreted_features["features"], list):
        # New format - features are in a single list with model identification in description
        features_list = interpreted_features["features"]
        base_features = []
        target_features = []
        
        # Log the total number of features found
        logger.info(f"Found {len(features_list)} features in new format")
        
        # Enhanced classification logic based on multiple attributes
        for feature in features_list:
            # Default to unclassified
            model_attribution = None
            
            # 0. First check for the explicit model_attribution field
            if "model_attribution" in feature:
                model_attribution = feature["model_attribution"].lower()
                logger.info(f"Feature '{feature.get('name', 'Unnamed')}' has explicit model_attribution: {model_attribution}")
            
            # If model_attribution not explicitly set, use other heuristics
            if model_attribution is None:
                # 1. Try classification based on description keywords
                if "description" in feature:
                    description = feature.get("description", "").lower()
                    if any(kw in description for kw in ["base model", "stronger in the base", "weakened", "removed"]):
                        model_attribution = "base"
                    elif any(kw in description for kw in ["target model", "stronger in the target", "added", "enhanced"]):
                        model_attribution = "target"
                
                # 2. Try to use validation info if available
                if model_attribution is None and "validation_score" in feature:
                    # Higher validation scores indicate stronger effect in target model
                    if feature["validation_score"] > 0.5:
                        model_attribution = "target"
                    else:
                        model_attribution = "base"
                
                # 3. Check for themes or primary_themes
                if model_attribution is None and "themes" in feature:
                    # If the model has specific themes that indicate model preference
                    themes = feature.get("themes", {})
                    if isinstance(themes, dict) and "primary" in themes:
                        primary_theme = themes["primary"].lower()
                        if "removed" in primary_theme or "weakened" in primary_theme:
                            model_attribution = "base"
                        elif "added" in primary_theme or "enhanced" in primary_theme:
                            model_attribution = "target"
                
                # 4. Look at the feature name as last resort
                if model_attribution is None and "name" in feature:
                    name = feature["name"].lower()
                    if "base" in name or "removed" in name:
                        model_attribution = "base"
                    elif "target" in name or "added" in name:
                        model_attribution = "target"
                
                # 5. Check for other possible model indicators
                if model_attribution is None:
                    # If the feature has indicators like "model_attribution" directly
                    if "model_attribution" in feature:
                        model_attribution = feature["model_attribution"].lower()
                    # Or if it has a "source_model" field
                    elif "source_model" in feature:
                        source = feature["source_model"].lower()
                        if "base" in source:
                            model_attribution = "base"
                        elif "target" in source:
                            model_attribution = "target"
            
            # For research debugging - log feature and its classification
            name = feature.get("name", "Unnamed feature")
            layer = feature.get("layer", "unknown")
            logger.info(f"Feature '{name}' in layer '{layer}' classified as: {model_attribution or 'unclassified'}")
            
            # If still not classified, use much more aggressive heuristics, but this should be rare now
            if model_attribution is None:
                # Fallback: alternate between base and target for even distribution
                # This is just to avoid all features defaulting to one model
                if len(base_features) <= len(target_features):
                    model_attribution = "base"
                    logger.warning(f"Feature '{name}' defaulted to 'base' model with fallback logic")
                else:
                    model_attribution = "target"
                    logger.warning(f"Feature '{name}' defaulted to 'target' model with fallback logic")
            
            # Assign to appropriate list
            if model_attribution == "base":
                base_features.append(feature)
            else:
                target_features.append(feature)
    else:
        # Old format with separate keys
        base_features = interpreted_features.get("base_model_specific_features", [])
        target_features = interpreted_features.get("target_model_specific_features", [])
    
    # Log the number of features found for debugging
    logger.info(f"Found {len(base_features)} base model features and {len(target_features)} target model features")
    
    # Check if we have advanced visualizations requested AND feature data available
    if include_advanced_visualizations and feature_data:
        # Create a more sophisticated visualization inspired by Transformer Circuits
        try:
            create_advanced_feature_distribution_plot(
                base_features=base_features,
                target_features=target_features,
                feature_data=feature_data,
                crosscoder_data=crosscoder_data,
                output_path=output_path
            )
            logger.info(f"Created advanced feature distribution plot with decoder analysis data at {output_path}")
            # Exit early since we've created the advanced plot
            return
        except Exception as e:
            logger.warning(f"Failed to create advanced feature distribution plot: {str(e)}")
            logger.warning("Falling back to basic visualization")
    
    # If we reach here, either advanced visualization was not requested,
    # feature data was not available, or advanced visualization failed.
    # Fall back to the basic visualization.
    
    # Group features by layer
    layer_features = {}
    
    # Extract layer information from base features
    for feature in base_features:
        layer = feature.get("layer", "unknown")
        if layer not in layer_features:
            layer_features[layer] = {"base": 0, "target": 0}
        layer_features[layer]["base"] += 1
    
    # Extract layer information from target features
    for feature in target_features:
        layer = feature.get("layer", "unknown")
        if layer not in layer_features:
            layer_features[layer] = {"base": 0, "target": 0}
        layer_features[layer]["target"] += 1
    
    # Sort layers by number if possible
    sorted_layers = []
    for layer in layer_features:
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
    
    # Create layer labels with feature counts
    layer_labels = []
    for layer in sorted_layer_names:
        # Extract layer number if possible
        match = re.search(r'(\d+)', layer)
        if match:
            layer_num = match.group(1)
            # Add feature counts to the label
            base_count = layer_features[layer]["base"]
            target_count = layer_features[layer]["target"]
            total_count = base_count + target_count
            
            # Only add feature text if there are any features
            feature_text = ""
            if total_count > 0:
                feature_text = f" ({base_count}B/{target_count}T)"
                
            layer_labels.append(f"Layer {layer_num}{feature_text}")
        else:
            layer_labels.append(f"Layer {layer_num}")
    
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
    
    # Create bars with improved styling for better readability
    plt.bar(r1, base_counts, width=bar_width, label='Base Model', color='#3498db', alpha=0.8, edgecolor='#2980b9')
    plt.bar(r2, target_counts, width=bar_width, label='Target Model', color='#e74c3c', alpha=0.8, edgecolor='#c0392b')
    
    # Add value labels on top of bars
    for i, v in enumerate(base_counts):
        if v > 0:
            plt.text(r1[i], v + 0.1, str(v), ha='center', fontsize=9)
    
    for i, v in enumerate(target_counts):
        if v > 0:
            plt.text(r2[i], v + 0.1, str(v), ha='center', fontsize=9)
    
    # Add labels and title with improved styling
    plt.xlabel('Layer', fontsize=12, fontweight='bold')
    plt.ylabel('Number of Unique Features', fontsize=12, fontweight='bold')
    plt.title('Distribution of Model-Specific Features Across Layers', fontsize=14, fontweight='bold')
    plt.xticks([r + bar_width/2 for r in range(len(layer_labels))], layer_labels, rotation=45, ha='right')
    plt.legend(loc='upper right', frameon=True, fontsize=10)
    
    # Add grid for better readability
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    
    # Add a summary text box at the bottom of the chart
    total_base = sum(base_counts)
    total_target = sum(target_counts)
    total_features = total_base + total_target
    summary_text = (
        f"Total Features: {total_features}\n"
        f"Base Model: {total_base} features ({total_base/total_features*100:.1f}% of total)\n"
        f"Target Model: {total_target} features ({total_target/total_features*100:.1f}% of total)\n"
        f"Feature Analysis Method: Activation difference threshold {feature_threshold:.4f}"
    )
    
    # Add the summary box
    plt.figtext(0.5, 0.01, summary_text, ha='center', fontsize=10, 
                bbox=dict(facecolor='#f8f9fa', alpha=0.8, boxstyle='round,pad=0.5', edgecolor='#d3d3d3'))
    
    # Add reasoning categories as a subtitle if available
    if "reasoning_categories" in interpreted_features:
        categories = interpreted_features["reasoning_categories"]
        categories_str = ", ".join(categories) if len(categories) <= 5 else ", ".join(categories[:5]) + "..."
        plt.figtext(0.5, 0.96, f"Reasoning Categories: {categories_str}", ha='center', fontsize=10, 
                    style='italic', color='#555555')
    
    # Add tight layout to ensure everything fits
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)  # Make room for summary text
    
    # Save the plot
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    logger.info(f"Feature distribution plot saved to {output_path}")


def create_advanced_feature_distribution_plot(
    base_features: List[Dict],
    target_features: List[Dict],
    feature_data: Dict,
    crosscoder_data: Optional[Dict] = None,
    output_path: str = "feature_distribution_advanced.png"
) -> None:
    """
    Create an advanced visualization of feature distributions across models,
    inspired by the Transformer Circuits paper approach.
    
    Args:
        base_features: List of features attributed to the base model
        target_features: List of features attributed to the target model
        feature_data: Dictionary with feature decoder norm data from extract_feature_decoder_norms
        crosscoder_data: Optional dictionary with crosscoder cosine similarity data
        output_path: Path to save the visualization
    """
    logger.info("Creating advanced feature distribution visualization")
    
    # Create the output directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    # Group features by layer
    base_by_layer = {}
    target_by_layer = {}
    all_layers = set()
    
    # Extract layer information from base features
    for feature in base_features:
        layer = feature.get("layer", "unknown")
        all_layers.add(layer)
        if layer not in base_by_layer:
            base_by_layer[layer] = []
        base_by_layer[layer].append(feature)
    
    # Extract layer information from target features
    for feature in target_features:
        layer = feature.get("layer", "unknown")
        all_layers.add(layer)
        if layer not in target_by_layer:
            target_by_layer[layer] = []
        target_by_layer[layer].append(feature)
    
    # Sort layers by number if possible
    sorted_layers = []
    for layer in all_layers:
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
    
    # Check if we have feature norms data
    has_norm_data = "feature_norms" in feature_data
    
    # Create a figure with multiple subplots
    if has_norm_data:
        # With norm data, we can have a more complex visualization
        fig = plt.figure(figsize=(20, 15))
        gs = fig.add_gridspec(3, 3)
        
        # Feature count by layer subplot
        ax_count = fig.add_subplot(gs[0, :])
        
        # Norm distribution subplot
        ax_norm = fig.add_subplot(gs[1, :2])
        
        # Feature density subplot
        ax_density = fig.add_subplot(gs[1, 2])
        
        # Cosine similarity heatmap subplot
        ax_cosine = fig.add_subplot(gs[2, :])
    else:
        # Without norm data, just show counts and any available density information
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(2, 2)
        
        # Feature count by layer subplot
        ax_count = fig.add_subplot(gs[0, :])
        
        # Feature density subplot
        ax_density = fig.add_subplot(gs[1, 0])
        
        # Placeholder for norm data
        ax_norm = fig.add_subplot(gs[1, 1])
        
        # No cosine similarity subplot
        ax_cosine = None
    
    # Plot 1: Feature count by layer (similar to original but enhanced)
    _plot_feature_counts(ax_count, sorted_layer_names, base_by_layer, target_by_layer)
    
    # Plot 2: Norm distribution (if available)
    if has_norm_data:
        _plot_norm_distribution(ax_norm, feature_data, base_features, target_features)
    else:
        ax_norm.text(0.5, 0.5, "No decoder norm data available", 
                    ha='center', va='center', fontsize=14)
        ax_norm.set_title("Feature Decoder Norm Distribution")
        ax_norm.axis('off')
    
    # Plot 3: Feature density
    _plot_feature_density(ax_density, base_features, target_features, feature_data)
    
    # Plot 4: Cosine similarity (if available)
    if has_norm_data and crosscoder_data and ax_cosine is not None:
        _plot_cosine_similarity(ax_cosine, feature_data, crosscoder_data)
    elif ax_cosine is not None:
        ax_cosine.text(0.5, 0.5, "No cosine similarity data available", 
                      ha='center', va='center', fontsize=14)
        ax_cosine.set_title("Feature Cosine Similarity Between Models")
        ax_cosine.axis('off')
    
    # Add overall title
    fig.suptitle("Advanced Feature Analysis based on Transformer Circuits Methodology", 
                fontsize=16, fontweight='bold', y=0.99)
    
    # Add citation and methodology note
    fig.text(0.5, 0.01, 
            "Methodology inspired by 'Transformer Circuits' papers and mechanistic interpretability research", 
            ha='center', fontsize=10, style='italic')
    
    # Adjust layout and save figure
    plt.tight_layout(rect=[0, 0.02, 1, 0.98])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Advanced feature distribution visualization saved to {output_path}")


def _plot_feature_counts(ax, layer_names, base_by_layer, target_by_layer):
    """
    Plot feature counts by layer as a bar chart with enhanced styling.
    """
    # Prepare data for plotting
    base_counts = []
    target_counts = []
    
    for layer in layer_names:
        base_count = len(base_by_layer.get(layer, []))
        target_count = len(target_by_layer.get(layer, []))
        base_counts.append(base_count)
        target_counts.append(target_count)
    
    # Create simplified layer labels
    layer_labels = []
    for layer in layer_names:
        # Extract layer number if possible
        match = re.search(r'(\d+)', layer)
        layer_num = match.group(1) if match else layer
        layer_labels.append(f"Layer {layer_num}")
    
    # Set bar width
    bar_width = 0.35
    
    # Set the position of the bars on the x-axis
    r1 = np.arange(len(layer_labels))
    r2 = [x + bar_width for x in r1]
    
    # Create bars with enhanced styling
    bars1 = ax.bar(r1, base_counts, width=bar_width, label='Base Model', 
                  color='#3498db', alpha=0.8, edgecolor='#2980b9')
    bars2 = ax.bar(r2, target_counts, width=bar_width, label='Target Model', 
                  color='#e74c3c', alpha=0.8, edgecolor='#c0392b')
    
    # Add value labels on top of bars for better readability
    for bar in bars1:
        height = bar.get_height()
        if height > 0:
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                  f'{int(height)}', ha='center', va='bottom', fontsize=9)
            
    for bar in bars2:
        height = bar.get_height()
        if height > 0:
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                  f'{int(height)}', ha='center', va='bottom', fontsize=9)
    
    # Add labels and title with enhanced styling
    ax.set_xlabel('Layer', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Unique Features', fontsize=12, fontweight='bold')
    ax.set_title('Distribution of Model-Specific Features Across Layers', fontsize=14, fontweight='bold')
    ax.set_xticks([r + bar_width/2 for r in range(len(layer_labels))])
    ax.set_xticklabels(layer_labels, rotation=45, ha='right')
    
    # Add grid for better readability
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    
    # Add a legend with better positioning and styling
    ax.legend(loc='upper right', frameon=True, fontsize=10)
    
    # Add summary text
    total_base = sum(base_counts)
    total_target = sum(target_counts)
    ax.text(0.01, 0.97, 
           f"Total features: {total_base + total_target} (Base: {total_base}, Target: {total_target})",
           transform=ax.transAxes, fontsize=10, ha='left', va='top',
           bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))


def _plot_norm_distribution(ax, feature_data, base_features, target_features):
    """
    Plot distribution of decoder norms as histograms with enhanced visualization.
    Inspired by Transformer Circuits papers for norm analysis.
    """
    # Extract norm values for all features
    base_norms = []
    target_norms = []
    norm_ratios = []
    
    for feature_id, norm_info in feature_data['feature_norms'].items():
        base_norm = norm_info.get('base_norm', 0)
        target_norm = norm_info.get('target_norm', 0)
        norm_ratio = norm_info.get('norm_ratio', 0)
        
        # Avoid infinity values
        if norm_ratio == float('inf'):
            norm_ratio = 1000  # Cap at a large value
        
        base_norms.append(base_norm)
        target_norms.append(target_norm)
        norm_ratios.append(norm_ratio)
    
    # Create density plots instead of histograms for smoother visualization
    if len(base_norms) > 1 and len(target_norms) > 1:
        # Use KDE plots if we have enough data
        sns.kdeplot(base_norms, ax=ax, label='Base Model', color='#3498db', shade=True, alpha=0.5)
        sns.kdeplot(target_norms, ax=ax, label='Target Model', color='#e74c3c', shade=True, alpha=0.5)
    else:
        # Fall back to histograms if not enough data
        ax.hist(base_norms, bins=min(30, len(base_norms)), alpha=0.7, label='Base Model', color='#3498db')
        ax.hist(target_norms, bins=min(30, len(target_norms)), alpha=0.7, label='Target Model', color='#e74c3c')
    
    # Add vertical lines for mean values
    if base_norms:
        mean_base = sum(base_norms) / len(base_norms)
        ax.axvline(x=mean_base, color='#2980b9', linestyle='--', 
                  label=f'Mean Base: {mean_base:.3f}')
    
    if target_norms:
        mean_target = sum(target_norms) / len(target_norms)
        ax.axvline(x=mean_target, color='#c0392b', linestyle='--', 
                  label=f'Mean Target: {mean_target:.3f}')
    
    # Add labels and title with enhanced styling
    ax.set_xlabel('Decoder Weight Norm', fontsize=12, fontweight='bold')
    ax.set_ylabel('Density', fontsize=12, fontweight='bold')
    ax.set_title('Distribution of Feature Decoder Norms', fontsize=14, fontweight='bold')
    
    # Add grid for better readability
    ax.grid(axis='both', linestyle='--', alpha=0.3)
    
    # Add a legend with better positioning
    ax.legend(loc='upper right', frameon=True, fontsize=9)
    
    # Add annotation about norm ratios
    if norm_ratios:
        avg_ratio = sum(norm_ratios) / len(norm_ratios)
        median_ratio = sorted(norm_ratios)[len(norm_ratios)//2]
        
        # Create more informative annotation
        ratio_info = (
            f"Norm Ratios (Target/Base):\n"
            f"Mean: {avg_ratio:.2f}\n"
            f"Median: {median_ratio:.2f}\n"
        )
        
        # Add interpretation
        if avg_ratio > 1.2:
            ratio_info += "Interpretation: Target model has stronger features on average"
        elif avg_ratio < 0.8:
            ratio_info += "Interpretation: Base model has stronger features on average"
        else:
            ratio_info += "Interpretation: Similar feature strengths in both models"
            
        # Add the annotation with better styling
        ax.annotate(ratio_info, xy=(0.03, 0.97), xycoords='axes fraction',
                   va='top', ha='left', fontsize=9,
                   bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))


def _plot_feature_density(ax, base_features, target_features, feature_data):
    """
    Plot feature density by model type with enhanced visualization.
    Uses categorization by norm when available for better analysis.
    """
    # Count features by type using the categorization function if norm data is available
    if "feature_norms" in feature_data:
        # Use the categorization function to get better counts
        categorized = categorize_features_by_norm(feature_data)
        base_specific = len(categorized.get('base_specific', []))
        target_specific = len(categorized.get('target_specific', []))
        shared = len(categorized.get('shared', []))
    else:
        # Fallback to simple counts
        base_specific = len(base_features)
        target_specific = len(target_features)
        shared = 0  # We don't have information about shared features
    
    # Prepare data for pie chart
    labels = ['Base-Specific', 'Target-Specific', 'Shared']
    sizes = [base_specific, target_specific, shared]
    colors = ['#3498db', '#e74c3c', '#9b59b6']  # Blue, Red, Purple
    explode = (0.1, 0.1, 0.1)  # explode all slices
    
    # Filter out zero values
    non_zero_indices = [i for i, size in enumerate(sizes) if size > 0]
    non_zero_labels = [labels[i] for i in non_zero_indices]
    non_zero_sizes = [sizes[i] for i in non_zero_indices]
    non_zero_colors = [colors[i] for i in non_zero_indices]
    non_zero_explode = [explode[i] for i in non_zero_indices]
    
    if non_zero_sizes:
        # Create pie chart with enhanced styling
        wedges, texts, autotexts = ax.pie(
            non_zero_sizes, 
            explode=non_zero_explode, 
            labels=non_zero_labels, 
            colors=non_zero_colors,
            autopct='%1.1f%%', 
            shadow=True, 
            startangle=90,
            textprops={'fontsize': 9},
            wedgeprops={'edgecolor': 'w', 'linewidth': 1}
        )
        
        # Style the percentage text
        for autotext in autotexts:
            autotext.set_fontweight('bold')
    else:
        ax.text(0.5, 0.5, "No feature data available", 
               ha='center', va='center', fontsize=14)
    
    # Equal aspect ratio ensures that pie is drawn as a circle
    ax.axis('equal')
    ax.set_title('Feature Density by Model Type', fontsize=14, fontweight='bold')
    
    # Add annotation with totals
    if non_zero_sizes:
        total = sum(non_zero_sizes)
        ax.annotate(f"Total Features: {total}", xy=(0, -0.1), xycoords='axes fraction',
                   ha='center', fontsize=10, fontweight='bold')


def _plot_cosine_similarity(ax, feature_data, crosscoder_data):
    """
    Plot cosine similarity heatmap between feature vectors with enhanced visualization.
    Inspired by Transformer Circuits papers for similarity analysis.
    """
    # Check if we have feature decoders available
    if 'feature_decoders' not in feature_data:
        ax.text(0.5, 0.5, "No decoder weight data available for similarity calculation", 
               ha='center', va='center', fontsize=14)
        ax.set_title("Feature Decoder Similarity")
        return
    
    # Get decoder vectors for each feature
    feature_decoders = feature_data['feature_decoders']
    feature_ids = list(feature_decoders.keys())
    
    # If too many features, select a representative subset
    max_features_to_show = 15
    if len(feature_ids) > max_features_to_show:
        # Try to select features evenly distributed across norm ratios
        if 'feature_norms' in feature_data:
            norm_data = feature_data['feature_norms']
            # Sort features by norm ratio
            sorted_features = sorted(
                [(id, norm_data[id].get('norm_ratio', 1.0)) for id in feature_ids if id in norm_data],
                key=lambda x: x[1]
            )
            
            # Select evenly spaced features
            if sorted_features:
                step = max(1, len(sorted_features) // max_features_to_show)
                selected_features = [sorted_features[i][0] for i in range(0, len(sorted_features), step)]
                # Ensure we don't exceed max and include some high-ratio features
                feature_ids = selected_features[:max_features_to_show-2]
                # Add highest ratio feature if not already included
                if sorted_features and sorted_features[-1][0] not in feature_ids:
                    feature_ids.append(sorted_features[-1][0])
                # Add lowest ratio feature if not already included
                if sorted_features and sorted_features[0][0] not in feature_ids:
                    feature_ids.append(sorted_features[0][0])
        else:
            # Simple approach: just take the first N features
            feature_ids = feature_ids[:max_features_to_show]
    
    # Compute similarity matrix
    n_features = len(feature_ids)
    similarity_matrix = np.zeros((n_features, n_features))
    
    for i, id1 in enumerate(feature_ids):
        for j, id2 in enumerate(feature_ids):
            if i == j:
                similarity_matrix[i, j] = 1.0
            else:
                # Get decoder vectors
                vec1 = np.array(feature_decoders[id1]['base_decoder'])
                vec2 = np.array(feature_decoders[id2]['target_decoder'])
                
                # Compute cosine similarity
                norm1 = np.linalg.norm(vec1)
                norm2 = np.linalg.norm(vec2)
                
                if norm1 > 0 and norm2 > 0:
                    similarity = np.dot(vec1, vec2) / (norm1 * norm2)
                    similarity_matrix[i, j] = similarity
    
    # Create heatmap with enhanced styling
    heatmap = sns.heatmap(
        similarity_matrix, 
        annot=True if n_features <= 10 else False,  # Only show numbers if few features
        cmap='coolwarm', 
        vmin=-1, 
        vmax=1, 
        ax=ax,
        fmt='.2f',
        linewidths=0.5,
        cbar_kws={'label': 'Cosine Similarity'}
    )
    
    # Rotate the ticks if needed
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    plt.setp(ax.get_yticklabels(), rotation=0)
    
    # Shorten feature names for readability
    short_names = []
    for feature_id in feature_ids:
        # Extract a shorter name from the full ID
        if '_' in feature_id:
            parts = feature_id.split('_')
            if len(parts) > 2:
                short_name = f"{parts[0]}_{parts[-1]}"
            else:
                short_name = feature_id
        else:
            short_name = feature_id
        
        # Truncate if still too long
        if len(short_name) > 15:
            short_name = short_name[:12] + "..."
            
        short_names.append(short_name)
    
    # Set axis labels
    ax.set_xticks(np.arange(n_features) + 0.5)
    ax.set_yticks(np.arange(n_features) + 0.5)
    ax.set_xticklabels(short_names, rotation=90)
    ax.set_yticklabels(short_names)
    
    # Add title with enhanced styling
    ax.set_title("Feature Decoder Cross-Model Similarity", fontsize=14, fontweight='bold')
    
    # Add annotation with interpretation
    sim_values = similarity_matrix[~np.eye(n_features, dtype=bool)]  # Exclude diagonals
    avg_similarity = np.mean(sim_values) if sim_values.size > 0 else 0
    
    interpretation = "Interpretation: "
    if avg_similarity > 0.7:
        interpretation += "High similarity between models suggests shared feature representations"
    elif avg_similarity > 0.3:
        interpretation += "Moderate similarity indicates partially shared feature spaces"
    elif avg_similarity > 0:
        interpretation += "Low positive similarity suggests different but related feature spaces"
    else:
        interpretation += "Negative similarity indicates opposing feature representations"
    
    # Add the annotation with better styling
    ax.annotate(
        f"Average Similarity: {avg_similarity:.3f}\n{interpretation}",
        xy=(0.5, -0.15), 
        xycoords='axes fraction',
        ha='center', 
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
    )


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