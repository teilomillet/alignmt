# Visualization Module

This module provides tools for visualizing feature interpretation results, helping researchers understand and communicate findings about model differences.

## Key Components

### basic.py
Basic visualization functions:
- `create_feature_distribution_plot`: Creates plots showing the distribution of feature activations
- `create_feature_heatmap`: Generates heatmaps to visualize feature activations across different inputs

### advanced.py
Advanced visualization techniques:
- `create_anthropic_style_visualization`: Creates visualizations similar to those used in Anthropic research papers
- `visualize_interpretable_features`: Creates visualizations that help interpret feature behaviors

### analysis.py
Tools for analyzing visualization data:
- `categorize_features_by_norm`: Categorizes features based on their activation norms
- `calculate_feature_alignment`: Calculates alignment between features across models

### pipeline.py
Orchestrates the visualization workflow:
- `create_visualizations`: Runs a complete visualization pipeline

## Usage

The visualization module is typically used after features have been extracted, named, and analyzed. The workflow involves:

1. Selecting appropriate visualization techniques for the data
2. Generating plots, heatmaps, or interactive visualizations
3. Using these visualizations to gain insights into model differences

These visualizations help researchers communicate their findings about model differences and feature interpretations, making complex neural network behavior more understandable. 