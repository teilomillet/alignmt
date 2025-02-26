"""
Visualization pipeline module.

This module provides the main pipeline for creating all visualizations
for feature interpretation.
"""

import os
import json
import logging
import argparse
from typing import Dict

from .basic import create_feature_distribution_plot, create_feature_heatmap
from .advanced import create_anthropic_style_visualization, visualize_interpretable_features
from .analysis import categorize_features_by_norm, calculate_feature_alignment

# Configure logging
logger = logging.getLogger(__name__)

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
        os.path.join(output_dir, "feature_distribution.png")
    )
    
    # Extract layer similarities or create empty dict if not available
    layer_similarities = feature_data.get("layer_similarities", {})
    
    # Create Anthropic-style visualization
    create_anthropic_style_visualization(
        feature_data,
        layer_similarities,
        os.path.join(output_dir, "feature_anthropic_style.png")
    )
    
    # Create feature heatmap
    create_feature_heatmap(
        feature_data,
        os.path.join(output_dir, "feature_heatmap.png"),
        title="Feature Confidence Heatmap"
    )
    
    # Categorize features by norm
    categorize_features_by_norm(
        feature_data,
        output_path=os.path.join(output_dir, "categorized_features.json")
    )
    
    # Calculate feature alignment
    calculate_feature_alignment(
        feature_data,
        output_path=os.path.join(output_dir, "feature_alignment.png")
    )
    
    # Extract or create placeholder for interpretable features
    interpretable_features = feature_data.get("interpretable_features", {
        "base": {
            "refusal": [],
            "code_review": [],
            "personal_questions": [],
            "roleplay": [],
            "assistant_interactions": []
        },
        "target": {
            "refusal": [],
            "code_review": [],
            "personal_questions": [],
            "roleplay": [],
            "assistant_interactions": []
        }
    })
    
    # Create interpretable features visualization
    visualize_interpretable_features(
        feature_data,
        interpretable_features,
        os.path.join(output_dir, "interpretable_features.png")
    )
    
    logger.info("Visualizations created successfully")


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    parser = argparse.ArgumentParser(description="Create visualizations for feature interpretation")
    parser.add_argument("--feature-file", required=True, help="Path to feature interpretation JSON file")
    parser.add_argument("--output-dir", default="feature_visualizations", help="Output directory")
    parser.add_argument("--norm-ratio-threshold", type=float, default=1.5, 
                        help="Threshold for categorizing features by norm ratio")
    
    args = parser.parse_args()
    
    # Load feature data
    with open(args.feature_file, "r") as f:
        feature_data = json.load(f)
    
    # Create visualizations
    create_visualizations(feature_data, args.output_dir) 