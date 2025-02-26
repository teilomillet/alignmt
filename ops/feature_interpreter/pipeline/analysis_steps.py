"""
Analysis steps module for the feature interpretation pipeline.

This module provides functions for the various analysis steps in the pipeline,
including capability testing, decoder analysis, and visualization generation.
"""

import os
import json
import logging
import numpy as np
from typing import Dict, Any, Optional

from ..capability.evaluation import evaluate_feature_capability
from ..visualization import create_feature_distribution_plot, create_anthropic_style_visualization
from ..decoder_analysis import generate_comprehensive_analysis
from ..reporting.report_generator import generate_report

logger = logging.getLogger(__name__)

def perform_capability_testing(
    base_model: str,
    target_model: str,
    output_dir: str,
    interpreted_features: Dict[str, Any],
    device: str = "cuda",
    cache_dir: Optional[str] = None,
    skip: bool = False
) -> Dict[str, Any]:
    """
    Test feature capabilities using contrastive examples.
    
    Args:
        base_model: Name of the base model
        target_model: Name of the target model
        output_dir: Directory to save outputs
        interpreted_features: Dictionary of interpreted features
        device: Device to use (cuda or cpu)
        cache_dir: Directory to cache models
        skip: Skip capability testing
        
    Returns:
        Updated features dictionary with capability testing results
    """
    if skip:
        logger.info("Skipping capability testing")
        return interpreted_features
    
    features_path = os.path.join(output_dir, "features.json")
    
    try:
        logger.info("Performing capability testing of features...")
        capability_dir = os.path.join(output_dir, "capability_testing")
        os.makedirs(capability_dir, exist_ok=True)
        
        # Run capability testing
        capability_results = evaluate_feature_capability(
            base_model=base_model,
            target_model=target_model,
            interpreted_features=interpreted_features,
            output_dir=capability_dir,
            device=device,
            cache_dir=cache_dir
        )
        
        # Add capability testing results to interpreted features
        interpreted_features["capability_testing"] = capability_results
        
        # Save updated features
        with open(features_path, "w") as f:
            json.dump(interpreted_features, f, indent=2)
            
        logger.info("Capability testing complete")
    except Exception as e:
        logger.warning(f"Capability testing failed: {str(e)}")
        logger.warning("Continuing without capability testing")
    
    return interpreted_features

def create_visualizations(
    output_dir: str,
    interpreted_features: Dict[str, Any],
    layer_similarities: Dict[str, float],
    skip: bool = False
) -> None:
    """
    Create visualizations of feature differences.
    
    Args:
        output_dir: Directory to save outputs
        interpreted_features: Dictionary of interpreted features
        layer_similarities: Dictionary of layer similarities
        skip: Skip visualization creation
    """
    if skip or not interpreted_features:
        logger.info("Skipping visualization creation")
        return
    
    # Create feature distribution plot
    try:
        output_path = os.path.join(output_dir, "feature_distribution.png")
        create_feature_distribution_plot(
            interpreted_features,
            output_path
        )
        logger.info(f"Feature distribution plot saved to {output_path}")
    except Exception as e:
        logger.warning(f"Failed to create feature distribution plot: {str(e)}")
    
    # Create Anthropic-style visualization
    try:
        output_path = os.path.join(output_dir, "anthropic_style_visualization.png")
        create_anthropic_style_visualization(
            interpreted_features,
            layer_similarities,
            output_path
        )
        logger.info(f"Anthropic-style visualization saved to {output_path}")
    except Exception as e:
        logger.warning(f"Failed to create Anthropic-style visualization: {str(e)}")

def perform_decoder_analysis(
    output_dir: str,
    activations: Dict[str, Any],
    interpreted_features: Dict[str, Any],
    norm_ratio_threshold: float = 1.5,
    n_clusters: int = 5,
    skip: bool = False
) -> Dict[str, Any]:
    """
    Perform decoder weight analysis.
    
    Args:
        output_dir: Directory to save outputs
        activations: Dictionary of activations
        interpreted_features: Dictionary of interpreted features
        norm_ratio_threshold: Threshold for norm ratio to categorize features
        n_clusters: Number of clusters for feature clustering
        skip: Skip decoder weight analysis
        
    Returns:
        Updated features dictionary with decoder analysis results
    """
    if skip or not interpreted_features:
        logger.info("Skipping decoder weight analysis")
        return interpreted_features
    
    logger.info("Performing decoder weight analysis")
    
    # Create directory for decoder analysis results
    decoder_analysis_dir = os.path.join(output_dir, "decoder_analysis")
    os.makedirs(decoder_analysis_dir, exist_ok=True)
    
    # Check if the required data is available
    if not (interpreted_features and
            "feature_decoders" in interpreted_features and
            interpreted_features["feature_decoders"]):
        logger.warning("Feature decoder data not found, skipping decoder analysis")
        return interpreted_features
    
    feature_decoders = interpreted_features["feature_decoders"]
    feature_names = list(feature_decoders.keys())
    
    # Prepare decoder weights matrices
    base_decoder_weights = []
    target_decoder_weights = []
    
    for feature_id in feature_names:
        feature_data = feature_decoders[feature_id]
        if "base_decoder" in feature_data and "target_decoder" in feature_data:
            base_decoder_weights.append(feature_data["base_decoder"])
            target_decoder_weights.append(feature_data["target_decoder"])
    
    if not base_decoder_weights or not target_decoder_weights:
        logger.warning("Decoder weights not found, skipping decoder analysis")
        return interpreted_features
    
    # Convert to numpy arrays
    base_decoder_weights = np.array(base_decoder_weights)
    target_decoder_weights = np.array(target_decoder_weights)
    
    # Extract prompt labels for activation analysis
    prompt_labels = list(activations.keys())
    
    # For a subset of the first prompt examples (to avoid excessive computation)
    sample_prompts = prompt_labels[:20] if len(prompt_labels) > 20 else prompt_labels
    
    # Prepare activation matrices if available
    base_activations = None
    target_activations = None
    
    if all("base_activations" in activations[p] and "target_activations" in activations[p] for p in sample_prompts):
        try:
            # Stack activation matrices for the selected prompts
            base_activations = np.stack([activations[p]["base_activations"] for p in sample_prompts], axis=1)
            target_activations = np.stack([activations[p]["target_activations"] for p in sample_prompts], axis=1)
        except Exception as e:
            logger.warning(f"Failed to prepare activation matrices: {str(e)}")
            logger.warning("Continuing with decoder norm analysis only")
    
    # Generate comprehensive decoder analysis
    try:
        analysis_results = generate_comprehensive_analysis(
            base_decoder=base_decoder_weights,
            target_decoder=target_decoder_weights,
            base_activations=base_activations,
            target_activations=target_activations,
            prompt_labels=sample_prompts if base_activations is not None else None,
            feature_names=feature_names,
            norm_ratio_threshold=norm_ratio_threshold,
            n_clusters=n_clusters,
            output_dir=decoder_analysis_dir
        )
        
        # Add analysis results to interpreted features
        interpreted_features["decoder_analysis"] = analysis_results
        
        logger.info(f"Decoder analysis completed and saved to {decoder_analysis_dir}")
    except Exception as e:
        logger.warning(f"Decoder analysis failed: {str(e)}")
        logger.warning("Continuing without decoder analysis")
    
    return interpreted_features

def generate_final_report(
    base_model: str,
    target_model: str,
    output_dir: str,
    interpreted_features: Dict[str, Any],
    layer_similarities: Dict[str, float],
    report_format: str = "markdown",
    skip: bool = False
) -> None:
    """
    Generate comprehensive report.
    
    Args:
        base_model: Name of the base model
        target_model: Name of the target model
        output_dir: Directory to save outputs
        interpreted_features: Dictionary of interpreted features
        layer_similarities: Dictionary of layer similarities
        report_format: Report format (markdown, html, or both)
        skip: Skip report generation
    """
    if skip or not interpreted_features:
        logger.info("Skipping report generation")
        return
    
    logger.info(f"Generating {report_format} report")
    
    feature_data = {
        "base_model": base_model,
        "target_model": target_model,
        "layer_similarities": layer_similarities,
        **interpreted_features
    }
    
    try:
        generate_report(
            feature_data=feature_data,
            crosscoder_data=None,  # We're not passing crosscoder data directly
            output_dir=output_dir,
            report_format=report_format
        )
        logger.info(f"Report generated in {output_dir}")
    except Exception as e:
        logger.error(f"Failed to generate report: {str(e)}") 