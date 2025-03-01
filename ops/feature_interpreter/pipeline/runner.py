"""
Pipeline runner module for feature interpretation.

This module provides the main runner function that orchestrates
the entire feature interpretation pipeline.
"""

import os
import pickle
import logging
from typing import Dict, Any

from .config import PipelineConfig
from .utils import setup_logging, extract_or_load_activations, flatten_prompts
from .layer_analysis import compute_layer_similarities, create_layer_similarity_plot
from .feature_analysis import get_activation_differences, perform_feature_analysis, perform_causal_validation
from .analysis_steps import (
    perform_capability_testing,
    create_visualizations, 
    perform_decoder_analysis,
    generate_final_report
)

logger = logging.getLogger(__name__)

def run_feature_interpretation_pipeline(config: PipelineConfig) -> Dict[str, Any]:
    """
    Run the feature interpretation pipeline.
    
    Args:
        config: Pipeline configuration
        
    Returns:
        Dictionary with results
    """
    # Set up logging
    setup_logging(os.path.join(config.output_dir, "feature_interpreter.log"))
    
    # Flatten prompts
    all_prompts, prompt_to_category = flatten_prompts(config.prompt_categories)
    
    logger.info("Running feature interpretation pipeline:")
    logger.info(f"  Base model: {config.base_model}")
    logger.info(f"  Target model: {config.target_model}")
    logger.info(f"  Output directory: {config.output_dir}")
    logger.info(f"  Number of prompts: {len(all_prompts)}")
    
    # Step 1: Extract activations
    activations = extract_or_load_activations(
        base_model=config.base_model,
        target_model=config.target_model,
        output_dir=config.output_dir,
        all_prompts=all_prompts,
        device=config.device,
        cache_dir=config.cache_dir,
        quantization=config.quantization,
        skip_activations=config.skip_activations
    )
    
    # Step 2: Compute layer similarities
    layer_similarities = compute_layer_similarities(activations)
    
    # Save layer similarities
    layer_similarities_path = os.path.join(config.output_dir, "layer_similarities.pkl")
    with open(layer_similarities_path, "wb") as f:
        pickle.dump(layer_similarities, f)
    
    # Step 3: Create layer similarity plot
    if not config.skip_visualization:
        create_layer_similarity_plot(
            layer_similarities,
            os.path.join(config.output_dir, "layer_similarities.png")
        )
    
    # Skip feature naming if requested
    if config.skip_naming:
        logger.info("Skipping feature naming")
        features_path = os.path.join(config.output_dir, "features.json")
        
        # Try to load existing features
        if os.path.exists(features_path):
            import json
            with open(features_path, 'r') as f:
                interpreted_features = json.load(f)
                logger.info(f"Loaded existing features from {features_path}")
        else:
            interpreted_features = {}
            logger.warning("No existing features found, skipping feature analysis")
    else:
        # Step 4: Compute activation differences
        activation_differences = get_activation_differences(
            activations, 
            prompt_to_category
        )
        
        # Step 5: Analyze and interpret features
        interpreted_features = perform_feature_analysis(
            output_dir=config.output_dir,
            activation_differences=activation_differences,
            prompt_to_category=prompt_to_category,
            layer_similarities=layer_similarities,
            feature_threshold=config.feature_threshold
        )
        
        # Step 6: Perform causal validation
        # Check if we have crosscoder results available from the integrated pipeline
        crosscoder_analysis = None
        if hasattr(config, 'causal_validation_data'):
            logger.info("Using pre-formatted causal validation data from integrated pipeline")
            crosscoder_analysis = config.causal_validation_data
        elif hasattr(config, 'crosscoder_results'):
            logger.info("Using crosscoder results from integrated pipeline for causal validation")
            crosscoder_analysis = config.crosscoder_results
        
        interpreted_features = perform_causal_validation(
            model_path=config.target_model,  # Use target model for validation
            output_dir=config.output_dir,
            interpreted_features=interpreted_features,
            crosscoder_analysis=crosscoder_analysis
        )
    
    # Step 7: Perform capability testing
    interpreted_features = perform_capability_testing(
        base_model=config.base_model,
        target_model=config.target_model,
        output_dir=config.output_dir,
        interpreted_features=interpreted_features,
        device=config.device,
        cache_dir=config.cache_dir,
        skip=config.skip_capability_testing
    )
    
    # Step 9: Perform decoder analysis
    interpreted_features = perform_decoder_analysis(
        output_dir=config.output_dir,
        activations=activations,
        interpreted_features=interpreted_features,
        norm_ratio_threshold=config.norm_ratio_threshold,
        n_clusters=config.n_clusters,
        skip=config.skip_decoder_analysis
    )
    
    # Extract feature data from decoder analysis for visualization
    feature_data = None
    crosscoder_data = None
    
    if not config.skip_decoder_analysis and "decoder_analysis" in interpreted_features:
        feature_data = interpreted_features["decoder_analysis"].get("feature_data", None)
        logger.info("Found decoder analysis feature data for enhanced visualization")
    
    # Get crosscoder data if available
    if hasattr(config, 'crosscoder_results'):
        crosscoder_data = config.crosscoder_results
        logger.info("Found crosscoder data for enhanced visualization")
    
    # Step 8: Create visualizations with enhanced data
    create_visualizations(
        output_dir=config.output_dir,
        interpreted_features=interpreted_features,
        layer_similarities=layer_similarities,
        skip=config.skip_visualization,
        feature_data=feature_data,
        crosscoder_data=crosscoder_data
    )
    
    # Step 10: Generate report
    generate_final_report(
        base_model=config.base_model,
        target_model=config.target_model,
        output_dir=config.output_dir,
        interpreted_features=interpreted_features,
        layer_similarities=layer_similarities,
        report_format=config.report_format,
        skip=config.skip_report
    )
    
    return {
        "base_model": config.base_model,
        "target_model": config.target_model,
        "output_dir": config.output_dir,
        "features": interpreted_features,
        "layer_similarities": layer_similarities,
        "decoder_analysis": interpreted_features.get("decoder_analysis", None)
    } 