"""
Analysis steps module for the feature interpretation pipeline.

This module provides functions for the various analysis steps in the pipeline,
including capability testing, decoder analysis, and visualization generation.
"""

import os
import json
import logging
import numpy as np
from typing import Dict, Any, List, Optional

# Get evaluate_feature_capability through the lazy import function
from ..capability import get_evaluation_functions
_, evaluate_feature_capability = get_evaluation_functions()

from ..visualization import create_feature_distribution_plot, create_reasoning_category_visualization
from ..decoder_analysis import generate_comprehensive_analysis, extract_feature_decoder_norms
from ..reporting.report_generator import generate_report
from ..naming import (
    compute_activation_differences, 
    interpret_feature_differences,
    extract_distinctive_features
)

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
    skip: bool = False,
    feature_data: Optional[Dict] = None,
    crosscoder_data: Optional[Dict] = None
) -> None:
    """
    Create visualizations of feature differences.
    
    Args:
        output_dir: Directory to save outputs
        interpreted_features: Dictionary of interpreted features
        layer_similarities: Dictionary of layer similarities
        skip: Skip visualization creation
        feature_data: Optional dictionary with feature decoder norm data
        crosscoder_data: Optional dictionary with cosine similarity data between models
    """
    if skip or not interpreted_features:
        logger.info("Skipping visualization creation")
        return
    
    # Create visualization directory if it doesn't exist
    visualization_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(visualization_dir, exist_ok=True)
    
    # Check decoder_analysis data for feature data if not explicitly provided
    if feature_data is None and "decoder_analysis" in interpreted_features:
        decoder_analysis = interpreted_features["decoder_analysis"]
        if isinstance(decoder_analysis, dict):
            # First check if feature_data is directly in decoder_analysis
            if "feature_data" in decoder_analysis:
                feature_data = decoder_analysis["feature_data"]
                logger.info("Using feature data from decoder analysis for enhanced visualization")
            
            # Or if it contains feature_norms directly
            elif "feature_norms" in decoder_analysis:
                feature_data = decoder_analysis
                logger.info("Using feature norms from decoder analysis for visualization")
    
    # Log what data we have for visualizations
    if feature_data:
        logger.info("Feature data available for enhanced visualization")
        if "feature_norms" in feature_data:
            logger.info(f"Found {len(feature_data['feature_norms'])} feature norms")
        if "feature_decoders" in feature_data:
            logger.info(f"Found {len(feature_data['feature_decoders'])} feature decoders")
    
    if crosscoder_data:
        logger.info(f"Crosscoder data available with {len(crosscoder_data)} layers")
    
    # Create feature distribution plot
    try:
        output_path = os.path.join(visualization_dir, "feature_distribution.png")
        create_feature_distribution_plot(
            interpreted_features,
            output_path,
            feature_data=feature_data,
            crosscoder_data=crosscoder_data,
            include_advanced_visualizations=(feature_data is not None)
        )
        logger.info(f"Feature distribution plot saved to {output_path}")
    except Exception as e:
        logger.warning(f"Failed to create feature distribution plot: {str(e)}")
        logger.exception(e)  # Log full traceback for easier debugging
    
    # If we have feature data, also create a dedicated advanced visualization
    if feature_data:
        try:
            # Create advanced visualization in a separate file
            advanced_output_path = os.path.join(visualization_dir, "feature_distribution_advanced.png")
            
            # Extract base and target features
            if "features" in interpreted_features and isinstance(interpreted_features["features"], list):
                # New format
                features_list = interpreted_features["features"]
                base_features = [f for f in features_list if f.get("model_attribution", "").lower() == "base"]
                target_features = [f for f in features_list if f.get("model_attribution", "").lower() == "target"]
            else:
                # Old format
                base_features = interpreted_features.get("base_model_specific_features", [])
                target_features = interpreted_features.get("target_model_specific_features", [])
            
            from ..visualization.basic import create_advanced_feature_distribution_plot
            
            create_advanced_feature_distribution_plot(
                base_features=base_features,
                target_features=target_features,
                feature_data=feature_data,
                crosscoder_data=crosscoder_data,
                output_path=advanced_output_path
            )
            logger.info(f"Advanced feature distribution visualization saved to {advanced_output_path}")
        except Exception as e:
            logger.warning(f"Failed to create advanced feature visualization: {str(e)}")
            logger.exception(e)
    
    # Create reasoning category visualization
    try:
        output_path = os.path.join(visualization_dir, "reasoning_category_visualization.png")
        create_reasoning_category_visualization(
            interpreted_features,
            layer_similarities,
            output_path
        )
        logger.info(f"Reasoning category visualization saved to {output_path}")
    except Exception as e:
        logger.warning(f"Failed to create reasoning category visualization: {str(e)}")
        
    # If we have feature data with norms, create a feature heatmap
    if feature_data and "feature_norms" in feature_data:
        try:
            output_path = os.path.join(visualization_dir, "feature_heatmap.png")
            from ..visualization.basic import create_feature_heatmap
            
            create_feature_heatmap(
                feature_data=feature_data,
                output_path=output_path,
                title="Feature Confidence Heatmap"
            )
            logger.info(f"Feature heatmap saved to {output_path}")
        except Exception as e:
            logger.warning(f"Failed to create feature heatmap: {str(e)}")
            
    # Create a symlink to the visualization directory at the top level for easier access
    top_level_link = os.path.join(output_dir, "visualizations.png")
    main_visualization = os.path.join(visualization_dir, "feature_distribution.png")
    if os.path.exists(main_visualization):
        try:
            # Remove existing link if any
            if os.path.exists(top_level_link):
                os.remove(top_level_link)
            # Create relative symlink for better portability
            os.symlink(os.path.relpath(main_visualization, output_dir), top_level_link)
            logger.info(f"Created symlink to main visualization at {top_level_link}")
        except Exception as e:
            logger.warning(f"Failed to create visualization symlink: {str(e)}")
            
    # Create an HTML report with all visualizations
    try:
        html_report_path = os.path.join(output_dir, "visualization_report.html")
        
        # Collect all visualization files
        visualization_files = [f for f in os.listdir(visualization_dir) if f.endswith('.png')]
        
        # Create a simple HTML report
        with open(html_report_path, 'w') as f:
            f.write("<!DOCTYPE html>\n")
            f.write("<html>\n<head>\n")
            f.write("<title>Feature Interpretation Visualizations</title>\n")
            f.write("<style>body{font-family:Arial,sans-serif;margin:20px;max-width:1200px;margin:0 auto;}")
            f.write("h1,h2{color:#333}img{max-width:100%;margin:10px 0;border:1px solid #ddd;}")
            f.write(".viz-container{margin:20px 0;padding:20px;background:#f9f9f9;border-radius:5px;}")
            f.write("</style>\n</head>\n<body>\n")
            
            f.write("<h1>Feature Interpretation Visualizations</h1>\n")
            
            for viz_file in visualization_files:
                # Extract name without extension for the heading
                name_parts = viz_file.split('.')
                if len(name_parts) > 1:
                    viz_name = name_parts[0].replace('_', ' ').title()
                else:
                    viz_name = viz_file
                    
                f.write(f"<div class='viz-container'>\n")
                f.write(f"<h2>{viz_name}</h2>\n")
                # Use relative path for better portability
                f.write(f"<img src='visualizations/{viz_file}' alt='{viz_name}'>\n")
                f.write("</div>\n")
            
            f.write("</body>\n</html>")
            
        logger.info(f"Created HTML visualization report at {html_report_path}")
    except Exception as e:
        logger.warning(f"Failed to create visualization HTML report: {str(e)}")

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
    
    # First check if we have feature_decoders
    if "feature_decoders" in interpreted_features and interpreted_features["feature_decoders"]:
        # Original implementation for feature_decoders
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
        
        # Instead of limiting to first 20 prompts, use all prompts or implement stratified sampling
        # First, try to extract category information from prompts if available
        prompt_to_category = {}
        try:
            # Extract category from prompt name if it follows a pattern like "category_name/prompt_text"
            for prompt in prompt_labels:
                if '/' in prompt:
                    category, _ = prompt.split('/', 1)
                    prompt_to_category[prompt] = category
        except Exception as e:
            logger.warning(f"Could not extract categories from prompts: {e}")
        
        # If we have category information, do stratified sampling
        if prompt_to_category and len(set(prompt_to_category.values())) > 1:
            logger.info("Performing stratified sampling of prompts by category")
            prompts_by_category = {}
            for prompt, category in prompt_to_category.items():
                if category not in prompts_by_category:
                    prompts_by_category[category] = []
                prompts_by_category[category].append(prompt)
            
            # Take an equal number from each category, up to 5 per category
            sample_prompts = []
            samples_per_category = 5  # Adjust if you want more or fewer samples per category
            for category, prompts in prompts_by_category.items():
                category_samples = prompts[:samples_per_category]
                sample_prompts.extend(category_samples)
                logger.info(f"Added {len(category_samples)} prompts from category '{category}'")
            
            logger.info(f"Using {len(sample_prompts)} prompts from {len(prompts_by_category)} categories for decoder analysis")
        else:
            # If no category information, use all prompts (up to a reasonable limit to avoid memory issues)
            max_prompts = 50  # Increased from 20 to 50, adjust based on memory availability
            if len(prompt_labels) > max_prompts:
                logger.warning(f"Too many prompts ({len(prompt_labels)}), using first {max_prompts} for decoder analysis")
                sample_prompts = prompt_labels[:max_prompts]
            else:
                logger.info(f"Using all {len(prompt_labels)} prompts for decoder analysis")
                sample_prompts = prompt_labels
        
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
    
    # If we don't have feature_decoders but we have features in the new format, create a basic analysis
    elif "features" in interpreted_features and isinstance(interpreted_features["features"], list):
        features = interpreted_features["features"]
        
        # Create a simple report summarizing features without decoder analysis
        analysis_results = {
            "summary": f"Basic feature analysis without decoder weights. Found {len(features)} features.",
            "features": {}
        }
        
        for i, feature in enumerate(features):
            feature_id = feature.get("name", f"feature_{i}")
            layer = feature.get("layer", "unknown")
            
            analysis_results["features"][feature_id] = {
                "layer": layer,
                "avg_difference": feature.get("avg_difference", 0.0),
                "significance": feature.get("significance", 0.0),
                "num_prompts": len(feature.get("prompts", [])),
                "interpretation": "No decoder analysis available for this feature."
            }
        
        # Save basic analysis
        with open(os.path.join(decoder_analysis_dir, "basic_feature_summary.json"), "w") as f:
            json.dump(analysis_results, f, indent=2)
        
        # Add analysis results to interpreted features
        interpreted_features["decoder_analysis"] = analysis_results
        
        logger.info(f"Basic feature summary created without decoder analysis")
    else:
        logger.warning("Neither feature_decoders nor features found, skipping decoder analysis")
    
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

def run_integrated_analysis(
    output_dir: str,
    base_activations: Dict[str, Any],
    target_activations: Dict[str, Any],
    output_analyses: Optional[Dict[str, Any]] = None,
    layer_similarities: Optional[Dict[str, float]] = None,
    crosscoder_data: Optional[Dict] = None,
    skip_capability_testing: bool = False,
    skip_visualization: bool = False,
    skip_decoder_analysis: bool = False,
    feature_threshold: float = 0.1,
    norm_ratio_threshold: float = 1.5,
    n_clusters: int = 5
) -> Dict[str, Any]:
    """
    Run the complete feature interpretation pipeline.
    
    Args:
        output_dir: Directory to save outputs
        base_activations: Dictionary of base model activations
        target_activations: Dictionary of target model activations
        output_analyses: Optional analyses of output differences
        layer_similarities: Optional dictionary of layer similarities
        crosscoder_data: Optional crosscoder analysis data
        skip_capability_testing: Skip capability testing
        skip_visualization: Skip visualization creation
        skip_decoder_analysis: Skip decoder weight analysis
        feature_threshold: Threshold for feature detection
        norm_ratio_threshold: Threshold for norm ratio in decoder analysis
        n_clusters: Number of clusters for feature clustering
        
    Returns:
        Dictionary with interpreted features
    """
    # ... keep existing code ...
    # After decoder analysis
    
    # Create directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Compute activation differences
    activation_differences = {}
    for prompt in base_activations:
        if prompt not in target_activations:
            continue
            
        # Convert activations to the expected format
        base_prompt_activations = base_activations[prompt]
        target_prompt_activations = target_activations[prompt]
        
        # Compute differences for each layer
        for layer_name in base_prompt_activations.get("activations", {}):
            if layer_name not in target_prompt_activations.get("activations", {}):
                continue
                
            # Extract activations for this layer
            base_layer_act = {
                prompt: {"activations": {layer_name: base_prompt_activations["activations"][layer_name]}, 
                         "text": base_prompt_activations.get("text", "")}
            }
            target_layer_act = {
                prompt: {"activations": {layer_name: target_prompt_activations["activations"][layer_name]}, 
                         "text": target_prompt_activations.get("text", "")}
            }
            
            # Compute differences
            differences = compute_activation_differences(base_layer_act, target_layer_act, layer_name)
            
            # Merge differences into the main dictionary
            if prompt in differences:
                if prompt not in activation_differences:
                    activation_differences[prompt] = differences[prompt]
                elif differences[prompt]["difference"] > activation_differences[prompt]["difference"]:
                    activation_differences[prompt] = differences[prompt]
    
    # Step 2: Interpret features
    prompt_specific_analyses = {prompt: output_analyses.get(prompt, {}) for prompt in activation_differences} if output_analyses else None
    interpreted_features = interpret_feature_differences(
        activation_differences=activation_differences,
        output_analyses=prompt_specific_analyses,
        threshold=feature_threshold
    )
    
    # Step 3: Evaluate capabilities
    interpreted_features = evaluate_feature_capability(
        base_model=None,  # Not needed for basic capability tests
        target_model=None,  # Not needed for basic capability tests
        interpreted_features=interpreted_features,
        output_dir=None,  # Will be automatically created if needed
        device=None,  # Not needed for basic capability tests
        cache_dir=None,  # Not needed for basic capability tests
        skip=skip_capability_testing
    )
    
    # Step 4: Perform decoder analysis
    feature_data = None
    if not skip_decoder_analysis:
        # Run decoder analysis
        interpreted_features = perform_decoder_analysis(
            output_dir=output_dir,
            activations={
                prompt: {
                    "base_activations": base_activations[prompt].get("activations", {}),
                    "target_activations": target_activations[prompt].get("activations", {})
                } for prompt in base_activations if prompt in target_activations
            },
            interpreted_features=interpreted_features,
            norm_ratio_threshold=norm_ratio_threshold,
            n_clusters=n_clusters,
            skip=skip_decoder_analysis
        )
        
        # Extract feature data for visualization
        if "decoder_analysis" in interpreted_features:
            feature_data = interpreted_features["decoder_analysis"].get("feature_data", None)
    
    # Step 5: Create visualizations
    if not skip_visualization:
        create_visualizations(
            output_dir=output_dir,
            interpreted_features=interpreted_features,
            layer_similarities=layer_similarities or {},
            skip=skip_visualization,
            feature_data=feature_data,
            crosscoder_data=crosscoder_data
        )
    
    # Save final results
    with open(os.path.join(output_dir, "interpreted_features.json"), "w") as f:
        json.dump(interpreted_features, f, indent=2)
    
    return interpreted_features 