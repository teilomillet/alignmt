"""
Run integrated crosscoder + feature interpretation pipeline.

This script runs the integrated pipeline with the actual prompts from the project,
with fixes for the feature interpretation process.
"""

import os
import json
import pickle
from pathlib import Path
import logging

from ops.integrated import IntegratedPipelineConfig, run_integrated_pipeline
from ops.crosscoder.crosscode import analyze_model_changes

# Set up logging to see more detailed output
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("run_analysis")

def main():
    # Define models to compare
    base_model = "Qwen/Qwen2-1.5B"
    target_model = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    
    # Define output directory
    output_dir = "integrated_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Define cache directory (adjust this path if needed)
    cache_dir = str(Path.home() / "backups" / "huggingface_cache")
    
    # Load the prompts file
    prompts_file = os.path.join("ops", "feature_interpreter", "prompts.json")
    
    try:
        with open(prompts_file, 'r') as f:
            prompt_categories = json.load(f)
            logger.info(f"Loaded {len(prompt_categories)} reasoning categories from {prompts_file}")
            
            # For debugging, use just two categories with the most distinctive prompts
            selected_categories = [
                "step_by_step_reasoning",
                "formal_logic",
                "causal_reasoning",
                "probabilistic_reasoning",
                "counterfactual_reasoning",
                "analogical_reasoning",
                "abductive_reasoning",
                "adversarial_reasoning",
                "constraint_satisfaction"
            ]
            
            if selected_categories:
                prompt_categories = {k: prompt_categories[k] for k in selected_categories if k in prompt_categories}
                logger.info(f"Selected {len(prompt_categories)} reasoning categories for analysis")
            
    except Exception as e:
        logger.error(f"Error loading prompts: {e}")
        return
    
    # First run crosscoder analysis separately to ensure we get proper results
    crosscoder_output_dir = os.path.join(output_dir, "crosscoder")
    os.makedirs(crosscoder_output_dir, exist_ok=True)
    
    logger.info("Running standalone crosscoder analysis first...")
    
    # Run crosscoder analysis directly
    crosscoder_results = analyze_model_changes(
        base_model,
        target_model,
        cache_dir=cache_dir,
        device="cuda",
        dtype="float16"
    )
    
    # Save crosscoder results in a format feature interpreter can use
    crosscoder_save_path = os.path.join(crosscoder_output_dir, "crosscoder_results.pkl")
    with open(crosscoder_save_path, 'wb') as f:
        pickle.dump(crosscoder_results, f)
    
    logger.info(f"Saved crosscoder results to {crosscoder_save_path}")
    
    # Log the layer names in crosscoder results for debugging
    crosscoder_layer_names = list(crosscoder_results.keys())
    logger.info(f"Crosscoder analysis contains {len(crosscoder_layer_names)} layers:")
    logger.info(f"Sample layer names: {crosscoder_layer_names[:5] if len(crosscoder_layer_names) > 5 else crosscoder_layer_names}")
    
    # Transform crosscoder results for causal validation format
    # The causal validation expects a dictionary where keys are layer names
    # and values contain validation information
    causal_validation_format = {}
    
    # Function to normalize layer names for consistent matching
    def normalize_layer_name(name):
        # Handle various layer name formats
        if name.startswith("model.layers."):
            return name
        elif name.startswith("layers."):
            return f"model.{name}"
        else:
            # Try to extract just the layer number
            import re
            match = re.search(r'(\d+)', name)
            if match:
                return f"model.layers.{match.group(1)}"
            return name
    
    # Process crosscoder results with normalized layer names
    for layer_name, layer_data in crosscoder_results.items():
        normalized_name = normalize_layer_name(layer_name)
        logger.info(f"Processing crosscoder layer: {layer_name} -> {normalized_name}")
        
        # Extract strength information from the crosscoder data
        # We can derive strength from parameter differences
        if 'differences' in layer_data:
            # Calculate average difference as validation strength
            differences = list(layer_data['differences'].values())
            if differences:
                # For differences, higher values indicate greater change
                strength = sum(differences) / len(differences)
                logger.info(f"  Layer {normalized_name}: avg difference = {strength:.4f}")
                
                # Store both the strength and the raw differences
                causal_validation_format[normalized_name] = {
                    'strength': strength,
                    'validated': strength > 0.5,  # Lower threshold to detect more subtle differences
                    'differences': layer_data['differences']
                }
            else:
                logger.warning(f"  No differences data for layer {normalized_name}")
                
        # Also check similarities as an alternative measure
        elif 'similarities' in layer_data:
            similarities = list(layer_data['similarities'].values())
            if similarities:
                # For similarities, lower values indicate greater change (1.0 means identical)
                sim_strength = 1.0 - (sum(similarities) / len(similarities))
                logger.info(f"  Layer {normalized_name}: avg dissimilarity = {sim_strength:.4f}")
                
                causal_validation_format[normalized_name] = {
                    'strength': sim_strength,
                    'validated': sim_strength > 0.3,  # Threshold for dissimilarity
                    'similarities': layer_data['similarities']
                }
            else:
                logger.warning(f"  No similarities data for layer {normalized_name}")
        else:
            logger.warning(f"  No usable metrics for layer {normalized_name}")
    
    # Check if we have any layers in the validation format
    if not causal_validation_format:
        logger.warning("No layers were processed for causal validation. This will cause validation to fail.")
    else:
        logger.info(f"Prepared {len(causal_validation_format)} layers for causal validation")
        # Log sample of prepared validation data
        sample_layers = list(causal_validation_format.keys())[:3]
        for layer in sample_layers:
            logger.info(f"  {layer}: strength={causal_validation_format[layer]['strength']:.4f}, validated={causal_validation_format[layer]['validated']}")
    
    # Create configuration with adjusted parameters
    config = IntegratedPipelineConfig(
        # Model configs
        base_model=base_model,
        target_model=target_model,
        output_dir=output_dir,
        device="cuda",
        cache_dir=cache_dir,
        quantization="fp16",
        
        # Feature interpreter options - adjust to debug the issue
        skip_activations=False,
        skip_naming=False,
        skip_visualization=False,
        skip_report=False,
        skip_capability_testing=False,
        skip_decoder_analysis=False,
        
        # Analysis parameters - reduce thresholds to catch more differences
        feature_threshold=0.005,  # Even lower threshold to catch more subtle differences
        norm_ratio_threshold=1.1,  # Lower ratio threshold to detect more features
        n_clusters=3,  # Fewer clusters to find more patterns
        report_format="markdown",
        
        # Reasoning-focused prompt categories
        prompt_categories=prompt_categories,
        
        # Crosscoder configs
        crosscoder_output_dir=crosscoder_output_dir,
        skip_crosscoder=True,  # Skip because we already ran it
        crosscoder_param_types=["gate_proj.weight", "up_proj.weight", "down_proj.weight"],  # Add more parameter types
        crosscoder_save_crosscoded_models=True  # Save crosscoded models for better feature analysis
    )
    
    # Add the crosscoder results directly to the config
    # We'll store both formats - original for general use and transformed for causal validation
    setattr(config, 'crosscoder_results', crosscoder_results)
    setattr(config, 'causal_validation_data', causal_validation_format)
    
    logger.info("\nRunning integrated analysis between models:")
    logger.info(f"Base model: {base_model}")
    logger.info(f"Target model: {target_model}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Analysis categories: {', '.join(prompt_categories.keys())}")
    
    # Run the integrated pipeline
    results = run_integrated_pipeline(config)
    
    # Print summary of where to find results
    logger.info("\nAnalysis complete!")
    logger.info(f"Results saved to: {output_dir}")
    logger.info(f"Crosscoder analysis: {os.path.join(output_dir, 'crosscoder')}")
    logger.info(f"Feature interpretation: {os.path.join(output_dir, 'feature_activations')}")
    
    # Check if features were found
    features_path = os.path.join(output_dir, "features.json")
    if os.path.exists(features_path):
        try:
            with open(features_path, 'r') as f:
                features_data = json.load(f)
                if "features" in features_data and features_data["features"]:
                    logger.info(f"Found {len(features_data['features'])} features in {features_path}")
                    
                    # Check if causal validation was performed
                    if "causal_validation" in features_data:
                        logger.info("Causal validation was performed successfully")
                    else:
                        logger.warning("Causal validation results not found in features.json")
                else:
                    logger.warning("No features found in features.json")
        except Exception as e:
            logger.error(f"Error reading features.json: {e}")
    
    return results

if __name__ == "__main__":
    main() 