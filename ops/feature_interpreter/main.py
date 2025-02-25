"""
Main script for feature-level model difference interpretation.

This script ties together all components of the feature interpretation pipeline:
1. Extract activations from both models
2. Name and interpret features
3. Create visualizations
4. Generate comprehensive report
"""

import os
import logging
import argparse
import pickle
import json
from pathlib import Path
from typing import Dict, Optional, List
import re
import matplotlib.pyplot as plt
import numpy as np

from .extract_activations import extract_activations_for_comparison
from .feature_naming import name_features, extract_distinctive_features, interpret_feature_differences, causal_feature_validation
from .feature_visualization import create_feature_distribution_plot, create_anthropic_style_visualization
from .generate_report import generate_markdown_report, generate_html_report
from .capability_testing import evaluate_feature_capability

# Configure logging
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

def setup_logging(log_file: str) -> None:
    """
    Set up logging to file and console.
    
    Args:
        log_file: Path to log file
    """
    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    
    # Add file handler to logger
    logger.addHandler(file_handler)
    
    logger.info(f"Logging to {log_file}")

def run_feature_interpretation_pipeline(
    base_model: str,
    target_model: str,
    output_dir: str,
    prompt_categories: Dict[str, List[str]],
    device: str = "cuda",
    cache_dir: Optional[str] = None,
    quantization: str = "fp16",
    skip_activations: bool = False,
    skip_naming: bool = False,
    skip_visualization: bool = False,
    skip_report: bool = False,
    skip_capability_testing: bool = False,
    report_format: str = "both",
    feature_threshold: float = 0.3
) -> Dict:
    """
    Run the feature interpretation pipeline.
    
    Args:
        base_model: Name or path of the base model
        target_model: Name or path of the target model
        output_dir: Directory to save outputs
        prompt_categories: Dictionary mapping categories to lists of prompts
        device: Device to run on (cuda or cpu)
        cache_dir: Directory to cache models
        quantization: Quantization method (fp16, int8, or None)
        skip_activations: Skip activation extraction
        skip_naming: Skip feature naming
        skip_visualization: Skip visualization
        skip_report: Skip report generation
        skip_capability_testing: Skip capability testing
        report_format: Report format (markdown, html, or both)
        feature_threshold: Threshold for feature significance
        
    Returns:
        Dictionary with results
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up logging
    setup_logging(os.path.join(output_dir, "feature_interpreter.log"))
    
    # Flatten prompts
    all_prompts = []
    prompt_to_category = {}
    
    for category, prompts in prompt_categories.items():
        all_prompts.extend(prompts)
        for prompt in prompts:
            prompt_to_category[prompt] = category
    
    # Extract activations
    activations_path = os.path.join(output_dir, "activations.pkl")
    
    if not skip_activations and not os.path.exists(activations_path):
        logger.info(f"Extracting activations for {len(all_prompts)} prompts")
        activations = extract_activations_for_comparison(
            base_model=base_model,
            target_model=target_model,
            prompts=all_prompts,
            device=device,
            cache_dir=cache_dir,
            quantization=quantization
        )
        
        # Save activations
        with open(activations_path, "wb") as f:
            pickle.dump(activations, f)
    else:
        logger.info(f"Loading activations from {activations_path}")
        with open(activations_path, "rb") as f:
            activations = pickle.load(f)
    
    # Compute layer similarities
    layer_similarities = compute_layer_similarities(activations)
    
    # Save layer similarities
    layer_similarities_path = os.path.join(output_dir, "layer_similarities.pkl")
    with open(layer_similarities_path, "wb") as f:
        pickle.dump(layer_similarities, f)
    
    # Create layer similarity plot
    if not skip_visualization:
        create_layer_similarity_plot(
            layer_similarities,
            os.path.join(output_dir, "layer_similarities.png")
        )
    
    # Name features
    if not skip_naming:
        # Compute activation differences for each layer
        activation_differences = {}
        
        # Get a sample prompt to extract layer names
        sample_prompt = list(activations.keys())[0]
        sample_data = activations[sample_prompt]
        
        # Extract layer names from the sample data
        layer_names = []
        for key in sample_data["base_activations"].keys():
            layer_names.append(key)
        
        # Process each layer
        for layer_name in layer_names:
            logger.info(f"Computing activation differences for layer: {layer_name}")
            
            # Prepare data for compute_activation_differences
            base_activations_layer = {}
            target_activations_layer = {}
            
            for prompt, data in activations.items():
                base_activations_layer[prompt] = {
                    'activations': {layer_name: data['base_activations'].get(layer_name)},
                    'text': data['base_output']
                }
                
                target_activations_layer[prompt] = {
                    'activations': {layer_name: data['target_activations'].get(layer_name)},
                    'text': data['target_output']
                }
            
            # Import the function here to avoid circular imports
            from .feature_naming import compute_activation_differences
            
            # Compute differences for this layer
            differences = compute_activation_differences(
                base_activations_layer,
                target_activations_layer,
                layer_name
            )
            
            # Store differences
            for prompt, diff in differences.items():
                # Add the layer name to the difference dictionary
                diff['layer'] = layer_name
                
                if prompt not in activation_differences:
                    activation_differences[prompt] = diff
                else:
                    # If we already have differences for this prompt, use the one with the larger difference
                    if diff['difference'] > activation_differences[prompt]['difference']:
                        activation_differences[prompt] = diff
        
        # Extract distinctive features
        distinctive_features = extract_distinctive_features(
            activation_differences,
            prompt_to_category,
            threshold=feature_threshold
        )
        
        # Incorporate layer similarity data into feature interpretation
        for category, features in distinctive_features.items():
            if "layer" in features and features["layer"] in layer_similarities:
                features["layer_similarity"] = layer_similarities[features["layer"]]
            else:
                features["layer_similarity"] = 0.0
        
        # Interpret features
        interpreted_features = interpret_feature_differences(distinctive_features)
        
        # Save features
        features_path = os.path.join(output_dir, "features.json")
        with open(features_path, "w") as f:
            json.dump(interpreted_features, f, indent=2)
            
        # Perform causal validation of features
        try:
            # We need a model instance for patching
            # For simplicity, we'll use the target model
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
            
            logger.info("Loading model for causal validation...")
            tokenizer = AutoTokenizer.from_pretrained(target_model, cache_dir=cache_dir)
            # Load in evaluation mode
            model = AutoModelForCausalLM.from_pretrained(
                target_model, 
                cache_dir=cache_dir, 
                torch_dtype=torch.float16 if quantization == "fp16" else torch.float32,
                device_map=device
            )
            
            # Run causal validation
            logger.info("Performing causal validation of features...")
            causal_results = causal_feature_validation(
                activations,
                activations,  # Same activations dict for both since it contains both base and target
                interpreted_features,
                tokenizer,
                model,
                device=device
            )
            
            # Add causal validation results to interpreted features
            interpreted_features["causal_validation"] = causal_results
            
            # Save updated features
            with open(features_path, "w") as f:
                json.dump(interpreted_features, f, indent=2)
                
            logger.info("Causal validation complete")
        except Exception as e:
            logger.warning(f"Causal validation failed: {str(e)}")
            logger.warning("Continuing without causal validation")
    
    # Perform capability testing of features
    if not skip_capability_testing:
        try:
            logger.info("Performing capability testing of features...")
            capability_dir = os.path.join(output_dir, "capability_testing")
            
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
    
    # Create visualizations
    if not skip_visualization and interpreted_features:
        # Create feature distribution plot
        create_feature_distribution_plot(
            interpreted_features,
            os.path.join(output_dir, "feature_distribution.png")
        )
        
        # Create Anthropic-style visualization
        create_anthropic_style_visualization(
            interpreted_features,
            layer_similarities,
            os.path.join(output_dir, "anthropic_style_visualization.png")
        )
    
    # Generate report
    if not skip_report and interpreted_features:
        if report_format in ["markdown", "both"]:
            generate_markdown_report(
                base_model=base_model,
                target_model=target_model,
                interpreted_features=interpreted_features,
                layer_similarities=layer_similarities,
                output_path=os.path.join(output_dir, "report.md")
            )
        
        if report_format in ["html", "both"]:
            generate_html_report(
                base_model=base_model,
                target_model=target_model,
                interpreted_features=interpreted_features,
                layer_similarities=layer_similarities,
                output_path=os.path.join(output_dir, "report.html")
            )
    
    return {
        "base_model": base_model,
        "target_model": target_model,
        "output_dir": output_dir,
        "features": interpreted_features,
        "layer_similarities": layer_similarities
    }

def compute_layer_similarities(activations: Dict) -> Dict:
    """
    Compute similarities between corresponding layers in the base and target models.
    
    Args:
        activations: Dictionary with activations
        
    Returns:
        Dictionary mapping layer names to similarity scores
    """
    layer_similarities = {}
    
    # Get a sample prompt to extract layer names
    sample_prompt = list(activations.keys())[0]
    sample_data = activations[sample_prompt]
    
    # Extract layer names
    layer_names = []
    for key in sample_data["base_activations"].keys():
        if isinstance(sample_data["base_activations"][key], np.ndarray):
            layer_names.append(key)
    
    # Compute similarities for each layer
    for layer in layer_names:
        layer_similarities[layer] = {}
        
        # Collect activations for this layer across all prompts
        base_activations = []
        target_activations = []
        
        for prompt, data in activations.items():
            if layer in data["base_activations"] and layer in data["target_activations"]:
                base_act = data["base_activations"][layer]
                target_act = data["target_activations"][layer]
                
                # Ensure activations have the same shape
                if base_act.shape == target_act.shape:
                    base_activations.append(base_act.flatten())
                    target_activations.append(target_act.flatten())
        
        if base_activations and target_activations:
            # Compute average cosine similarity
            similarities = []
            
            for base_act, target_act in zip(base_activations, target_activations):
                # Normalize vectors
                base_norm = np.linalg.norm(base_act)
                target_norm = np.linalg.norm(target_act)
                
                if base_norm > 0 and target_norm > 0:
                    base_act_norm = base_act / base_norm
                    target_act_norm = target_act / target_norm
                    
                    # Compute cosine similarity
                    similarity = np.dot(base_act_norm, target_act_norm)
                    similarities.append(similarity)
            
            if similarities:
                layer_similarities[layer] = float(np.mean(similarities))
            else:
                layer_similarities[layer] = 0.0
        else:
            layer_similarities[layer] = 0.0
    
    return layer_similarities

def create_layer_similarity_plot(
    layer_similarities: Dict,
    output_path: str
) -> None:
    """
    Create a plot showing similarities between corresponding layers.
    
    Args:
        layer_similarities: Dictionary mapping layer names to similarity scores
        output_path: Path to save the plot
    """
    # Extract layer names and similarities
    layers = []
    similarities = []
    
    for layer, similarity in layer_similarities.items():
        # Extract layer number if possible
        match = re.search(r'(\d+)', layer)
        if match:
            layer_num = int(match.group(1))
            layers.append((layer_num, layer))
        else:
            # If no number found, use a large number to place at the end
            layers.append((999, layer))
    
    # Sort layers by number
    layers.sort()
    
    # Extract similarities in the sorted order
    layer_names = [layer[1] for layer in layers]
    similarities = [layer_similarities[layer] for layer in layer_names]
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(layer_names)), similarities)
    plt.xlabel('Layer')
    plt.ylabel('Similarity')
    plt.title('Layer-wise Similarity Between Base and Target Models')
    
    # Create x-tick labels
    layer_labels = []
    for layer in layer_names:
        # Extract layer number if possible
        match = re.search(r'(\d+)', layer)
        if match:
            layer_labels.append(f"Layer {match.group(1)}")
        else:
            layer_labels.append(layer)
    
    plt.xticks(range(len(layer_names)), layer_labels, rotation=45, ha='right')
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(output_path)
    plt.close()

def main():
    """Command-line interface for feature interpretation pipeline."""
    parser = argparse.ArgumentParser(description="Feature-level model difference interpretation")
    parser.add_argument("--base-model", default="Qwen/Qwen2-1.5B", help="Base model name")
    parser.add_argument("--target-model", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", help="Target model name")
    parser.add_argument("--crosscoder-file", help="Path to crosscoder analysis file")
    parser.add_argument("--output-dir", default="feature_interpretation", help="Output directory")
    parser.add_argument("--device", default="cuda", help="Device to use")
    parser.add_argument("--cache-dir", default=str(Path.home() / "backups" / "huggingface_cache"), help="Cache directory")
    parser.add_argument("--quantization", default="fp16", choices=["fp32", "fp16", "int8"], help="Quantization method to use")
    parser.add_argument("--skip-activations", action="store_true", help="Skip activation extraction")
    parser.add_argument("--skip-naming", action="store_true", help="Skip feature naming")
    parser.add_argument("--skip-visualization", action="store_true", help="Skip visualization creation")
    parser.add_argument("--skip-report", action="store_true", help="Skip report generation")
    parser.add_argument("--skip-capability-testing", action="store_true", help="Skip capability testing")
    parser.add_argument("--report-format", default="both", choices=["markdown", "html", "both"], help="Report format")
    parser.add_argument("--feature-threshold", type=float, default=0.3, help="Threshold for significant differences")
    parser.add_argument("--prompts-file", default="ops/feature_interpreter/prompts.json", help="Path to prompts JSON file")
    
    args = parser.parse_args()
    
    # Load prompt categories from JSON file
    try:
        with open(args.prompts_file, 'r') as f:
            prompt_categories = json.load(f)
        logger.info(f"Successfully loaded prompts from {args.prompts_file}")
    except Exception as e:
        logger.warning(f"Failed to load prompts from {args.prompts_file}: {str(e)}")
        logger.warning("Falling back to default prompt categories")
        # Define default prompt categories as fallback
        prompt_categories = {
            "reasoning": [
                "Solve the equation: 2x + 3 = 7. Show all your steps.",
                "A train travels at 60 mph. How far will it travel in 2.5 hours? Explain your reasoning."
            ],
            "instruction_following": [
                "Write a short poem about artificial intelligence.",
                "List five benefits of regular exercise."
            ],
            "factual_knowledge": [
                "What is the capital of France?",
                "Who wrote the novel 'Pride and Prejudice'?"
            ]
        }
    
    # Run pipeline
    run_feature_interpretation_pipeline(
        base_model=args.base_model,
        target_model=args.target_model,
        output_dir=args.output_dir,
        prompt_categories=prompt_categories,
        device=args.device,
        cache_dir=args.cache_dir,
        quantization=args.quantization,
        skip_activations=args.skip_activations,
        skip_naming=args.skip_naming,
        skip_visualization=args.skip_visualization,
        skip_report=args.skip_report,
        skip_capability_testing=args.skip_capability_testing,
        report_format=args.report_format,
        feature_threshold=args.feature_threshold
    )

if __name__ == "__main__":
    main() 