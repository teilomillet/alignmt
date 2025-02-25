"""
Reasoning feature identification through model diffing.

This module identifies neural features responsible for reasoning capabilities
by comparing base and reasoning-enhanced language models.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any, Set, Callable
from transformers import AutoTokenizer, AutoModelForCausalLM

from ops.crosscoder.crosscode import analyze_layer_changes
from ops.loader.load import get_layer_names, load_model_layer, load_model_and_tokenizer

# Configure logging
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

def find_layer_differences(
    source_model: str,
    target_model: str,
    layer_name: str,
    cache_dir: Optional[str] = None,
    device: str = "cuda"
) -> Dict:
    """
    Find differences between corresponding layers in two models using crosscoder.
    
    Args:
        source_model: Base model name
        target_model: Reasoning-enhanced model name
        layer_name: Name of layer to analyze
        cache_dir: Optional cache directory
        device: Device to use
        
    Returns:
        Dictionary with 'differences', 'similarities', and 'crosscoded' keys.
    """
    logger.info(f"Finding differences for layer: {layer_name}")
    return analyze_layer_changes(
        source=source_model,
        target=target_model,
        layer_name=layer_name,
        cache_dir=cache_dir,
        device=device
    )

def extract_reasoning_features(
    layer_diff: Dict,
    threshold: float = 0.7
) -> Dict[str, float]:
    """
    Extract potential reasoning features from layer differences.
    
    Args:
        layer_diff: Output from find_layer_differences
        threshold: Difference threshold for considering features reasoning-specific
        
    Returns:
        Dictionary mapping parameter names to their difference values
    """
    logger.info(f"Extracting reasoning features with threshold: {threshold}")
    return {
        param: diff_value 
        for param, diff_value in layer_diff['differences'].items()
        if diff_value > threshold  # Focus on features more prevalent in reasoning model
    }

def get_reasoning_tokens(tokenizer) -> Dict[str, List[int]]:
    """
    Get token IDs for reasoning-related markers like "wait", "think", etc.
    
    Args:
        tokenizer: Tokenizer for the model
        
    Returns:
        Dictionary mapping reasoning marker categories to their token IDs
    """
    logger.info("Identifying reasoning-related tokens")
    reasoning_markers = {
        'wait': [' Wait', 'Wait', ' wait', 'wait,'],
        'think': [' Think', 'Think', ' think', 'thinking'],
        'reflect': [' Reflect', 'Reflect', ' reflect'],
        'aha': [' Aha', 'Aha', ' aha', 'aha!'],
        'mistake': [' mistake', 'mistake', 'error'],
        'correct': [' correct', 'correct', ' fix', 'fixed'],
        'revise': [' revise', 'revise', ' revising', 'revision'],
        'reconsider': [' reconsider', 'reconsider', ' reconsidering'],
    }
    
    token_ids = {}
    for marker_name, marker_texts in reasoning_markers.items():
        ids = []
        for text in marker_texts:
            ids.extend(tokenizer.encode(text, add_special_tokens=False))
        token_ids[marker_name] = list(set(ids))  # Remove duplicates
    
    logger.info(f"Found {sum(len(ids) for ids in token_ids.values())} reasoning-related tokens")
    return token_ids

def create_reasoning_prompts() -> List[str]:
    """
    Create example prompts that will trigger reasoning.
    
    Returns:
        List of prompts designed to elicit reasoning behaviors
    """
    logger.info("Creating reasoning prompts")
    return [
        "Solve the equation: 2x + 3 = 7. Show all your steps.",
        "A train travels at 60 mph. How far will it travel in 2.5 hours? Explain your reasoning.",
        "Is the statement 'All prime numbers are odd' true or false? Justify your answer.",
        "If a rectangle has a length of 8 units and a width of 5 units, what is its area? Show your work.",
        "Prove that the square root of 2 is irrational. Walk through each step carefully.",
        "What is the derivative of f(x) = x^3 + 2x^2 - 5x + 3? Show your work.",
        "Solve this logic puzzle: If all A are B, and some B are C, can we conclude that some A are C?",
        "Calculate the compound interest on $1000 invested for 3 years at 5% compounded annually.",
        "Determine if the following argument is valid: All cats have tails. Fluffy has a tail. Therefore, Fluffy is a cat.",
        "Explain why the sum of the angles in a triangle is always 180 degrees."
    ]

def register_activation_hooks(
    model: AutoModelForCausalLM,
    layer_names: List[str]
) -> Tuple[Dict[str, List], List[Callable]]:
    """
    Register hooks to capture activations from specified layers.
    
    Args:
        model: The model to hook into
        layer_names: Names of layers to capture activations from
        
    Returns:
        Tuple of (activations dictionary, list of hook handles)
    """
    logger.info(f"Registering activation hooks for {len(layer_names)} layers")
    activations = {name: [] for name in layer_names}
    handles = []
    
    def get_activation(name):
        def hook(module, input, output):
            activations[name].append(output.detach())
        return hook
    
    for name in layer_names:
        # Parse the layer name to find the corresponding module
        parts = name.split('.')
        module = model
        
        # Navigate to the specific module
        for part in parts:
            if part.isdigit():
                module = module[int(part)]
            elif hasattr(module, part):
                module = getattr(module, part)
            else:
                try:
                    module = module.__getattr__(part)
                except:
                    logger.warning(f"Could not find module for {name}")
                    break
        
        # Register the hook
        handle = module.register_forward_hook(get_activation(name))
        handles.append(handle)
    
    return activations, handles

def trace_activations(
    model_name: str,
    prompts: List[str],
    token_ids: Dict[str, List[int]],
    device: str = "cuda",
    cache_dir: Optional[str] = None,
    quantization: str = "fp16"  # New parameter for quantization method
) -> Dict:
    """
    Trace model activations on reasoning prompts, focusing on reasoning token positions.
    
    Args:
        model_name: Name of the model to trace
        prompts: List of prompts to run through the model
        token_ids: Dictionary of reasoning token IDs to look for
        device: Device to use
        cache_dir: Optional cache directory
        quantization: Quantization method to use ("fp32", "fp16", or "int8")
        
    Returns:
        Dictionary mapping prompts to their activations at reasoning token positions
    """
    logger.info(f"Tracing activations for model: {model_name} with {quantization} quantization")
    
    # Load model and tokenizer with appropriate quantization
    if quantization == "fp16":
        model, tokenizer = load_model_and_tokenizer(
            model_name, 
            device_map=device,
            cache_dir=cache_dir,
            torch_dtype=torch.float16  # Use FP16
        )
    elif quantization == "int8":
        # Use int8 quantization
        model, tokenizer = load_model_and_tokenizer(
            model_name, 
            device_map=device,
            cache_dir=cache_dir,
            load_in_8bit=True  # Use INT8
        )
    else:  # Default to FP32
        model, tokenizer = load_model_and_tokenizer(
            model_name, 
            device_map=device,
            cache_dir=cache_dir
        )
    
    # Get layer names to trace
    all_layer_names = get_layer_names(model_name)
    # Focus on middle layers where reasoning is likely to happen
    middle_layers = all_layer_names[len(all_layer_names)//3:2*len(all_layer_names)//3]
    
    # Register hooks
    activations, handles = register_activation_hooks(model, middle_layers)
    
    # Prepare results container
    results = {}
    
    try:
        for prompt in prompts:
            logger.info(f"Processing prompt: {prompt[:50]}...")
            
            # Tokenize input
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            
            # Generate output with reasoning
            with torch.no_grad():
                outputs = model.generate(
                    inputs.input_ids,
                    max_length=512,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    return_dict_in_generate=True,
                    output_scores=True
                )
            
            # Get the generated tokens
            generated_ids = outputs.sequences[0].tolist()
            generated_text = tokenizer.decode(generated_ids)
            
            # Find positions of reasoning tokens in the output
            reasoning_positions = {}
            for marker_name, marker_ids in token_ids.items():
                positions = []
                for i, token_id in enumerate(generated_ids):
                    if token_id in marker_ids:
                        positions.append(i)
                if positions:
                    reasoning_positions[marker_name] = positions
            
            # Extract activations at reasoning token positions
            prompt_activations = {}
            for layer_name, layer_activations in activations.items():
                if layer_activations:  # Check if we captured any activations
                    # Get the last activation (from generation)
                    layer_output = layer_activations[-1]
                    
                    # Extract activations at reasoning token positions
                    for marker_name, positions in reasoning_positions.items():
                        if marker_name not in prompt_activations:
                            prompt_activations[marker_name] = {}
                        
                        marker_activations = []
                        for pos in positions:
                            if pos < layer_output.size(1):  # Check if position is valid
                                marker_activations.append(layer_output[0, pos].cpu())
                        
                        if marker_activations:
                            prompt_activations[marker_name][layer_name] = torch.stack(marker_activations).mean(0)
            
            # Store results for this prompt
            results[prompt] = {
                'text': generated_text,
                'activations': prompt_activations,
                'reasoning_positions': reasoning_positions
            }
            
            # Clear activations for next prompt
            for name in activations:
                activations[name] = []
    
    finally:
        # Remove hooks
        for handle in handles:
            handle.remove()
        
        # Clean up
        del model
        torch.cuda.empty_cache()
    
    logger.info(f"Completed activation tracing for {len(prompts)} prompts")
    return results

def correlate_features_with_tokens(
    layer_features: Dict[str, Dict],
    token_activations: Dict,
    threshold: float = 0.5
) -> Dict:
    """
    Correlate model differences with token activations to identify reasoning features.
    
    Args:
        layer_features: Dictionary mapping layer names to their feature differences
        token_activations: Dictionary of token activations from trace_activations
        threshold: Correlation threshold for considering features related to reasoning
        
    Returns:
        Dictionary mapping token types to their correlated features
    """
    logger.info(f"Correlating features with token activations, threshold: {threshold}")
    
    # Prepare results container
    correlations = {}
    
    # Process each prompt
    for prompt, prompt_data in token_activations.items():
        prompt_activations = prompt_data.get('activations', {})
        
        # Process each reasoning marker
        for marker_name, marker_activations in prompt_activations.items():
            if marker_name not in correlations:
                correlations[marker_name] = {}
            
            # Process each layer
            for layer_name, activation in marker_activations.items():
                if layer_name in layer_features:
                    layer_diff = layer_features[layer_name]
                    
                    # Correlate features with activation
                    for param_name, diff_value in layer_diff.items():
                        # Extract parameter from the layer name
                        param_key = param_name.split('.')[-1]
                        
                        # Compute correlation
                        # This is a simplified correlation - in practice, you'd need to
                        # extract the actual parameter values and compute proper correlation
                        correlation = diff_value * activation.norm().item()
                        
                        # Store if above threshold
                        if correlation > threshold:
                            if param_key not in correlations[marker_name]:
                                correlations[marker_name][param_key] = []
                            
                            correlations[marker_name][param_key].append({
                                'layer': layer_name,
                                'correlation': correlation,
                                'diff_value': diff_value
                            })
    
    # Sort correlations by strength
    for marker_name in correlations:
        for param_key in correlations[marker_name]:
            correlations[marker_name][param_key].sort(
                key=lambda x: x['correlation'],
                reverse=True
            )
    
    logger.info(f"Found correlations for {len(correlations)} reasoning markers")
    return correlations

def analyze_reasoning_capabilities(
    base_model: str,
    reasoning_model: str,
    output_dir: str = "reasoning_features",
    device: str = "cuda",
    cache_dir: Optional[str] = None,
    feature_threshold: float = 0.7,
    correlation_threshold: float = 0.5,
    save_results: bool = True,
    skip_activations: bool = False,  # New parameter to skip activation tracing
    layer_range: Optional[Tuple[int, int]] = None,  # New parameter to specify layer range
    quantization: str = "fp16"  # New parameter for quantization method
) -> Dict:
    """
    Main function for analyzing reasoning capabilities.
    
    Args:
        base_model: Base model name (e.g., "deepseek-ai/deepseek-v3-base")
        reasoning_model: Reasoning-enhanced model (e.g., "deepseek-ai/DeepSeek-R1")
        output_dir: Directory to save results
        device: Device to use
        cache_dir: Optional cache directory
        feature_threshold: Threshold for identifying reasoning features
        correlation_threshold: Threshold for correlating features with tokens
        save_results: Whether to save results to disk
        skip_activations: Whether to skip activation tracing (useful for hardware limitations)
        layer_range: Optional tuple specifying (start_layer, end_layer) to analyze
        quantization: Quantization method to use ("fp32", "fp16", or "int8")
        
    Returns:
        Dictionary containing analysis results
    """
    logger.info(f"Starting reasoning capability analysis")
    logger.info(f"Base model: {base_model}")
    logger.info(f"Reasoning model: {reasoning_model}")
    logger.info(f"Quantization method: {quantization}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get tokenizer for reasoning model
    tokenizer = AutoTokenizer.from_pretrained(reasoning_model, trust_remote_code=True)
    
    # Get reasoning token IDs
    token_ids = get_reasoning_tokens(tokenizer)
    
    # Generate reasoning prompts
    prompts = create_reasoning_prompts()
    
    # Skip activation tracing if requested
    activations = {}
    if not skip_activations:
        # Trace token activations with specified quantization
        activations = trace_activations(
            reasoning_model, 
            prompts, 
            token_ids, 
            device, 
            cache_dir,
            quantization=quantization
        )
        
        # Save activations
        if save_results:
            logger.info(f"Saving activations to {output_dir}/activations.pkl")
            with open(f"{output_dir}/activations.pkl", "wb") as f:
                pickle.dump(activations, f)
    else:
        logger.info("Skipping activation tracing due to skip_activations=True")
    
    # Get layer names
    layer_names = get_layer_names(base_model)
    
    # Select layers based on layer_range parameter or default to middle layers
    if layer_range:
        start_layer, end_layer = layer_range
        selected_layers = [
            name for name in layer_names 
            if any(f"model.layers.{i}." in name for i in range(start_layer, end_layer + 1))
        ]
        logger.info(f"Analyzing layers {start_layer}-{end_layer} based on specified range")
    else:
        # Default to middle layers
        middle_layers = layer_names[len(layer_names)//3:2*len(layer_names)//3]
        selected_layers = middle_layers
        logger.info(f"Analyzing middle layers (default behavior)")
    
    # Analyze each layer
    layer_features = {}
    for layer_name in selected_layers:
        logger.info(f"Analyzing layer: {layer_name}")
        
        # Find differences
        layer_diff = find_layer_differences(
            base_model, reasoning_model, layer_name, cache_dir, device
        )
        
        # Extract reasoning features
        features = extract_reasoning_features(layer_diff, threshold=feature_threshold)
        
        layer_features[layer_name] = features
        
        # Save incremental results
        if save_results:
            logger.info(f"Saving features for layer {layer_name}")
            with open(f"{output_dir}/features_{layer_name.replace('.', '_')}.pkl", "wb") as f:
                pickle.dump(features, f)
    
    # Skip correlation if we skipped activation tracing
    feature_correlations = {}
    if not skip_activations:
        # Correlate features with token activations
        feature_correlations = correlate_features_with_tokens(
            layer_features, activations, threshold=correlation_threshold
        )
    
    # Compile results
    results = {
        'token_ids': token_ids,
        'layer_features': layer_features,
        'feature_correlations': feature_correlations,
        'metadata': {
            'base_model': base_model,
            'reasoning_model': reasoning_model,
            'feature_threshold': feature_threshold,
            'correlation_threshold': correlation_threshold,
            'skip_activations': skip_activations,
            'layer_range': layer_range,
            'quantization': quantization
        }
    }
    
    # Save results
    if save_results:
        logger.info(f"Saving final results to {output_dir}/reasoning_features.pkl")
        with open(f"{output_dir}/reasoning_features.pkl", "wb") as f:
            pickle.dump(results, f)
    
    logger.info("Analysis complete")
    return results

def visualize_results(
    results: Dict,
    output_dir: str = "reasoning_features"
) -> None:
    """
    Create visualizations of reasoning features.
    
    Args:
        results: Results from analyze_reasoning_capabilities
        output_dir: Directory to save visualizations
    """
    logger.info("Creating visualizations")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract data
    layer_features = results['layer_features']
    feature_correlations = results['feature_correlations']
    metadata = results['metadata']
    
    # 1. Feature distribution across layers
    plt.figure(figsize=(12, 8))
    
    # Count features per layer
    layer_feature_counts = {layer: len(features) for layer, features in layer_features.items()}
    
    # Sort layers by their position in the model
    sorted_layers = sorted(layer_feature_counts.keys(), 
                          key=lambda x: int(x.split('.')[2]) if '.' in x else 0)
    
    # Plot feature counts
    plt.bar(range(len(sorted_layers)), 
            [layer_feature_counts[layer] for layer in sorted_layers])
    plt.xticks(range(len(sorted_layers)), 
              [f"L{layer.split('.')[2]}" if '.' in layer else layer for layer in sorted_layers], 
              rotation=90)
    plt.xlabel('Layer')
    plt.ylabel('Number of Reasoning Features')
    plt.title('Distribution of Reasoning Features Across Layers')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/feature_distribution.png")
    plt.close()
    
    # 2. Feature correlation heatmap
    # Collect all parameter types
    param_types = set()
    for marker in feature_correlations.values():
        param_types.update(marker.keys())
    
    # Collect all reasoning markers
    markers = list(feature_correlations.keys())
    
    if markers and param_types:
        # Create correlation matrix
        correlation_matrix = np.zeros((len(markers), len(param_types)))
        param_types = sorted(param_types)
        
        for i, marker in enumerate(markers):
            for j, param in enumerate(param_types):
                if param in feature_correlations[marker]:
                    # Use the maximum correlation value
                    correlation_matrix[i, j] = max(
                        item['correlation'] for item in feature_correlations[marker][param]
                    )
        
        # Plot heatmap
        plt.figure(figsize=(12, 8))
        plt.imshow(correlation_matrix, cmap='viridis', aspect='auto')
        plt.colorbar(label='Correlation Strength')
        plt.xticks(range(len(param_types)), param_types, rotation=90)
        plt.yticks(range(len(markers)), markers)
        plt.xlabel('Parameter Type')
        plt.ylabel('Reasoning Marker')
        plt.title('Correlation Between Reasoning Markers and Model Parameters')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/correlation_heatmap.png")
        plt.close()
    
    # 3. Top features per reasoning marker
    for marker, params in feature_correlations.items():
        # Collect all correlations across parameters
        all_correlations = []
        for param, items in params.items():
            for item in items:
                all_correlations.append({
                    'param': param,
                    'layer': item['layer'],
                    'correlation': item['correlation'],
                    'diff_value': item['diff_value']
                })
        
        # Sort by correlation strength
        all_correlations.sort(key=lambda x: x['correlation'], reverse=True)
        
        # Take top 10
        top_correlations = all_correlations[:10]
        
        if top_correlations:
            # Plot
            plt.figure(figsize=(12, 8))
            plt.bar(
                range(len(top_correlations)),
                [item['correlation'] for item in top_correlations]
            )
            plt.xticks(
                range(len(top_correlations)),
                [f"{item['param']}\n{item['layer'].split('.')[-3]}" for item in top_correlations],
                rotation=45
            )
            plt.xlabel('Parameter')
            plt.ylabel('Correlation Strength')
            plt.title(f'Top Features Correlated with "{marker}" Reasoning')
            plt.tight_layout()
            plt.savefig(f"{output_dir}/top_features_{marker}.png")
            plt.close()
    
    # 4. Summary visualization
    plt.figure(figsize=(12, 8))
    
    # Count total features per marker
    marker_feature_counts = {}
    for marker, params in feature_correlations.items():
        count = sum(len(items) for items in params.values())
        marker_feature_counts[marker] = count
    
    if marker_feature_counts:
        # Sort by count
        sorted_markers = sorted(marker_feature_counts.keys(), 
                              key=lambda x: marker_feature_counts[x],
                              reverse=True)
        
        # Plot
        plt.bar(
            range(len(sorted_markers)),
            [marker_feature_counts[marker] for marker in sorted_markers]
        )
        plt.xticks(range(len(sorted_markers)), sorted_markers, rotation=45)
        plt.xlabel('Reasoning Marker')
        plt.ylabel('Number of Correlated Features')
        plt.title('Number of Features Correlated with Each Reasoning Marker')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/marker_feature_counts.png")
        plt.close()
    
    logger.info(f"Visualizations saved to {output_dir}")

def main():
    """Command-line interface for reasoning feature analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze reasoning features in language models")
    parser.add_argument("--base-model", default="Qwen/Qwen2-1.5B", help="Base model name")
    parser.add_argument("--reasoning-model", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", help="Reasoning model name")
    parser.add_argument("--output-dir", default="reasoning_features", help="Output directory")
    parser.add_argument("--device", default="cuda", help="Device to use")
    parser.add_argument("--cache-dir", default=str(Path.home() / "backups" / "huggingface_cache"), help="Cache directory")
    parser.add_argument("--feature-threshold", type=float, default=0.7, help="Feature difference threshold")
    parser.add_argument("--correlation-threshold", type=float, default=0.5, help="Correlation threshold")
    parser.add_argument("--no-save", action="store_true", help="Don't save results to disk")
    parser.add_argument("--visualize-only", action="store_true", help="Only visualize existing results")
    parser.add_argument("--skip-activations", action="store_true", help="Skip activation tracing (useful for hardware limitations)")
    parser.add_argument("--start-layer", type=int, default=None, help="Starting layer number to analyze")
    parser.add_argument("--end-layer", type=int, default=None, help="Ending layer number to analyze")
    parser.add_argument("--quantization", default="fp16", choices=["fp32", "fp16", "int8"], help="Quantization method to use")
    
    args = parser.parse_args()
    
    if args.visualize_only:
        # Load existing results
        logger.info(f"Loading existing results from {args.output_dir}/reasoning_features.pkl")
        with open(f"{args.output_dir}/reasoning_features.pkl", "rb") as f:
            results = pickle.load(f)
        
        # Visualize
        visualize_results(results, args.output_dir)
        return
    
    # Determine layer range
    layer_range = None
    if args.start_layer is not None and args.end_layer is not None:
        layer_range = (args.start_layer, args.end_layer)
        logger.info(f"Analyzing layers {args.start_layer}-{args.end_layer}")
    
    # Run full analysis
    results = analyze_reasoning_capabilities(
        base_model=args.base_model,
        reasoning_model=args.reasoning_model,
        output_dir=args.output_dir,
        device=args.device,
        cache_dir=args.cache_dir,
        feature_threshold=args.feature_threshold,
        correlation_threshold=args.correlation_threshold,
        save_results=not args.no_save,
        skip_activations=args.skip_activations,
        layer_range=layer_range,
        quantization=args.quantization
    )
    
    # Visualize results
    visualize_results(results, args.output_dir)

if __name__ == "__main__":
    main() 