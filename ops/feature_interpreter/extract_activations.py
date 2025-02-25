"""
Extract activations from models for feature-level interpretation.

This module extracts activations from both base and target models
for a set of prompts, focusing on key layers where differences are significant.
"""

import torch
import numpy as np
import logging
import os
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any, Set, Callable
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..loader.load import load_model_and_tokenizer, get_layer_names

# Configure logging
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

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
            # Handle different output types
            if isinstance(output, tuple):
                # Some modules return tuples, take the first element
                activation = output[0].detach()
            else:
                activation = output.detach()
            
            # Store the activation
            activations[name].append(activation)
        return hook
    
    # Map from layer name patterns to actual module paths
    # This is needed because weight names (like 'model.layers.0.mlp.gate_proj.weight')
    # don't directly map to module paths for hooks
    layer_name_to_module_path = {
        # For transformer models, we typically want to hook into the output of layers
        # rather than the weights themselves
        'model.layers.': 'model.layers.',  # Base pattern
    }
    
    for name in layer_names:
        # Convert weight names to module paths
        module_path = None
        for pattern, path_prefix in layer_name_to_module_path.items():
            if pattern in name:
                # Extract the layer number
                parts = name.split('.')
                for i, part in enumerate(parts):
                    if part == 'layers' and i+1 < len(parts) and parts[i+1].isdigit():
                        layer_num = parts[i+1]
                        # Create a module path for the transformer layer output
                        module_path = f"model.layers.{layer_num}"
                        break
        
        if not module_path:
            logger.warning(f"Could not map layer name {name} to a module path, skipping")
            continue
            
        try:
            # Navigate to the specific module
            module = model
            for part in module_path.split('.'):
                if part.isdigit():
                    module = module[int(part)]
                elif hasattr(module, part):
                    module = getattr(module, part)
                else:
                    logger.warning(f"Could not find module part {part} in {module_path}")
                    module = None
                    break
            
            if module is None:
                continue
                
            # Make sure we're hooking into a module, not a parameter
            if not hasattr(module, 'register_forward_hook'):
                logger.warning(f"Module {module_path} does not support hooks, skipping")
                continue
                
            # Register the hook
            handle = module.register_forward_hook(get_activation(name))
            handles.append(handle)
            logger.info(f"Successfully registered hook for {name} at {module_path}")
        except Exception as e:
            logger.warning(f"Error registering hook for {name}: {str(e)}")
    
    return activations, handles

def extract_activations(
    model_name: str,
    prompts: List[str],
    layer_names: List[str],
    output_dir: str = "feature_activations",
    device: str = "cuda",
    cache_dir: Optional[str] = None,
    quantization: str = "fp16"
) -> Dict:
    """
    Extract activations from a model for a set of prompts.
    
    Args:
        model_name: Name of the model to analyze
        prompts: List of prompts to run through the model
        layer_names: Names of layers to extract activations from
        output_dir: Directory to save activations
        device: Device to use
        cache_dir: Optional cache directory
        quantization: Quantization method to use ("fp32", "fp16", or "int8")
        
    Returns:
        Dictionary mapping prompts to their activations
    """
    logger.info(f"Extracting activations for model: {model_name} with {quantization} quantization")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model and tokenizer with appropriate quantization
    # Note: Our load_model_and_tokenizer only supports use_bf16 parameter
    # We'll adapt our quantization options to work with it
    if quantization == "fp16":
        # For fp16, we'll use bfloat16 which is similar in precision
        use_bf16 = True
    else:
        # For fp32 and int8, we'll load in fp32 first
        use_bf16 = False
    
    # Load the model with the appropriate precision
    model, tokenizer = load_model_and_tokenizer(
        model_name, 
        use_bf16=use_bf16,
        device_map=device,
        cache_dir=cache_dir
    )
    
    # Set pad token if it's not set
    if tokenizer.pad_token is None:
        # If pad token is not set but eos token is, use a different token as pad
        if tokenizer.eos_token is not None:
            # Try to find a suitable token for padding
            if tokenizer.unk_token is not None:
                tokenizer.pad_token = tokenizer.unk_token
            else:
                # Add a new token as pad token
                tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                # Resize model embeddings to match
                model.resize_token_embeddings(len(tokenizer))
    
    # If int8 quantization is requested, we need to handle it separately
    # Note: This is a simplified approach and may not provide true int8 performance
    if quantization == "int8":
        logger.warning("True int8 quantization not supported with current load_model_and_tokenizer. Using fp32 instead.")
    
    # Register hooks
    activations, handles = register_activation_hooks(model, layer_names)
    
    # Prepare results container
    results = {}
    
    try:
        for i, prompt in enumerate(prompts):
            logger.info(f"Processing prompt {i+1}/{len(prompts)}: {prompt[:50]}...")
            
            # Tokenize input
            inputs = tokenizer(
                prompt, 
                return_tensors="pt",
                padding=True,  # Add padding
                truncation=True,  # Truncate if too long
                return_attention_mask=True  # Explicitly return attention mask
            ).to(device)
            
            # Generate output
            with torch.no_grad():
                outputs = model.generate(
                    inputs.input_ids,
                    max_new_tokens=512,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    return_dict_in_generate=True,
                    output_scores=True
                )
            
            # Get the generated tokens
            generated_ids = outputs.sequences[0].tolist()
            generated_text = tokenizer.decode(generated_ids)
            
            # Store activations for this prompt
            prompt_activations = {}
            for layer_name, layer_activations in activations.items():
                if layer_activations:  # Check if we captured any activations
                    # Get the last activation (from generation)
                    layer_output = layer_activations[-1]
                    prompt_activations[layer_name] = layer_output.cpu()
            
            # Store results for this prompt
            results[prompt] = {
                'text': generated_text,
                'activations': prompt_activations
            }
            
            # Clear activations for next prompt
            for name in activations:
                activations[name] = []
            
            # Save incremental results
            with open(f"{output_dir}/{model_name.replace('/', '_')}_activations.pkl", "wb") as f:
                pickle.dump(results, f)
    
    finally:
        # Remove hooks
        for handle in handles:
            handle.remove()
        
        # Clean up
        del model
        torch.cuda.empty_cache()
    
    logger.info(f"Completed activation extraction for {len(prompts)} prompts")
    return results

def extract_activations_for_comparison(
    base_model: str,
    target_model: str,
    prompts: List[str],
    key_layers: Optional[List[str]] = None,
    output_dir: str = "feature_activations",
    device: str = "cuda",
    cache_dir: Optional[str] = None,
    quantization: str = "fp16"
) -> Dict:
    """
    Extract activations from both base and target models for comparison.
    
    Args:
        base_model: Name of the base model
        target_model: Name of the target model
        prompts: List of prompts to process
        key_layers: Optional list of key layers to focus on (if None, middle layers are used)
        output_dir: Directory to save activations
        device: Device to use
        cache_dir: Optional cache directory
        quantization: Quantization method to use
        
    Returns:
        Dictionary with activations from both models
    """
    logger.info(f"Extracting activations for comparison between {base_model} and {target_model}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get layer names
    if key_layers is None:
        # Instead of using weight names, we'll use module paths that can be hooked
        # For most transformer models, we want to hook into the transformer layers
        num_layers = 28  # Default for many models like Qwen2-1.5B
        
        # Create layer names for all layers instead of just the middle third
        key_layers = [f"model.layers.{i}" for i in range(num_layers)]
        
        logger.info(f"Using all layers for analysis: {key_layers}")
    
    # Extract activations from base model
    logger.info(f"Extracting activations from base model: {base_model}")
    base_activations = extract_activations(
        base_model,
        prompts,
        key_layers,
        output_dir,
        device,
        cache_dir,
        quantization
    )
    
    # Extract activations from target model
    logger.info(f"Extracting activations from target model: {target_model}")
    target_activations = extract_activations(
        target_model,
        prompts,
        key_layers,
        output_dir,
        device,
        cache_dir,
        quantization
    )
    
    # Combine results
    results = {}
    
    # Process each prompt
    for prompt in prompts:
        if prompt in base_activations and prompt in target_activations:
            base_output = base_activations[prompt]['text']
            base_act = base_activations[prompt]['activations']
            
            target_output = target_activations[prompt]['text']
            target_act = target_activations[prompt]['activations']
            
            # Store results for this prompt
            results[prompt] = {
                'base_output': base_output,
                'target_output': target_output,
                'base_activations': base_act,
                'target_activations': target_act,
                'layer': key_layers[0] if key_layers else "unknown"  # Use first layer as reference
            }
    
    # Save combined results
    with open(f"{output_dir}/comparison_activations.pkl", "wb") as f:
        pickle.dump(results, f)
    
    logger.info(f"Completed activation extraction for comparison")
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract activations from models for feature-level interpretation")
    parser.add_argument("--base-model", default="Qwen/Qwen2-1.5B", help="Base model name")
    parser.add_argument("--target-model", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", help="Target model name")
    parser.add_argument("--output-dir", default="feature_activations", help="Output directory")
    parser.add_argument("--device", default="cuda", help="Device to use")
    parser.add_argument("--cache-dir", default=str(Path.home() / "backups" / "huggingface_cache"), help="Cache directory")
    parser.add_argument("--quantization", default="fp16", choices=["fp32", "fp16", "int8"], help="Quantization method to use")
    
    args = parser.parse_args()
    
    # Define prompt categories for testing
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
    
    # Extract activations for comparison
    extract_activations_for_comparison(
        args.base_model,
        args.target_model,
        list(prompt_categories.values())[0],
        output_dir=args.output_dir,
        device=args.device,
        cache_dir=args.cache_dir,
        quantization=args.quantization
    ) 