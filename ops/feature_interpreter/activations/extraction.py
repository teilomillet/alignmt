"""
Extraction module for model activations.

This module provides functions to extract activations from language models
given a set of prompts, focusing on specific layers.
"""

import logging
import os
import pickle
import torch
from typing import Dict, List, Optional


from ...loader.load import load_model_and_tokenizer
from .hooks import register_activation_hooks

# Configure logging
logger = logging.getLogger(__name__)

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
        prompts: List of prompts to run through the model (may contain encoded category information)
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
    
    # Process prompt keys to extract actual prompts if they're in the encoded format
    prompt_texts = []
    prompt_mapping = {}  # Maps original prompt keys to extracted prompt texts
    
    for prompt_key in prompts:
        # Check if the prompt is in the encoded format (category/index:prompt_text)
        if '/' in prompt_key and ':' in prompt_key:
            try:
                # Extract the actual prompt text after the colon
                category_and_index, prompt_text_prefix = prompt_key.split(':', 1)
                
                # If the prompt text was truncated, we need to find the original
                if len(prompt_text_prefix) == 50 and prompt_text_prefix.endswith('...'):
                    # This is a truncated prompt, which we can't use directly
                    # Log a warning and use the prefix as-is
                    logger.warning(f"Using truncated prompt text: {prompt_text_prefix}")
                    prompt_text = prompt_text_prefix
                else:
                    # Use the full prompt text
                    prompt_text = prompt_text_prefix
                
                prompt_texts.append(prompt_text)
                prompt_mapping[prompt_key] = prompt_text
            except Exception as e:
                # If there's an error parsing, use the prompt key as-is
                logger.warning(f"Error parsing prompt key '{prompt_key}': {e}. Using as-is.")
                prompt_texts.append(prompt_key)
                prompt_mapping[prompt_key] = prompt_key
        else:
            # If it's not in the encoded format, use it as-is
            prompt_texts.append(prompt_key)
            prompt_mapping[prompt_key] = prompt_key
    
    logger.info(f"Processing {len(prompt_texts)} prompts")
    
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
        for i, prompt in enumerate(prompt_texts):
            logger.info(f"Processing prompt {i+1}/{len(prompt_texts)}: {prompt[:50]}...")
            
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
                    attention_mask=inputs.attention_mask,  # Pass the attention mask
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
            
            # Process and store activations for selected layers
            for layer_name, activation in activations.items():
                if activation:  # Check if we captured any activations
                    # Get the last activation (from generation)
                    layer_output = activation[-1]
                    prompt_activations[layer_name] = layer_output.cpu()
            
            # Store results for this prompt
            # Use the original prompt key from the prompts list
            original_prompt_key = prompts[i]
            results[original_prompt_key] = {
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
    
    logger.info(f"Completed activation extraction for {len(prompt_texts)} prompts")
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
        key_layers: Optional list of key layers to focus on (if None, all layers are used)
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
    
    logger.info("Completed activation extraction for comparison")
    return results 