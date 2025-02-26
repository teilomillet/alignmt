"""
Evaluation module for capability testing.

This module provides functions to load models, generate responses, 
and evaluate model capabilities through contrastive testing.
"""

import logging
import json
import os
import torch
from typing import Dict, List, Any, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer

from .metrics import calculate_human_experience_score

# Configure logging
logger = logging.getLogger(__name__)

def generate_response(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int = 512
) -> str:
    """
    Generate a response from a model given a prompt.
    
    Args:
        model: The model to use
        tokenizer: The tokenizer to use
        prompt: The prompt to generate from
        max_new_tokens: Maximum number of new tokens to generate
        
    Returns:
        Generated text
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False
        )
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response

def evaluate_feature_capability(
    base_model: str,
    target_model: str, 
    interpreted_features: Dict,
    output_dir: str,
    device: str = "cuda",
    cache_dir: Optional[str] = None,
    num_examples_per_feature: int = 2
) -> Dict:
    """
    Evaluate capabilities associated with identified features through contrastive testing.
    
    Args:
        base_model: Name of the base model
        target_model: Name of the target model
        interpreted_features: Dictionary with interpreted features
        output_dir: Directory to save evaluation results
        device: Device to use
        cache_dir: Optional cache directory
        num_examples_per_feature: Number of contrastive examples to test per feature
        
    Returns:
        Dictionary with evaluation results
    """
    logger.info(f"Evaluating feature capabilities for {base_model} vs {target_model}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract features
    base_features = interpreted_features.get("base_model_specific_features", [])
    target_features = interpreted_features.get("target_model_specific_features", [])
    
    # Initialize results
    base_feature_evaluations = []
    target_feature_evaluations = []
    
    # Process base model features first
    if base_features:
        logger.info(f"Loading base model: {base_model}")
        try:
            # Load base model
            base_tokenizer = AutoTokenizer.from_pretrained(base_model, cache_dir=cache_dir)
            base_model_obj = AutoModelForCausalLM.from_pretrained(
                base_model, 
                cache_dir=cache_dir, 
                device_map=device,
                torch_dtype=torch.float16  # Use fp16 to reduce memory usage
            )
            
            # Process examples using base model only
            _process_base_model_features(
                base_model_obj, 
                base_tokenizer, 
                base_features, 
                base_feature_evaluations, 
                num_examples_per_feature
            )
            
            # Free GPU memory
            del base_model_obj
            torch.cuda.empty_cache()
            logger.info("Released base model from memory")
        
        except Exception as e:
            logger.warning(f"Error processing base model: {str(e)}")
    
    # Now process target model features
    logger.info(f"Loading target model: {target_model}")
    try:
        # Load target model
        target_tokenizer = AutoTokenizer.from_pretrained(target_model, cache_dir=cache_dir)
        target_model_obj = AutoModelForCausalLM.from_pretrained(
            target_model, 
            cache_dir=cache_dir, 
            device_map=device,
            torch_dtype=torch.float16  # Use fp16 to reduce memory usage
        )
        
        # Process target features first if they exist
        if target_features:
            _process_target_model_features(
                target_model_obj, 
                target_tokenizer, 
                target_features, 
                target_feature_evaluations, 
                num_examples_per_feature
            )
        
        # Then process target responses for base features
        if base_feature_evaluations:
            base_target_evaluations = _process_target_responses_for_base_features(
                target_model_obj, 
                target_tokenizer, 
                base_feature_evaluations
            )
            base_feature_evaluations = base_target_evaluations
        
        # Free GPU memory
        del target_model_obj
        torch.cuda.empty_cache()
        logger.info("Released target model from memory")
    
    except Exception as e:
        logger.warning(f"Error processing target model: {str(e)}")
    
    # Calculate capability scores for base features vs target model
    for feature in base_feature_evaluations:
        _calculate_feature_capability_scores(feature)
    
    # Calculate capability scores for target features vs base model (if available)
    if target_feature_evaluations:
        # We need to get base model responses for target features
        try:
            # Load base model again for target feature comparison
            base_tokenizer = AutoTokenizer.from_pretrained(base_model, cache_dir=cache_dir)
            base_model_obj = AutoModelForCausalLM.from_pretrained(
                base_model, 
                cache_dir=cache_dir, 
                device_map=device,
                torch_dtype=torch.float16
            )
            
            # Process base responses for target features
            target_base_evaluations = _process_base_responses_for_target_features(
                base_model_obj, 
                base_tokenizer, 
                target_feature_evaluations
            )
            target_feature_evaluations = target_base_evaluations
            
            # Calculate capability scores for target features
            for feature in target_feature_evaluations:
                _calculate_feature_capability_scores(feature, is_target_feature=True)
            
            # Free GPU memory
            del base_model_obj
            torch.cuda.empty_cache()
            
        except Exception as e:
            logger.warning(f"Error processing base responses for target features: {str(e)}")
    
    # Combine results
    results = {
        "base_feature_evaluations": base_feature_evaluations,
        "target_feature_evaluations": target_feature_evaluations
    }
    
    # Save results to file
    output_path = os.path.join(output_dir, "capability_evaluation.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Capability evaluation results saved to {output_path}")
    
    return results

def _calculate_feature_capability_scores(feature, is_target_feature=False):
    """
    Calculate capability scores for a feature based on example responses.
    
    Args:
        feature: Feature data with examples
        is_target_feature: Whether this is a target model feature (vs base model feature)
    """
    # Skip if already processed
    if feature.get("capability_scores_calculated", False):
        return
    
    base_key = "target" if is_target_feature else "base"
    comparison_key = "base" if is_target_feature else "target"
    
    for example in feature["examples"]:
        # Use special metric for human experience features
        if "human_experience" in feature["feature_name"].lower():
            # Calculate human experience scores
            base_positive_score = calculate_human_experience_score(example[f"{base_key}_positive_response"])
            base_negative_score = calculate_human_experience_score(example[f"{base_key}_negative_response"])
            comparison_positive_score = calculate_human_experience_score(example[f"{comparison_key}_positive_response"])
            comparison_negative_score = calculate_human_experience_score(example[f"{comparison_key}_negative_response"])
            
            # Compare positive responses to negative ones (differential)
            base_quality_ratio = base_positive_score - base_negative_score
            comparison_quality_ratio = comparison_positive_score - comparison_negative_score
            
            # Store human experience specific metrics
            example[f"{base_key}_positive_humanness"] = base_positive_score
            example[f"{base_key}_negative_humanness"] = base_negative_score
            example[f"{comparison_key}_positive_humanness"] = comparison_positive_score
            example[f"{comparison_key}_negative_humanness"] = comparison_negative_score
        else:
            # Use standard length-based comparison for other features
            base_quality_ratio = len(example[f"{base_key}_positive_response"]) / max(1, len(example[f"{base_key}_negative_response"]))
            comparison_quality_ratio = len(example[f"{comparison_key}_positive_response"]) / max(1, len(example[f"{comparison_key}_negative_response"]))
        
        # Calculate difference (positive means base is better, negative means comparison is better)
        difference = base_quality_ratio - comparison_quality_ratio
        
        # Add metrics to example result
        example[f"{base_key}_quality_ratio"] = base_quality_ratio
        example[f"{comparison_key}_quality_ratio"] = comparison_quality_ratio
        example["difference"] = difference
        
        # For target features, the difference interpretation is reversed
        if is_target_feature:
            # For target features, negative difference means target feature is stronger
            example["supports_feature"] = difference < -0.2
        else:
            # For base features, positive difference means base feature is stronger
            example["supports_feature"] = difference > 0.2
    
    # Mark as processed
    feature["capability_scores_calculated"] = True
    
    # Determine overall result for this feature
    supporting_examples = [ex for ex in feature["examples"] if ex.get("supports_feature", False)]
    feature["percent_supported"] = len(supporting_examples) / max(1, len(feature["examples"])) * 100
    feature["is_validated"] = feature["percent_supported"] >= 50

def _process_model_features(
    model, 
    tokenizer, 
    features, 
    results,
    num_examples_per_feature,
    is_base_model=True,
    base_feature_evaluations=None
):
    """
    Generic function to process model features - works for both base and target models.
    
    Args:
        model: The model to use
        tokenizer: The tokenizer to use
        features: List of features to process
        results: Results dictionary to update
        num_examples_per_feature: Number of examples to process per feature
        is_base_model: Whether this is the base model (True) or target model (False)
        base_feature_evaluations: Base feature evaluations (only needed for target model)
    """
    from ..capability.examples import generate_contrastive_examples
    
    model_type = "base" if is_base_model else "target"
    logger.info(f"Processing {model_type} model features")
    
    if not is_base_model and not base_feature_evaluations:
        logger.warning("No base feature evaluations provided for target model processing")
        return
    
    for feature in features:
        feature_name = feature.get("name", "Unknown")
        feature_description = feature.get("description", "")
        
        logger.info(f"Generating contrastive examples for {model_type} feature: {feature_name}")
        contrastive_examples = generate_contrastive_examples(feature_name, feature_description, num_examples_per_feature)
        
        # For target model processing with existing evaluations
        if not is_base_model and base_feature_evaluations:
            # Look for the feature in base evaluations
            matching_features = [f for f in base_feature_evaluations if f["feature_name"] == feature_name]
            if matching_features:
                # Use examples from the base model evaluation
                logger.info(f"Using examples from base model evaluation for {feature_name}")
                _process_feature_for_target_model(model, tokenizer, matching_features[0], results)
                continue
        
        # Setup the feature results object
        feature_results = {
            "feature_name": feature_name,
            "feature_description": feature_description,
            "examples": [],
            f"{model_type}_examples_processed": True,
            f"{'target' if is_base_model else 'base'}_examples_processed": False
        }
        
        # Process examples
        for example in contrastive_examples:
            positive_prompt = example["positive"]
            negative_prompt = example["negative"]
            description = example["description"]
            
            # Generate responses
            positive_response = generate_response(model, tokenizer, positive_prompt)
            negative_response = generate_response(model, tokenizer, negative_prompt)
            
            # Store result with the appropriate model prefix
            example_result = {
                "description": description,
                "positive_prompt": positive_prompt,
                "negative_prompt": negative_prompt,
                f"{model_type}_positive_response": positive_response,
                f"{model_type}_negative_response": negative_response
            }
            
            feature_results["examples"].append(example_result)
        
        results.append(feature_results)

def _process_feature_for_target_model(model, tokenizer, base_feature, results):
    """
    Process a feature for the target model when base feature evaluations exist.
    
    Args:
        model: The target model
        tokenizer: The tokenizer
        base_feature: The base feature evaluation
        results: Results to update
    """
    # Create a new feature result based on the base feature
    feature_results = {
        "feature_name": base_feature["feature_name"],
        "feature_description": base_feature["feature_description"],
        "examples": [],
        "base_examples_processed": True,
        "target_examples_processed": True
    }
    
    # Process each example from the base feature
    for example in base_feature["examples"]:
        positive_prompt = example["positive_prompt"]
        negative_prompt = example["negative_prompt"]
        
        # Generate target model responses
        target_positive = generate_response(model, tokenizer, positive_prompt)
        target_negative = generate_response(model, tokenizer, negative_prompt)
        
        # Create a new example with both base and target responses
        new_example = {
            "description": example["description"],
            "positive_prompt": positive_prompt,
            "negative_prompt": negative_prompt,
            "base_positive_response": example["base_positive_response"],
            "base_negative_response": example["base_negative_response"],
            "target_positive_response": target_positive,
            "target_negative_response": target_negative
        }
        
        feature_results["examples"].append(new_example)
    
    results.append(feature_results)

def _process_base_model_features(
    model, 
    tokenizer, 
    features, 
    results, 
    num_examples_per_feature
):
    """Process base model features using the provided model."""
    _process_model_features(
        model=model,
        tokenizer=tokenizer,
        features=features,
        results=results,
        num_examples_per_feature=num_examples_per_feature,
        is_base_model=True
    )

def _process_target_model_features(
    model, 
    tokenizer, 
    features, 
    results, 
    num_examples_per_feature
):
    """Process target model features using the provided model."""
    _process_model_features(
        model=model,
        tokenizer=tokenizer,
        features=features,
        results=results,
        num_examples_per_feature=num_examples_per_feature,
        is_base_model=False
    )

def _process_target_responses_for_base_features(
    model, 
    tokenizer, 
    base_feature_evaluations
):
    """Process target responses for base features."""
    logger.info("Processing target responses for base features")
    results = []
    
    for base_feature in base_feature_evaluations:
        _process_feature_for_target_model(model, tokenizer, base_feature, results)
    
    return results

def _process_base_responses_for_target_features(
    model, 
    tokenizer, 
    target_feature_evaluations
):
    """
    Process base model responses for target features.
    
    Args:
        model: The base model
        tokenizer: The tokenizer
        target_feature_evaluations: Target feature evaluations
        
    Returns:
        Updated target feature evaluations with base model responses
    """
    logger.info("Processing base responses for target features")
    results = []
    
    for target_feature in target_feature_evaluations:
        # Create a new feature result based on the target feature
        feature_results = {
            "feature_name": target_feature["feature_name"],
            "feature_description": target_feature["feature_description"],
            "examples": [],
            "base_examples_processed": True,
            "target_examples_processed": True
        }
        
        # Process each example from the target feature
        for example in target_feature["examples"]:
            positive_prompt = example["positive_prompt"]
            negative_prompt = example["negative_prompt"]
            
            # Generate base model responses
            base_positive = generate_response(model, tokenizer, positive_prompt)
            base_negative = generate_response(model, tokenizer, negative_prompt)
            
            # Create a new example with both base and target responses
            new_example = {
                "description": example["description"],
                "positive_prompt": positive_prompt,
                "negative_prompt": negative_prompt,
                "base_positive_response": base_positive,
                "base_negative_response": base_negative,
                "target_positive_response": example["target_positive_response"],
                "target_negative_response": example["target_negative_response"]
            }
            
            feature_results["examples"].append(new_example)
        
        results.append(feature_results)
    
    return results 