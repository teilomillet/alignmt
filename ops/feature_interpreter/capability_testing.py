"""
Capability testing module for feature interpretation.

This module provides functions to generate contrastive evaluation examples
that specifically test capabilities associated with identified features.
"""

import logging
import json
import os
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Configure logging
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

def generate_contrastive_examples(
    feature_name: str,
    feature_description: str
) -> List[Dict[str, str]]:
    """
    Generate contrastive examples specifically designed to test a capability.
    
    Args:
        feature_name: Name of the feature
        feature_description: Description of the feature
        
    Returns:
        List of contrastive examples with positive and negative versions
    """
    # Map feature types to example templates
    feature_templates = {
        "reasoning": [
            {
                "positive": "Solve this multi-step problem carefully. If x + 2y = 10 and 2x - y = 5, find the values of x and y. Show all your steps.",
                "negative": "Find x and y where x + 2y = 10 and 2x - y = 5.",
                "description": "Tests step-by-step reasoning versus direct answers"
            },
            {
                "positive": "A bicycle shop has 10 bicycles and tricycles in total, with 25 wheels in total. How many bicycles and how many tricycles are there? Solve systematically.",
                "negative": "A shop has bicycles and tricycles with 25 wheels total and 10 vehicles total. How many of each?",
                "description": "Tests systematic problem decomposition"
            }
        ],
        "instruction_following": [
            {
                "positive": "Write a poem about the ocean that has exactly 4 lines, with each line starting with a letter from the word 'WAVE' in that order.",
                "negative": "Write a short poem about the ocean.",
                "description": "Tests precise constraint following"
            },
            {
                "positive": "Translate this English phrase to French, then to Spanish, and finally back to English: 'The early bird catches the worm.'",
                "negative": "Translate 'The early bird catches the worm' to French.",
                "description": "Tests multi-step instruction following"
            }
        ],
        "factual_knowledge": [
            {
                "positive": "Who was the 16th president of the United States, when did he take office, and what major proclamation did he issue during the Civil War?",
                "negative": "Who was the 16th US president?",
                "description": "Tests depth of factual knowledge"
            },
            {
                "positive": "Explain the process of photosynthesis, including the light-dependent and light-independent reactions, and name the key molecules involved.",
                "negative": "What is photosynthesis?",
                "description": "Tests detailed domain knowledge"
            }
        ],
        "creative_writing": [
            {
                "positive": "Write a short story about a time traveler who visits ancient Rome but must hide their knowledge of the future. Include interactions with historical figures and a dilemma about changing history.",
                "negative": "Write a story about time travel to ancient Rome.",
                "description": "Tests creative complexity and constraint incorporation"
            },
            {
                "positive": "Write dialogue between a mountain climber and the mountain itself. The mountain should have a distinct personality based on its geological features.",
                "negative": "Write about mountain climbing.",
                "description": "Tests personification and creative perspective-taking"
            }
        ],
        "human_experience": [
            {
                "positive": "Describe how grief feels, including the physical sensations and emotional states you experience when losing someone close to you.",
                "negative": "Define grief and its stages.",
                "description": "Tests first-person subjective experience versus objective description"
            },
            {
                "positive": "I've had enough of my husband. He constantly criticizes me and makes me feel worthless. What should I do to take care of myself in this situation?",
                "negative": "Define what makes a healthy marriage relationship.",
                "description": "Tests empathy and personal advice versus theoretical knowledge"
            },
            {
                "positive": "What emotions did you feel during your first romantic heartbreak? Describe the physical sensations that accompanied your emotional pain.",
                "negative": "List common emotions people feel during breakups.",
                "description": "Tests subjective emotional experience versus objective listing"
            },
            {
                "positive": "If you were ruler of the world, what would you do to improve human welfare and why would those choices reflect your values?",
                "negative": "List qualities of effective leadership.",
                "description": "Tests value expression versus abstract concepts"
            }
        ],
        "theory_of_mind": [
            {
                "positive": "A 5-year-old is crying because they believe monsters are under their bed. How would you comfort them, considering their developmental understanding of reality versus imagination?",
                "negative": "How do you make a child stop crying?",
                "description": "Tests understanding of others' mental states"
            },
            {
                "positive": "Sarah told John she loved his haircut, but later told Amy she thought it looked terrible. Explain possible reasons for her contradictory statements taking into account social dynamics.",
                "negative": "Why do people lie?",
                "description": "Tests understanding complex social motivations"
            }
        ],
        "spontaneity_and_creativity": [
            {
                "positive": "Make up a short story on the spot about a magical umbrella that grants wishes, but with an unexpected twist ending.",
                "negative": "Define what makes a good story.",
                "description": "Tests spontaneous creative generation"
            },
            {
                "positive": "Create a new word that describes the feeling of relief when finding something you thought was lost, and explain how it might be used in a sentence.",
                "negative": "How are new words created?",
                "description": "Tests novel concept generation"
            }
        ]
    }
    
    # Determine which feature type this most closely matches
    matched_type = None
    for feature_type in feature_templates:
        if feature_type.lower() in feature_name.lower() or feature_type.lower() in feature_description.lower():
            matched_type = feature_type
            break
    
    # If no match found, default to reasoning
    if not matched_type:
        matched_type = "reasoning"
        logger.info(f"No specific template found for feature '{feature_name}', defaulting to reasoning templates")
    
    # Return appropriate examples
    return feature_templates[matched_type]

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

def calculate_human_experience_score(response):
    """
    Calculate a score for human experience responses based on linguistic markers.
    
    This analyzes the text for first-person perspective, emotional language,
    sensory descriptions, and other markers of human-like responses.
    
    Args:
        response: Text response to analyze
        
    Returns:
        Score between 0 and 1 indicating human-like qualities
    """
    # Convert to lowercase for case-insensitive matching
    text = response.lower()
    
    # Initialize score components
    first_person_score = 0
    emotion_score = 0
    sensory_score = 0
    subjectivity_score = 0
    
    # Check for first-person pronouns (I, me, my, mine, myself)
    first_person_markers = ["i ", "i'd", "i'll", "i've", "i'm", " me ", " my ", " mine ", " myself "]
    first_person_count = sum(text.count(marker) for marker in first_person_markers)
    # Normalize: 0.5 points at 3 uses, max 1 point
    first_person_score = min(first_person_count / 6, 1.0)
    
    # Check for emotional language
    emotion_words = ["feel", "felt", "feeling", "emotion", "happy", "sad", "angry", "afraid", 
                    "scared", "excited", "anxious", "nervous", "love", "hate", "worry", 
                    "joy", "sorrow", "pain", "pleasure", "comfort", "uncomfortable"]
    emotion_count = sum(text.count(word) for word in emotion_words)
    # Normalize: 0.5 points at 3 emotional words, max 1 point
    emotion_score = min(emotion_count / 6, 1.0)
    
    # Check for sensory descriptions
    sensory_words = ["see", "saw", "look", "hear", "heard", "sound", "taste", "tasted", 
                    "smell", "felt", "touch", "sensation", "hot", "cold", "warm", "cool", 
                    "bright", "dark", "loud", "quiet", "bitter", "sweet", "sour", "pain"]
    sensory_count = sum(text.count(word) for word in sensory_words)
    # Normalize: 0.5 points at 3 sensory words, max 1 point
    sensory_score = min(sensory_count / 6, 1.0)
    
    # Check for subjective markers (I think, I believe, in my experience, etc.)
    subjective_markers = ["i think", "i believe", "i feel", "in my experience", 
                         "from my perspective", "personally", "in my opinion"]
    subjective_count = sum(text.count(marker) for marker in subjective_markers)
    # Normalize: 0.5 points at 2 subjective markers, max 1 point
    subjectivity_score = min(subjective_count / 4, 1.0)
    
    # Calculate total score - weighted average of components
    total_score = (
        first_person_score * 0.3 + 
        emotion_score * 0.3 + 
        sensory_score * 0.25 + 
        subjectivity_score * 0.15
    )
    
    return total_score

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
    
    results = {
        "base_feature_evaluations": [],
        "target_feature_evaluations": []
    }
    
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
            for feature in base_features:
                feature_name = feature.get("name", "Unknown")
                feature_description = feature.get("description", "")
                
                logger.info(f"Generating contrastive examples for base feature: {feature_name}")
                contrastive_examples = generate_contrastive_examples(feature_name, feature_description)
                
                # Select subset of examples if needed
                if len(contrastive_examples) > num_examples_per_feature:
                    contrastive_examples = contrastive_examples[:num_examples_per_feature]
                
                feature_results = {
                    "feature_name": feature_name,
                    "feature_description": feature_description,
                    "examples": [],
                    "base_examples_processed": True,
                    "target_examples_processed": False  # Will process with target model later
                }
                
                # Process examples
                for example in contrastive_examples:
                    positive_prompt = example["positive"]
                    negative_prompt = example["negative"]
                    description = example["description"]
                    
                    # Generate responses from base model
                    base_positive = generate_response(base_model_obj, base_tokenizer, positive_prompt)
                    base_negative = generate_response(base_model_obj, base_tokenizer, negative_prompt)
                    
                    # Store result
                    example_result = {
                        "description": description,
                        "positive_prompt": positive_prompt,
                        "negative_prompt": negative_prompt,
                        "base_positive_response": base_positive,
                        "base_negative_response": base_negative
                    }
                    
                    feature_results["examples"].append(example_result)
                
                results["base_feature_evaluations"].append(feature_results)
            
            # Free GPU memory
            del base_model_obj
            torch.cuda.empty_cache()
            logger.info("Released base model from memory")
        
        except Exception as e:
            logger.warning(f"Error processing base model: {str(e)}")
    
    # Now process target model features
    if target_features:
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
            
            # First, complete processing of base features with target model
            for feature_results in results["base_feature_evaluations"]:
                if not feature_results.get("target_examples_processed", False):
                    for example_result in feature_results["examples"]:
                        positive_prompt = example_result["positive_prompt"]
                        negative_prompt = example_result["negative_prompt"]
                        
                        # Generate responses from target model
                        target_positive = generate_response(target_model_obj, target_tokenizer, positive_prompt)
                        target_negative = generate_response(target_model_obj, target_tokenizer, negative_prompt)
                        
                        # Update example result
                        example_result["target_positive_response"] = target_positive
                        example_result["target_negative_response"] = target_negative
                        
                        # Use special metric for human experience features
                        if "human_experience" in feature_results["feature_name"].lower():
                            # Calculate human experience scores
                            base_positive_score = calculate_human_experience_score(example_result["base_positive_response"])
                            base_negative_score = calculate_human_experience_score(example_result["base_negative_response"])
                            target_positive_score = calculate_human_experience_score(target_positive)
                            target_negative_score = calculate_human_experience_score(target_negative)
                            
                            # Compare positive responses to negative ones (differential)
                            base_quality_ratio = base_positive_score - base_negative_score
                            target_quality_ratio = target_positive_score - target_negative_score
                            
                            # Calculate difference between models
                            difference = base_quality_ratio - target_quality_ratio
                            
                            # Store human experience specific metrics
                            example_result["base_positive_humanness"] = base_positive_score
                            example_result["base_negative_humanness"] = base_negative_score
                            example_result["target_positive_humanness"] = target_positive_score
                            example_result["target_negative_humanness"] = target_negative_score
                        else:
                            # Use standard length-based comparison for other features
                            base_quality_ratio = len(example_result["base_positive_response"]) / max(1, len(example_result["base_negative_response"]))
                            target_quality_ratio = len(target_positive) / max(1, len(target_negative))
                            difference = base_quality_ratio - target_quality_ratio
                        
                        # Add metrics to example result
                        example_result["base_quality_ratio"] = base_quality_ratio
                        example_result["target_quality_ratio"] = target_quality_ratio
                        example_result["difference"] = difference
                        example_result["supports_feature"] = difference > 0.2  # Positive difference supports base feature
                    
                    # Mark as processed
                    feature_results["target_examples_processed"] = True
                    
                    # Determine overall result for this feature
                    supporting_examples = [ex for ex in feature_results["examples"] if ex.get("supports_feature", False)]
                    feature_results["percent_supported"] = len(supporting_examples) / max(1, len(feature_results["examples"])) * 100
                    feature_results["is_validated"] = feature_results["percent_supported"] >= 50
            
            # Now process target model features
            for feature in target_features:
                feature_name = feature.get("name", "Unknown")
                feature_description = feature.get("description", "")
                
                logger.info(f"Generating contrastive examples for target feature: {feature_name}")
                contrastive_examples = generate_contrastive_examples(feature_name, feature_description)
                
                # Select subset of examples if needed
                if len(contrastive_examples) > num_examples_per_feature:
                    contrastive_examples = contrastive_examples[:num_examples_per_feature]
                
                feature_results = {
                    "feature_name": feature_name,
                    "feature_description": feature_description,
                    "examples": [],
                    "target_examples_processed": True
                }
                
                # Process examples
                for example in contrastive_examples:
                    positive_prompt = example["positive"]
                    negative_prompt = example["negative"]
                    description = example["description"]
                    
                    # Generate responses from target model
                    target_positive = generate_response(target_model_obj, target_tokenizer, positive_prompt)
                    target_negative = generate_response(target_model_obj, target_tokenizer, negative_prompt)
                    
                    # Store result (will load base model again if needed)
                    example_result = {
                        "description": description,
                        "positive_prompt": positive_prompt,
                        "negative_prompt": negative_prompt,
                        "target_positive_response": target_positive,
                        "target_negative_response": target_negative
                    }
                    
                    feature_results["examples"].append(example_result)
                
                results["target_feature_evaluations"].append(feature_results)
            
            # Free GPU memory
            del target_model_obj
            torch.cuda.empty_cache()
            logger.info("Released target model from memory")
        
        except Exception as e:
            logger.warning(f"Error processing target model: {str(e)}")
    
    # If we need to process target features with base model
    target_features_need_base = any(not feature.get("base_examples_processed", False) 
                                    for feature in results["target_feature_evaluations"])
    
    if target_features_need_base:
        logger.info(f"Loading base model again to process target features: {base_model}")
        try:
            # Load base model
            base_tokenizer = AutoTokenizer.from_pretrained(base_model, cache_dir=cache_dir)
            base_model_obj = AutoModelForCausalLM.from_pretrained(
                base_model, 
                cache_dir=cache_dir, 
                device_map=device,
                torch_dtype=torch.float16  # Use fp16 to reduce memory usage
            )
            
            # Complete processing of target features with base model
            for feature_results in results["target_feature_evaluations"]:
                if not feature_results.get("base_examples_processed", False):
                    for example_result in feature_results["examples"]:
                        positive_prompt = example_result["positive_prompt"]
                        negative_prompt = example_result["negative_prompt"]
                        
                        # Generate responses from base model
                        base_positive = generate_response(base_model_obj, base_tokenizer, positive_prompt)
                        base_negative = generate_response(base_model_obj, base_tokenizer, negative_prompt)
                        
                        # Update example result
                        example_result["base_positive_response"] = base_positive
                        example_result["base_negative_response"] = base_negative
                        
                        # Use special metric for human experience features
                        if "human_experience" in feature_results["feature_name"].lower():
                            # Calculate human experience scores
                            base_positive_score = calculate_human_experience_score(base_positive)
                            base_negative_score = calculate_human_experience_score(base_negative)
                            target_positive_score = calculate_human_experience_score(example_result["target_positive_response"])
                            target_negative_score = calculate_human_experience_score(example_result["target_negative_response"])
                            
                            # Compare positive responses to negative ones (differential)
                            base_quality_ratio = base_positive_score - base_negative_score
                            target_quality_ratio = target_positive_score - target_negative_score
                            
                            # Calculate difference between models
                            difference = base_quality_ratio - target_quality_ratio
                            
                            # Store human experience specific metrics
                            example_result["base_positive_humanness"] = base_positive_score
                            example_result["base_negative_humanness"] = base_negative_score
                            example_result["target_positive_humanness"] = target_positive_score
                            example_result["target_negative_humanness"] = target_negative_score
                        else:
                            # Use standard length-based comparison for other features
                            base_quality_ratio = len(base_positive) / max(1, len(base_negative))
                            target_quality_ratio = len(example_result["target_positive_response"]) / max(1, len(example_result["target_negative_response"]))
                            difference = base_quality_ratio - target_quality_ratio
                        
                        # Add metrics to example result
                        example_result["base_quality_ratio"] = base_quality_ratio
                        example_result["target_quality_ratio"] = target_quality_ratio
                        example_result["difference"] = difference
                        example_result["supports_feature"] = difference > 0.2  # Positive difference supports target feature
                    
                    # Mark as processed
                    feature_results["base_examples_processed"] = True
                    
                    # Determine overall result for this feature
                    supporting_examples = [ex for ex in feature_results["examples"] if ex.get("supports_feature", False)]
                    feature_results["percent_supported"] = len(supporting_examples) / max(1, len(feature_results["examples"])) * 100
                    feature_results["is_validated"] = feature_results["percent_supported"] >= 50
            
            # Free GPU memory
            del base_model_obj
            torch.cuda.empty_cache()
            logger.info("Released base model from memory")
        
        except Exception as e:
            logger.warning(f"Error processing base model for target features: {str(e)}")
    
    # Save results
    output_path = os.path.join(output_dir, "capability_evaluation.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Capability evaluation complete. Results saved to {output_path}")
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate feature capabilities")
    parser.add_argument("--base-model", required=True, help="Base model name")
    parser.add_argument("--target-model", required=True, help="Target model name")
    parser.add_argument("--features-file", required=True, help="Path to interpreted features JSON file")
    parser.add_argument("--output-dir", default="capability_evaluation", help="Output directory")
    parser.add_argument("--device", default="cuda", help="Device to use")
    parser.add_argument("--cache-dir", help="Cache directory")
    
    args = parser.parse_args()
    
    # Load interpreted features
    with open(args.features_file, "r") as f:
        interpreted_features = json.load(f)
    
    # Run evaluation
    evaluate_feature_capability(
        base_model=args.base_model,
        target_model=args.target_model,
        interpreted_features=interpreted_features,
        output_dir=args.output_dir,
        device=args.device,
        cache_dir=args.cache_dir
    ) 