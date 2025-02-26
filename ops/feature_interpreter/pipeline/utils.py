"""
Utility functions for the feature interpretation pipeline.

This module provides helper functions for logging, file operations,
and other utilities used by the pipeline.
"""

import os
import logging
import pickle
from typing import Dict, Any, List, Tuple

logger = logging.getLogger(__name__)

def setup_logging(log_file: str, level: int = logging.INFO) -> None:
    """
    Set up logging to file and console.
    
    Args:
        log_file: Path to log file
        level: Logging level (default: INFO)
    """
    if not os.path.exists(os.path.dirname(log_file)):
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
    # Configure formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Set up console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Set up file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Set level
    logger.setLevel(level)
    logger.info(f"Logging to {log_file}")

def extract_or_load_activations(
    base_model: str,
    target_model: str,
    output_dir: str,
    all_prompts: List[str],
    device: str = "cuda",
    cache_dir: str = None,
    quantization: str = "fp16",
    skip_activations: bool = False
) -> Dict[str, Any]:
    """
    Extract activations from models or load from cache.
    
    Args:
        base_model: Name of the base model
        target_model: Name of the target model
        output_dir: Directory to save outputs
        all_prompts: List of all prompts to process
        device: Device to use (cuda or cpu)
        cache_dir: Directory to cache models
        quantization: Quantization method
        skip_activations: Skip activation extraction
        
    Returns:
        Dictionary of activations
    """
    from ..activations import extract_activations_for_comparison
    
    activations_path = os.path.join(output_dir, "activations.pkl")
    
    if not skip_activations and not os.path.exists(activations_path):
        logger.info(f"Extracting activations for {len(all_prompts)} prompts")
        try:
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
                
            logger.info(f"Activations saved to {activations_path}")
            return activations
            
        except Exception as e:
            logger.error(f"Failed to extract activations: {str(e)}")
            raise
    else:
        logger.info(f"Loading activations from {activations_path}")
        with open(activations_path, "rb") as f:
            return pickle.load(f)

def flatten_prompts(prompt_categories: Dict[str, List[str]]) -> Tuple[List[str], Dict[str, str]]:
    """
    Flatten prompt categories into a list and create a mapping of prompts to categories.
    
    Args:
        prompt_categories: Dictionary mapping categories to lists of prompts
        
    Returns:
        Tuple of (all_prompts, prompt_to_category)
    """
    all_prompts = []
    prompt_to_category = {}
    
    for category, prompts in prompt_categories.items():
        all_prompts.extend(prompts)
        for prompt in prompts:
            prompt_to_category[prompt] = category
    
    return all_prompts, prompt_to_category 