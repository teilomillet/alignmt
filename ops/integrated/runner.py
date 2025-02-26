"""
Runner module for integrated crosscoder + feature interpretation pipeline.

This module provides the main runner function that orchestrates
running crosscoder analysis and then feature interpretation in sequence.
"""

import os
import pickle
import json
import logging
import datetime
from typing import Dict, Any

import torch

from ..crosscoder.crosscode import analyze_model_changes
from ..feature_interpreter.pipeline.runner import run_feature_interpretation_pipeline
from .config import IntegratedPipelineConfig

logger = logging.getLogger(__name__)

def run_crosscoder_analysis(
    base_model: str,
    target_model: str,
    output_dir: str,
    param_types: list,
    cache_dir: str = None,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
    save_crosscoded_models: bool = False
) -> Dict:
    """
    Run crosscoder analysis between two models.
    
    Args:
        base_model: Base model name or path
        target_model: Target model name or path
        output_dir: Directory to save outputs
        param_types: Parameter types to analyze (e.g., "gate_proj.weight")
        cache_dir: Cache directory for models
        device: Device to use
        dtype: Data type to use
        save_crosscoded_models: Whether to save crosscoded model weights
        
    Returns:
        Dictionary with crosscoder analysis results
    """
    logger.info("Running crosscoder analysis:")
    logger.info(f"  Base model: {base_model}")
    logger.info(f"  Target model: {target_model}")
    logger.info(f"  Output directory: {output_dir}")
    
    # Generate timestamp for output files
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Dictionary to store filtered results
    filtered_results = {}
    
    # Run crosscoder analysis on all layers
    full_results = analyze_model_changes(
        base_model,
        target_model,
        cache_dir=cache_dir,
        device=device,
        dtype=dtype
    )
    
    # Filter results based on parameter types
    for layer_name, layer_data in full_results.items():
        for param_type in param_types:
            for param_name in layer_data.get('similarities', {}):
                if param_type in param_name:
                    key = param_name if param_name not in filtered_results else f"{param_name}_{layer_name}"
                    filtered_results[key] = {
                        'similarities': {param_name: layer_data['similarities'][param_name]},
                        'differences': {param_name: layer_data['differences'].get(param_name, 0.0)}
                    }
                    
                    # Only save crosscoded parameters if requested (to save memory)
                    if save_crosscoded_models and 'crosscoded' in layer_data:
                        filtered_results[key]['crosscoded'] = {
                            param_name: layer_data['crosscoded'].get(param_name, None)
                        }
    
    # Save full results (with crosscoded parameters)
    full_output_path = os.path.join(output_dir, f"crosscoder_analysis_{timestamp}_full.pkl")
    with open(full_output_path, "wb") as f:
        pickle.dump(full_results, f)
    
    # Save filtered results as JSON for easier analysis
    json_output_path = os.path.join(output_dir, f"crosscoder_analysis_{timestamp}_summary.json")
    
    # Create serializable version without tensor data
    serializable_results = {}
    for key, value in filtered_results.items():
        serializable_results[key] = {
            'similarities': value['similarities'],
            'differences': value['differences']
        }
    
    with open(json_output_path, "w") as f:
        json.dump(serializable_results, f, indent=2)
    
    logger.info("Crosscoder analysis complete.")
    logger.info(f"  Full results saved to: {full_output_path}")
    logger.info(f"  Summary saved to: {json_output_path}")
    
    return full_results

def run_integrated_pipeline(config: IntegratedPipelineConfig) -> Dict[str, Any]:
    """
    Run integrated pipeline combining crosscoder analysis and feature interpretation.
    
    Args:
        config: Pipeline configuration
        
    Returns:
        Dictionary with results
    """
    logger.info("Running integrated crosscoder+feature interpretation pipeline:")
    logger.info(f"  Base model: {config.base_model}")
    logger.info(f"  Target model: {config.target_model}")
    logger.info(f"  Output directory: {config.output_dir}")
    
    # Step 1: Run crosscoder analysis if not skipped
    crosscoder_results = None
    if not config.skip_crosscoder:
        crosscoder_results = run_crosscoder_analysis(
            base_model=config.base_model,
            target_model=config.target_model,
            output_dir=config.crosscoder_output_dir,
            param_types=config.crosscoder_param_types,
            cache_dir=config.cache_dir,
            device=config.device,
            dtype=torch.float32 if config.quantization == "fp32" else torch.float16,
            save_crosscoded_models=config.crosscoder_save_crosscoded_models
        )
    else:
        logger.info("Skipping crosscoder analysis as requested")
    
    # Step 2: Run feature interpretation pipeline with crosscoder results
    logger.info("Running feature interpretation pipeline")
    
    # Create a copy of the config to add crosscoder_results
    feature_config = config
    
    # Pass crosscoder results to the feature interpretation pipeline
    # by adding them to the config as an attribute
    if crosscoder_results:
        logger.info("Passing crosscoder results to feature interpretation pipeline")
        setattr(feature_config, 'crosscoder_results', crosscoder_results)
    
    feature_results = run_feature_interpretation_pipeline(feature_config)
    
    # Combine results
    results = {
        "base_model": config.base_model,
        "target_model": config.target_model,
        "output_dir": config.output_dir,
        "feature_interpretation": feature_results
    }
    
    if crosscoder_results:
        results["crosscoder"] = {
            "output_dir": config.crosscoder_output_dir,
            "analysis": crosscoder_results
        }
    
    return results 