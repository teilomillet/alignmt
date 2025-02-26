"""
Comprehensive Decoder Analysis Module.

This module provides the main functionality for generating comprehensive
decoder weight analysis by combining multiple analysis methods.
"""

import numpy as np
import logging
import json
from pathlib import Path
from typing import Dict, List, Optional

from .norms import extract_feature_decoder_norms, categorize_features_by_norm
from .activity import calculate_feature_alignment
from .clustering import cluster_features
from .responses import compare_feature_responses

# Configure logging
logger = logging.getLogger(__name__)

def generate_comprehensive_analysis(
    base_decoder: np.ndarray,
    target_decoder: np.ndarray,
    base_activations: Optional[np.ndarray] = None,
    target_activations: Optional[np.ndarray] = None,
    prompt_labels: Optional[List[str]] = None,
    feature_names: Optional[List[str]] = None,
    norm_ratio_threshold: float = 1.5,
    n_clusters: int = 5,
    output_dir: Optional[str] = None
) -> Dict:
    """
    Generate a comprehensive analysis of decoder weights and feature responses.
    
    Args:
        base_decoder: Base model decoder weights matrix
        target_decoder: Target model decoder weights matrix
        base_activations: Optional base model activation matrix
        target_activations: Optional target model activation matrix
        prompt_labels: Optional labels for prompts
        feature_names: Optional list of feature names
        norm_ratio_threshold: Threshold for norm ratio
        n_clusters: Number of clusters for feature clustering
        output_dir: Optional directory to save results
        
    Returns:
        Dictionary with comprehensive analysis results
    """
    logger.info("Generating comprehensive decoder weight analysis")
    
    # Create output directory if specified
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract and analyze feature decoder norms
    feature_data = extract_feature_decoder_norms(
        base_decoder=base_decoder,
        target_decoder=target_decoder,
        feature_names=feature_names
    )
    
    # Categorize features by norm
    categorized_features = categorize_features_by_norm(
        feature_data=feature_data,
        norm_ratio_threshold=norm_ratio_threshold,
        output_path=output_dir / 'categorized_features.json' if output_dir else None
    )
    
    # Calculate feature alignment
    alignment_results = calculate_feature_alignment(
        feature_data=feature_data,
        output_path=output_dir / 'feature_alignment.png' if output_dir else None
    )
    
    # Cluster features
    clustering_results = cluster_features(
        feature_data=feature_data,
        method="kmeans",
        n_clusters=n_clusters,
        output_dir=output_dir
    )
    
    # Analyze feature responses if activations are provided
    response_analysis = None
    if (base_activations is not None and target_activations is not None and 
        prompt_labels is not None):
        response_analysis = compare_feature_responses(
            base_activations=base_activations,
            target_activations=target_activations,
            prompt_labels=prompt_labels,
            feature_names=feature_names,
            output_dir=output_dir,
            top_n=10
        )
    
    # Combine all analysis results
    analysis_results = {
        "feature_data": feature_data,
        "categorized_features": categorized_features,
        "alignment_results": alignment_results,
        "clustering_results": clustering_results,
        "response_analysis": response_analysis
    }
    
    # Save comprehensive analysis results if output directory is provided
    if output_dir:
        try:
            # Save a summary JSON file with the analysis results
            summary_path = output_dir / 'comprehensive_analysis_summary.json'
            
            # Create a JSON-compatible summary (excluding large arrays)
            summary = {
                "categorized_features": categorized_features,
                "alignment_results": {
                    "average_alignment": alignment_results["average_alignment"]
                },
                "clustering_results": {
                    "num_clusters": len(clustering_results["feature_clusters"])
                }
            }
            
            # Add response analysis summary if available
            if response_analysis:
                summary["response_analysis"] = {
                    "num_top_features": len(response_analysis["top_different_features"]),
                    "num_prompts_analyzed": len(response_analysis["prompt_specific_features"])
                }
            
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
            
            # Try to save the full analysis results
            full_path = output_dir / 'comprehensive_analysis.json'
            try:
                with open(full_path, 'w') as f:
                    # Convert numpy arrays to lists for JSON serialization
                    json_compatible_results = json.dumps(
                        analysis_results,
                        default=lambda x: x.tolist() if hasattr(x, 'tolist') else x
                    )
                    f.write(json_compatible_results)
            except Exception as e:
                logger.warning(f"Failed to save full analysis results as JSON: {e}")
                
                # Try using pickle as fallback
                import pickle
                pickle_path = output_dir / 'comprehensive_analysis.pkl'
                with open(pickle_path, 'wb') as f:
                    pickle.dump(analysis_results, f)
                logger.info(f"Saved analysis results as pickle to {pickle_path}")
            
            logger.info(f"Saved comprehensive analysis summary to {summary_path}")
        except Exception as e:
            logger.error(f"Failed to save analysis results: {e}")
    
    return analysis_results 