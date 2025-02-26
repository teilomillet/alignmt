"""
Example script demonstrating decoder weight analysis functionality.

This script shows how to use the decoder_analysis module to analyze
the differences between base and target model features.
"""

import os
import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from typing import Dict, List, Any, Optional

# Add parent directory to path to allow importing modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import from the new modular package
from ops.feature_interpreter.decoder_analysis import (
    extract_feature_decoder_norms,
    identify_active_features,
    cluster_features,
    compare_feature_responses,
    generate_comprehensive_analysis
)

def run_decoder_analysis_example(
    base_decoder_path: str,
    target_decoder_path: str,
    base_activations_path: Optional[str] = None,
    target_activations_path: Optional[str] = None,
    prompt_labels_path: Optional[str] = None,
    output_dir: str = "decoder_analysis_results"
) -> Dict:
    """
    Run example decoder weight analysis on provided decoder weights.
    
    Args:
        base_decoder_path: Path to base model decoder weights (numpy format)
        target_decoder_path: Path to target model decoder weights (numpy format)
        base_activations_path: Optional path to base model activations
        target_activations_path: Optional path to target model activations
        prompt_labels_path: Optional path to prompt labels JSON
        output_dir: Directory to save results
        
    Returns:
        Dictionary with analysis results
    """
    print(f"Loading decoder weights from {base_decoder_path} and {target_decoder_path}")
    
    # Load decoder weights
    base_decoder = np.load(base_decoder_path)
    target_decoder = np.load(target_decoder_path)
    
    # Optional: Load activations and prompt labels if provided
    base_activations = None
    target_activations = None
    prompt_labels = None
    
    if (base_activations_path and target_activations_path and prompt_labels_path):
        print(f"Loading activations and prompt labels")
        base_activations = np.load(base_activations_path)
        target_activations = np.load(target_activations_path)
        
        with open(prompt_labels_path, 'r') as f:
            prompt_labels = json.load(f)
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate comprehensive analysis
    print(f"Generating comprehensive decoder weight analysis")
    analysis_results = generate_comprehensive_analysis(
        base_decoder=base_decoder,
        target_decoder=target_decoder,
        base_activations=base_activations,
        target_activations=target_activations,
        prompt_labels=prompt_labels,
        output_dir=output_dir
    )
    
    # Print summary of results
    categorized = analysis_results["categorized_features"]
    print("\nFeature Categorization Summary:")
    print(f"Base-specific features: {len(categorized['base_specific'])}")
    print(f"Target-specific features: {len(categorized['target_specific'])}")
    print(f"Shared features: {len(categorized['shared'])}")
    
    # Print top features with highest norm ratio differences
    print("\nTop Base-Specific Features (highest base/target norm ratio):")
    for i, feature in enumerate(sorted(categorized['base_specific'], 
                                     key=lambda x: 1/x['ratio'])[:5]):
        print(f"{i+1}. Feature {feature['id']}: base_norm={feature['base_norm']:.4f}, "
              f"target_norm={feature['target_norm']:.4f}, ratio={feature['ratio']:.4f}")
    
    print("\nTop Target-Specific Features (highest target/base norm ratio):")
    for i, feature in enumerate(sorted(categorized['target_specific'], 
                                     key=lambda x: x['ratio'])[:5]):
        print(f"{i+1}. Feature {feature['id']}: base_norm={feature['base_norm']:.4f}, "
              f"target_norm={feature['target_norm']:.4f}, ratio={feature['ratio']:.4f}")
    
    print(f"\nAnalysis results saved to {output_dir}")
    
    return analysis_results

def generate_synthetic_data(
    num_features: int = 1000,
    feature_dim: int = 768,
    num_prompts: int = 20,
    output_dir: str = "synthetic_data"
) -> Dict[str, str]:
    """
    Generate synthetic data for testing decoder analysis functionality.
    
    Args:
        num_features: Number of features
        feature_dim: Dimension of each feature
        num_prompts: Number of prompt examples
        output_dir: Directory to save synthetic data
        
    Returns:
        Dictionary with paths to generated files
    """
    print(f"Generating synthetic data with {num_features} features")
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate synthetic decoder weights
    # Base model has random normal features
    base_decoder = np.random.normal(0, 1, (num_features, feature_dim))
    
    # Target model has similar features but with some differences
    # 70% shared features, 15% base-specific, 15% target-specific
    target_decoder = base_decoder.copy()
    
    # Modify target decoder to create differences
    # Base-specific features: reduce norm in target model
    base_specific_indices = np.random.choice(
        num_features, int(0.15 * num_features), replace=False
    )
    target_decoder[base_specific_indices] *= 0.1
    
    # Target-specific features: increase norm in target model
    target_specific_indices = np.random.choice(
        np.setdiff1d(np.arange(num_features), base_specific_indices),
        int(0.15 * num_features), replace=False
    )
    target_decoder[target_specific_indices] *= 5.0
    
    # Add some noise to all features
    target_decoder += np.random.normal(0, 0.1, target_decoder.shape)
    
    # Generate synthetic activations
    base_activations = np.zeros((num_features, num_prompts))
    target_activations = np.zeros((num_features, num_prompts))
    
    # Create activation patterns
    for i in range(num_features):
        if i in base_specific_indices:
            # Base-specific features activate more in base model
            base_activations[i] = np.random.normal(0.5, 0.2, num_prompts)
            target_activations[i] = np.random.normal(0.1, 0.1, num_prompts)
        elif i in target_specific_indices:
            # Target-specific features activate more in target model
            base_activations[i] = np.random.normal(0.1, 0.1, num_prompts)
            target_activations[i] = np.random.normal(0.5, 0.2, num_prompts)
        else:
            # Shared features activate similarly in both models
            activation = np.random.normal(0.3, 0.2, num_prompts)
            base_activations[i] = activation + np.random.normal(0, 0.05, num_prompts)
            target_activations[i] = activation + np.random.normal(0, 0.05, num_prompts)
    
    # Generate prompt labels
    prompt_labels = [f"Prompt {i+1}" for i in range(num_prompts)]
    
    # Save synthetic data
    base_decoder_path = output_dir / "base_decoder.npy"
    target_decoder_path = output_dir / "target_decoder.npy"
    base_activations_path = output_dir / "base_activations.npy"
    target_activations_path = output_dir / "target_activations.npy"
    prompt_labels_path = output_dir / "prompt_labels.json"
    
    np.save(base_decoder_path, base_decoder)
    np.save(target_decoder_path, target_decoder)
    np.save(base_activations_path, base_activations)
    np.save(target_activations_path, target_activations)
    
    with open(prompt_labels_path, 'w') as f:
        json.dump(prompt_labels, f)
    
    print(f"Synthetic data saved to {output_dir}")
    
    return {
        "base_decoder_path": str(base_decoder_path),
        "target_decoder_path": str(target_decoder_path),
        "base_activations_path": str(base_activations_path),
        "target_activations_path": str(target_activations_path),
        "prompt_labels_path": str(prompt_labels_path)
    }

if __name__ == "__main__":
    # Check if synthetic data needs to be generated
    use_synthetic = True
    
    if use_synthetic:
        # Generate synthetic data
        data_paths = generate_synthetic_data(
            num_features=1000,
            feature_dim=768,
            num_prompts=20,
            output_dir="synthetic_data"
        )
        
        # Run decoder analysis on synthetic data
        analysis_results = run_decoder_analysis_example(
            base_decoder_path=data_paths["base_decoder_path"],
            target_decoder_path=data_paths["target_decoder_path"],
            base_activations_path=data_paths["base_activations_path"],
            target_activations_path=data_paths["target_activations_path"],
            prompt_labels_path=data_paths["prompt_labels_path"],
            output_dir="decoder_analysis_results"
        )
    else:
        # If you have real data, specify the paths here
        analysis_results = run_decoder_analysis_example(
            base_decoder_path="path/to/base_decoder.npy",
            target_decoder_path="path/to/target_decoder.npy",
            base_activations_path="path/to/base_activations.npy",
            target_activations_path="path/to/target_activations.npy",
            prompt_labels_path="path/to/prompt_labels.json",
            output_dir="decoder_analysis_results"
        ) 