"""
Layer analysis module for the feature interpretation pipeline.

This module provides functions for analyzing layer similarities
between base and target models.
"""

import re
import logging
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Dict, Any
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

def compute_layer_similarities(activations: Dict[str, Any]) -> Dict[str, float]:
    """
    Compute similarities between corresponding layers in the base and target models.
    
    Args:
        activations: Dictionary with activations
        
    Returns:
        Dictionary mapping layer names to similarity scores
    """
    logger.info("Computing layer similarities")
    layer_similarities = {}
    
    # Get a sample prompt to extract layer names
    sample_prompt = list(activations.keys())[0]
    sample_data = activations[sample_prompt]
    
    # Extract layer names - handle both numpy arrays and torch tensors
    layer_names = []
    for key in sample_data["base_activations"].keys():
        base_act = sample_data["base_activations"][key]
        if isinstance(base_act, (np.ndarray, torch.Tensor)):
            layer_names.append(key)
    
    logger.info(f"Found {len(layer_names)} layers to analyze")
    
    # Compute similarities for each layer
    for layer in tqdm(layer_names, desc="Computing layer similarities"):
        # Collect activations for this layer across all prompts
        base_activations = []
        target_activations = []
        
        for prompt, data in activations.items():
            if layer in data["base_activations"] and layer in data["target_activations"]:
                base_act = data["base_activations"][layer]
                target_act = data["target_activations"][layer]
                
                # Convert torch tensors to numpy if needed
                if isinstance(base_act, torch.Tensor):
                    # First convert to float32 to handle bfloat16
                    base_act = base_act.to(torch.float32)
                    base_act = base_act.cpu().numpy() if base_act.device.type != 'cpu' else base_act.numpy()
                if isinstance(target_act, torch.Tensor):
                    # First convert to float32 to handle bfloat16
                    target_act = target_act.to(torch.float32)
                    target_act = target_act.cpu().numpy() if target_act.device.type != 'cpu' else target_act.numpy()
                
                # Ensure activations have the same shape
                if base_act.shape == target_act.shape:
                    base_activations.append(base_act.flatten())
                    target_activations.append(target_act.flatten())
        
        if base_activations and target_activations:
            # Compute average cosine similarity using batch operations for efficiency
            base_tensor = torch.tensor(np.vstack(base_activations), dtype=torch.float32)
            target_tensor = torch.tensor(np.vstack(target_activations), dtype=torch.float32)
            
            # Normalize tensors
            base_norm = torch.norm(base_tensor, dim=1, keepdim=True)
            target_norm = torch.norm(target_tensor, dim=1, keepdim=True)
            
            # Avoid division by zero
            valid_indices = (base_norm.squeeze() > 0) & (target_norm.squeeze() > 0)
            
            if valid_indices.any():
                base_normalized = base_tensor[valid_indices] / base_norm[valid_indices]
                target_normalized = target_tensor[valid_indices] / target_norm[valid_indices]
                
                # Compute cosine similarity
                similarities = torch.sum(base_normalized * target_normalized, dim=1)
                layer_similarities[layer] = float(torch.mean(similarities).item())
            else:
                layer_similarities[layer] = 0.0
        else:
            layer_similarities[layer] = 0.0
    
    return layer_similarities

def create_layer_similarity_plot(
    layer_similarities: Dict[str, float],
    output_path: str
) -> None:
    """
    Create a plot showing similarities between corresponding layers.
    
    Args:
        layer_similarities: Dictionary mapping layer names to similarity scores
        output_path: Path to save the plot
    """
    logger.info(f"Creating layer similarity plot at {output_path}")
    
    # Extract layer names and similarities
    layers = []
    
    for layer, similarity in layer_similarities.items():
        # Extract layer number if possible
        match = re.search(r'(\d+)', layer)
        if match:
            layer_num = int(match.group(1))
            layers.append((layer_num, layer, similarity))
        else:
            # If no number found, use a large number to place at the end
            layers.append((999, layer, similarity))
    
    # Sort layers by number
    layers.sort()
    
    # Extract data for plotting
    layer_names = [layer[1] for layer in layers]
    similarities = [layer[2] for layer in layers]
    
    # Find the maximum similarity value to use as the y-axis maximum
    max_similarity = max(similarities) if similarities else 1.0
    # Add a small padding (5%) above the maximum value for visual clarity
    y_max = max_similarity * 1.05
    
    # Create the plot with modern styling
    plt.figure(figsize=(12, 6))
    bars = plt.bar(
        range(len(layer_names)), 
        similarities,
        color='#3498db',
        alpha=0.8,
        width=0.7
    )
    
    # Set y-axis limit to the maximum similarity value with padding
    plt.ylim(0, y_max)
    
    # Add grid for better readability
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xlabel('Layer', fontsize=12)
    plt.ylabel('Similarity Score', fontsize=12)
    plt.title('Layer-wise Similarity Between Base and Target Models', fontsize=14)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2.,
            height + (y_max * 0.01),  # Position text slightly above bar
            f'{height:.2f}',
            ha='center', 
            va='bottom',
            fontsize=8
        )
    
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
    plt.savefig(output_path, dpi=300)
    plt.close() 