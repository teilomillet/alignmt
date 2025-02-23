"""
Example of using crosscoders for model diffing.

This script demonstrates how to:
1. Load two models layer by layer
2. Process layers in batches
3. Train a crosscoder
4. Analyze shared and model-specific features
"""

import torch
from pathlib import Path
import logging
import gc
from typing import Dict, Iterator, Tuple
from itertools import islice

from alignmt.weights import (
    iterate_model_layers,
    Crosscoder,
    CrosscoderTrainer
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_device_with_most_memory():
    """Get the device (CPU/GPU) with the most available memory."""
    if not torch.cuda.is_available():
        return "cpu"
        
    # Check available GPU memory
    device = "cpu"
    max_memory = 0
    
    for i in range(torch.cuda.device_count()):
        free_memory = torch.cuda.get_device_properties(i).total_memory - torch.cuda.memory_allocated(i)
        if free_memory > max_memory:
            max_memory = free_memory
            device = f"cuda:{i}"
            
    # If GPU has less than 4GB free, use CPU
    if max_memory < 4 * 1024 * 1024 * 1024:
        logger.warning("Low GPU memory, using CPU instead")
        device = "cpu"
        
    return device

def process_layer_batch(
    qwen_iter: Iterator[Tuple[str, torch.Tensor]],
    deepseek_iter: Iterator[Tuple[str, torch.Tensor]],
    batch_size: int,
    device: str
) -> Tuple[Dict[int, int], Dict[int, torch.Tensor]]:
    """
    Process a batch of layers from both models.
    
    Args:
        qwen_iter: Iterator for Qwen layers
        deepseek_iter: Iterator for DeepSeek layers
        batch_size: Number of layers to process at once
        device: Device to use
        
    Returns:
        Tuple of (layer dimensions, activations)
    """
    layer_dims = {}
    activations = {}
    
    # Get next batch of layers
    qwen_layers = list(islice(qwen_iter, batch_size))
    deepseek_layers = list(islice(deepseek_iter, batch_size))
    
    if not qwen_layers or not deepseek_layers:
        return None, None
    
    # Process each layer pair
    for i, ((qwen_name, qwen_w), (deepseek_name, deepseek_w)) in enumerate(
        zip(qwen_layers, deepseek_layers)
    ):
        if qwen_w.shape != deepseek_w.shape:
            logger.warning(
                f"Shape mismatch: {qwen_name} {qwen_w.shape} vs "
                f"{deepseek_name} {deepseek_w.shape}"
            )
            continue
            
        layer_idx = len(layer_dims)
        # Use the intermediate dimension (8960)
        layer_dims[layer_idx] = qwen_w.shape[0]
        
        # Stack the weights to create batches
        acts = torch.stack([qwen_w, deepseek_w])  # Shape: (2, 8960, 1536)
        activations[layer_idx] = acts.to(device)
        
        # Clean up
        del qwen_w, deepseek_w
        
    gc.collect()
    torch.cuda.empty_cache()
    
    return layer_dims, activations

def main():
    # Create output directory
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine device
    device = get_device_with_most_memory()
    logger.info(f"Using device: {device}")
    
    # Create iterators for both models
    qwen_iter = iterate_model_layers(
        "Qwen/Qwen2-1.5B",
        device=device,
        cache_dir="model_cache"
    )
    
    deepseek_iter = iterate_model_layers(
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        device=device,
        cache_dir="model_cache"
    )
    
    # Process layers in batches
    layer_batch_size = 2  # Process 2 layers at a time
    training_batch_size = 16 if device != "cpu" else 4  # Smaller training batches
    n_features = 128  # Reduced number of features
    all_layer_dims = {}
    crosscoders = []
    
    while True:
        logger.info(f"Processing batch of {layer_batch_size} layers...")
        layer_dims, activations = process_layer_batch(
            qwen_iter,
            deepseek_iter,
            layer_batch_size,
            device
        )
        
        if layer_dims is None:
            break
            
        # Update total layer dimensions
        all_layer_dims.update(layer_dims)
        
        # Create crosscoder for this batch
        logger.info("Creating crosscoder...")
        crosscoder = Crosscoder(
            layer_dims=layer_dims,
            n_features=n_features,
            sparsity_weight=1e-3
        )
        
        # Create trainer
        trainer = CrosscoderTrainer(
            crosscoder,
            learning_rate=1e-3,
            batch_size=training_batch_size,
            device=device
        )
        
        # Train on this batch
        logger.info("Training crosscoder...")
        try:
            trainer.train(
                activations=activations,
                n_epochs=25,  # Further reduced number of epochs
                checkpoint_dir=output_dir / f"checkpoints_batch_{len(crosscoders)}",
                checkpoint_freq=5
            )
            crosscoders.append(trainer.crosscoder)
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                logger.error(
                    "Out of memory error. Try:\n"
                    "1. Reducing batch_size\n"
                    "2. Reducing n_features\n"
                    "3. Using CPU instead of GPU\n"
                    "4. Processing fewer layers at once"
                )
            raise
            
        # Clean up
        del activations
        gc.collect()
        torch.cuda.empty_cache()
    
    # Analyze results for each batch
    logger.info("Analyzing features...")
    for i, crosscoder in enumerate(crosscoders):
        crosscoder = crosscoder.cpu()
        presence, categories = crosscoder.analyze_feature_sharing(threshold=0.1)
        
        print(f"\nBatch {i} Feature Analysis:")
        print(f"Shared features: {len(categories['shared'])}")
        print(f"Layer-specific features: {len(categories['single_layer'])}")
        print(f"Unused features: {len(categories['unused'])}")
        
        # Get feature importance
        norms = crosscoder.get_feature_layer_norms()
        feature_importance = norms.sum(dim=0)
        
        # Find most important features
        top_k = 5
        top_features = torch.topk(feature_importance, top_k)
        
        print(f"\nTop {top_k} Most Important Features in Batch {i}:")
        for idx, importance in zip(top_features.indices, top_features.values):
            print(f"Feature {idx}: {importance:.4f}")
            
        # Save batch results
        torch.save({
            "presence": presence,
            "categories": categories,
            "norms": norms,
            "feature_importance": feature_importance
        }, output_dir / f"analysis_results_batch_{i}.pt")
    
    logger.info("Analysis complete! Results saved to outputs/")

if __name__ == "__main__":
    main() 