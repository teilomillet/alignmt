"""
Integration tests for the complete crosscoder pipeline.
"""

import pytest
import torch
import numpy as np
from pathlib import Path

from alignmt.weights import (
    Crosscoder,
    CrosscoderTrainer,
    load_model_and_tokenizer,
    extract_model_weights
)

@pytest.fixture
def mock_model():
    """Create a small mock transformer model for testing."""
    class MockTransformer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([
                torch.nn.Linear(64, 64) for _ in range(2)
            ])
            
        def state_dict(self):
            return {
                f"layer.{i}.weight": layer.weight
                for i, layer in enumerate(self.layers)
            }
    
    return MockTransformer()

def test_complete_pipeline(mock_model, tmp_path):
    """Test the complete crosscoder pipeline from model loading to feature analysis."""
    
    # Extract weights from mock model
    weights = extract_model_weights(mock_model)
    
    # Prepare activations (simulate model outputs)
    batch_size = 100
    activations = {
        i: torch.randn(batch_size, 64)
        for i in range(2)
    }
    
    # Create and configure crosscoder
    crosscoder = Crosscoder(
        layer_dims={0: 64, 1: 64},
        n_features=32,
        sparsity_weight=1e-3
    )
    
    # Create trainer
    trainer = CrosscoderTrainer(
        crosscoder,
        learning_rate=1e-3,
        batch_size=32
    )
    
    # Train for a few epochs
    checkpoint_dir = tmp_path / "checkpoints"
    trainer.train(
        activations=activations,
        n_epochs=3,
        checkpoint_dir=checkpoint_dir,
        checkpoint_freq=1
    )
    
    # Verify training progress
    assert len(trainer.history["total_loss"]) == 3
    assert trainer.history["total_loss"][-1] < trainer.history["total_loss"][0]
    
    # Check that checkpoints were saved
    assert (checkpoint_dir / "checkpoint_epoch_1.pt").exists()
    assert (checkpoint_dir / "checkpoint_epoch_2.pt").exists()
    assert (checkpoint_dir / "checkpoint_epoch_3.pt").exists()
    
    # Analyze feature sharing
    presence, categories = crosscoder.analyze_feature_sharing(threshold=0.1)
    
    # Verify analysis results
    assert presence.shape == (2, 32)  # n_layers x n_features
    assert all(k in categories for k in ["shared", "single_layer", "unused"])
    assert sum(len(v) for v in categories.values()) == 32
    
    # Test feature extraction
    features, reconstructions = crosscoder(activations)
    
    # Verify reconstructions
    for layer_idx in [0, 1]:
        orig = activations[layer_idx]
        recon = reconstructions[layer_idx]
        
        # Check shapes
        assert orig.shape == recon.shape
        
        # Compute reconstruction error
        error = torch.mean((orig - recon) ** 2)
        assert error < 1.0  # Reasonable reconstruction
        
    # Test model diffing capabilities
    norms = crosscoder.get_feature_layer_norms()
    assert norms.shape == (2, 32)
    
    # Verify that some features are active
    assert torch.any(norms > 0.1)
    
    # Test loading from checkpoint
    new_trainer = CrosscoderTrainer(
        Crosscoder(layer_dims={0: 64, 1: 64}, n_features=32),
        learning_rate=1e-3
    )
    new_trainer.load_checkpoint(checkpoint_dir / "checkpoint_epoch_3.pt")
    
    # Verify state restoration
    assert len(new_trainer.history["total_loss"]) == 3
    assert torch.allclose(
        new_trainer.crosscoder.get_feature_layer_norms(),
        norms
    ) 