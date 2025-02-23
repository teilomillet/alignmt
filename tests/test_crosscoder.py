"""
Tests for the crosscoder implementation.
"""

import pytest
import torch
import torch.nn as nn
from pathlib import Path

from alignmt.weights.crosscoder import Crosscoder
from alignmt.weights.trainer import CrosscoderTrainer

@pytest.fixture
def sample_activations():
    """Create sample activations for testing."""
    torch.manual_seed(42)
    return {
        0: torch.randn(100, 64),  # 100 samples, 64 dimensions
        1: torch.randn(100, 64),  # Same for layer 1
    }

@pytest.fixture
def crosscoder():
    """Create a test crosscoder instance."""
    layer_dims = {0: 64, 1: 64}
    return Crosscoder(
        layer_dims=layer_dims,
        n_features=32,
        sparsity_weight=1e-3
    )

def test_crosscoder_initialization(crosscoder):
    """Test that crosscoder initializes correctly."""
    # Check basic attributes
    assert crosscoder.n_features == 32
    assert crosscoder.sparsity_weight == 1e-3
    assert isinstance(crosscoder.activation, nn.ReLU)
    
    # Check encoder/decoder structure
    assert len(crosscoder.encoders) == 2
    assert len(crosscoder.decoders) == 2
    
    # Check dimensions
    for layer_idx in [0, 1]:
        encoder = crosscoder.encoders[f"layer_{layer_idx}"]
        decoder = crosscoder.decoders[f"layer_{layer_idx}"]
        
        assert encoder.weight.shape == (32, 64)  # n_features x input_dim
        assert decoder.weight.shape == (64, 32)  # output_dim x n_features
        assert decoder.bias.shape == (64,)       # output_dim

def test_crosscoder_forward(crosscoder, sample_activations):
    """Test the forward pass of the crosscoder."""
    # Run forward pass
    features, reconstructions = crosscoder(sample_activations)
    
    # Check shapes
    assert features.shape == (100, 32)  # batch_size x n_features
    assert len(reconstructions) == 2
    
    for layer_idx in [0, 1]:
        assert reconstructions[layer_idx].shape == (100, 64)

def test_crosscoder_loss(crosscoder, sample_activations):
    """Test the loss computation."""
    features, reconstructions = crosscoder(sample_activations)
    loss, losses = crosscoder.compute_loss(
        sample_activations,
        features,
        reconstructions
    )
    
    # Check that loss components exist and are reasonable
    assert isinstance(loss, torch.Tensor)
    assert loss.item() > 0
    
    assert "total" in losses
    assert "reconstruction" in losses
    assert "l1_reg" in losses
    assert "recon_loss_0" in losses
    assert "recon_loss_1" in losses
    
    # Check that loss components sum correctly
    assert torch.isclose(
        losses["total"],
        losses["reconstruction"] + crosscoder.sparsity_weight * losses["l1_reg"]
    )

def test_feature_analysis(crosscoder):
    """Test feature analysis methods."""
    # Get feature norms
    norms = crosscoder.get_feature_layer_norms()
    assert norms.shape == (2, 32)  # n_layers x n_features
    
    # Test feature sharing analysis
    presence, categories = crosscoder.analyze_feature_sharing(threshold=0.1)
    
    assert presence.shape == (2, 32)  # n_layers x n_features
    assert all(k in categories for k in ["shared", "single_layer", "unused"])
    assert sum(len(v) for v in categories.values()) == 32  # All features categorized

@pytest.fixture
def trainer(crosscoder):
    """Create a test trainer instance."""
    return CrosscoderTrainer(
        crosscoder,
        learning_rate=1e-3,
        batch_size=32
    )

def test_trainer_initialization(trainer):
    """Test trainer initialization."""
    assert trainer.batch_size == 32
    assert isinstance(trainer.optimizer, torch.optim.Adam)
    assert all(k in trainer.history for k in [
        "total_loss", "reconstruction_loss", "l1_reg_loss"
    ])

def test_trainer_data_preparation(trainer, sample_activations):
    """Test data loader preparation."""
    dataloader = trainer.prepare_data(sample_activations)
    
    # Check dataloader properties
    assert isinstance(dataloader.dataset[0], tuple)
    assert len(dataloader.dataset[0]) == 2  # Two layers
    assert dataloader.batch_size == 32

def test_training_step(trainer, sample_activations):
    """Test a single training step."""
    dataloader = trainer.prepare_data(sample_activations)
    
    # Run one epoch
    losses = trainer.train_epoch(dataloader)
    
    # Check that losses were recorded
    assert len(trainer.history["total_loss"]) == 1
    assert len(trainer.history["reconstruction_loss"]) == 1
    assert len(trainer.history["l1_reg_loss"]) == 1
    
    # Check that losses decreased
    if len(trainer.history["total_loss"]) > 1:
        assert trainer.history["total_loss"][-1] < trainer.history["total_loss"][0]

def test_checkpointing(trainer, tmp_path):
    """Test checkpoint saving and loading."""
    # Save a checkpoint
    checkpoint_path = tmp_path / "checkpoint.pt"
    trainer.save_checkpoint(checkpoint_path, epoch=1, losses={
        "total": 1.0,
        "reconstruction": 0.8,
        "l1_reg": 0.2
    })
    
    # Load the checkpoint
    new_trainer = CrosscoderTrainer(Crosscoder(
        layer_dims={0: 64, 1: 64},
        n_features=32
    ))
    new_trainer.load_checkpoint(checkpoint_path)
    
    # Check that state was restored
    assert len(new_trainer.history["total_loss"]) == len(trainer.history["total_loss"])
    
    # Check model parameters are identical
    for p1, p2 in zip(
        trainer.crosscoder.parameters(),
        new_trainer.crosscoder.parameters()
    ):
        assert torch.allclose(p1, p2) 