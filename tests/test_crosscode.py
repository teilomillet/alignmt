"""Tests for the crosscoder implementation."""

import pytest
import torch
from unittest.mock import Mock, patch
import numpy as np

from ops.crosscoder.crosscode import (
    compute_cosine_similarity,
    create_crosscoder_mapping,
    crosscode_layer_params,
    diff_layer_params,
    analyze_layer_changes,
    analyze_model_changes,
)

# Test Constants
MOCK_DIM = 64
MOCK_SOURCE_PARAMS = {
    "model.layers.0.mlp.gate_proj.weight": torch.randn(MOCK_DIM, MOCK_DIM),
    "model.layers.0.mlp.gate_proj.bias": torch.randn(MOCK_DIM),
    "model.layers.0.input_layernorm.weight": torch.randn(MOCK_DIM),
    "model.layers.0.input_layernorm.bias": torch.randn(MOCK_DIM),
}
MOCK_TARGET_PARAMS = {
    "model.layers.0.mlp.gate_proj.weight": torch.randn(MOCK_DIM, MOCK_DIM),
    "model.layers.0.mlp.gate_proj.bias": torch.randn(MOCK_DIM),
    "model.layers.0.input_layernorm.weight": torch.randn(MOCK_DIM),
    "model.layers.0.input_layernorm.bias": torch.randn(MOCK_DIM),
}

def test_compute_cosine_similarity():
    """Test cosine similarity computation with new features."""
    # Create orthogonal vectors
    v1 = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    v2 = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    
    # Test basic similarity
    similarities = compute_cosine_similarity(v1, v2)
    assert similarities.shape == (2, 2)
    assert torch.allclose(similarities, torch.tensor([[0.0, 1.0], [1.0, 0.0]]), atol=1e-6)
    
    # Test non-linear transformation
    v3 = torch.tensor([[-1.0, 2.0], [3.0, -4.0]])
    v4 = torch.tensor([[2.0, -1.0], [-4.0, 3.0]])
    nl_similarities = compute_cosine_similarity(v3, v4, non_linear=True)
    assert nl_similarities.shape == (2, 2)
    
    # Test attention patterns
    attn1 = torch.randn(2, 4, 8, 8)  # (batch, heads, seq_len, seq_len)
    attn2 = torch.randn(2, 4, 8, 8)
    attn_similarities = compute_cosine_similarity(attn1, attn2, attention_patterns=True)
    assert attn_similarities.shape == (8, 8)  # (batch*heads, batch*heads)

def test_create_crosscoder_mapping():
    """Test crosscoder mapping creation with new features."""
    # Create test tensors with proper dimensions
    source = torch.randn(5, 10)
    target = torch.randn(3, 10)
    
    # Test with different temperatures
    mapping_hot = create_crosscoder_mapping(source, target, temperature=0.1)
    mapping_cold = create_crosscoder_mapping(source, target, temperature=10.0)
    
    # Check shapes
    assert mapping_hot.shape == (5, 3)
    assert mapping_cold.shape == (5, 3)
    
    # Check probability properties
    assert torch.allclose(mapping_hot.sum(dim=1), torch.ones(5))
    assert torch.allclose(mapping_cold.sum(dim=1), torch.ones(5))
    
    # Higher temperature should give more uniform probabilities
    assert mapping_hot.std() > mapping_cold.std()
    
    # Test with residual connections
    mapping_residual = create_crosscoder_mapping(source, target, residual_weight=0.5)
    assert mapping_residual.shape == (5, 3)
    assert torch.allclose(mapping_residual.sum(dim=1), torch.ones(5))
    
    # Test with non-linear similarity
    mapping_nonlinear = create_crosscoder_mapping(source, target, non_linear=True)
    assert mapping_nonlinear.shape == (5, 3)
    assert torch.allclose(mapping_nonlinear.sum(dim=1), torch.ones(5))
    
    # Test identical tensors
    source = torch.eye(5)
    target = source.clone()
    mapping = create_crosscoder_mapping(source, target)
    assert torch.allclose(mapping, torch.eye(5))

def test_crosscode_layer_params():
    """Test layer parameter crosscoding with new features."""
    # Create test parameters with proper dimensions
    dim = 64
    num_heads = 8
    head_size = dim // num_heads
    
    source_params = {
        "model.layers.0.mlp.gate_proj.weight": torch.randn(dim, dim),
        "model.layers.0.mlp.gate_proj.bias": torch.randn(dim),
        "model.layers.0.input_layernorm.weight": torch.randn(dim),
        "model.layers.0.input_layernorm.bias": torch.randn(dim),
        "model.layers.0.attention.weight": torch.randn(dim, num_heads * head_size),
        "model.layers.0.other_param": torch.randn(dim, dim)
    }
    
    target_params = {
        "model.layers.0.mlp.gate_proj.weight": torch.randn(dim, dim),
        "model.layers.0.mlp.gate_proj.bias": torch.randn(dim),
        "model.layers.0.input_layernorm.weight": torch.randn(dim),
        "model.layers.0.input_layernorm.bias": torch.randn(dim),
        "model.layers.0.attention.weight": torch.randn(dim, num_heads * head_size),
        "model.layers.0.other_param": torch.randn(dim, dim)
    }
    
    # Test standard crosscoding
    result = crosscode_layer_params(source_params, target_params)
    assert set(result.keys()) == set(source_params.keys())
    
    # Test with residual connections
    result_residual = crosscode_layer_params(
        source_params,
        target_params,
        residual_weight=0.5
    )
    assert set(result_residual.keys()) == set(source_params.keys())
    
    # Test with non-linear similarity
    result_nonlinear = crosscode_layer_params(
        source_params,
        target_params,
        non_linear=True
    )
    assert set(result_nonlinear.keys()) == set(source_params.keys())
    
    # Verify attention handling
    attn_name = "model.layers.0.attention.weight"
    assert result[attn_name].shape == source_params[attn_name].shape
    assert not torch.allclose(result[attn_name], target_params[attn_name])
    
    # Test error handling
    bad_source_params = {
        "bad.weight": torch.randn(dim),  # 1D weight (should be 2D)
        "bad.attention.weight": torch.randn(dim)  # 1D attention (should be 2D)
    }
    bad_target_params = {
        "bad.weight": torch.randn(dim),
        "bad.attention.weight": torch.randn(dim)
    }
    
    with pytest.raises(ValueError, match="Weight tensor .* must be a matrix"):
        crosscode_layer_params(bad_source_params, bad_target_params)

def test_diff_layer_params():
    """Test layer parameter differencing."""
    # Create orthonormal test parameters with proper dimensions
    dim = 5
    source_params = {
        "weight": torch.eye(dim),
        "bias": torch.ones(dim),  # Changed from zeros to ones
        "norm": torch.ones(dim)
    }
    target_params = {
        "weight": torch.eye(dim),
        "bias": torch.ones(dim) * 2,  # Scaled version of source
        "norm": torch.ones(dim) * 2
    }
    
    # Test without crosscoding
    results = diff_layer_params(source_params, target_params, crosscode=False)
    
    # Check structure
    assert set(results.keys()) == {"differences", "similarities", "crosscoded"}
    
    # Check differences
    assert isinstance(results["differences"]["weight"], float)
    assert abs(results["differences"]["weight"]) < 1e-5  # Identical matrices
    assert abs(results["differences"]["bias"] - 1.0) < 1e-5  # Scaled vector (2x)
    assert abs(results["differences"]["norm"] - 1.0) < 1e-5  # Scaled vector (2x)
    
    # Check similarities
    assert isinstance(results["similarities"]["weight"], float)
    assert abs(results["similarities"]["weight"] - 1.0) < 1e-5  # Perfect similarity for identical matrices

def test_analyze_layer_changes():
    """Test layer change analysis."""
    dim = 4
    source_params = {
        "model.layers.0.mlp.gate_proj.weight": torch.randn(dim, dim),
        "model.layers.0.mlp.gate_proj.bias": torch.randn(dim),
        "model.layers.0.input_layernorm.weight": torch.randn(dim),
        "model.layers.0.input_layernorm.bias": torch.randn(dim)
    }
    
    # Create target params with known differences
    target_params = {
        "model.layers.0.mlp.gate_proj.weight": torch.randn(dim, dim),
        "model.layers.0.mlp.gate_proj.bias": source_params["model.layers.0.mlp.gate_proj.bias"] * 2,  # Scaled
        "model.layers.0.input_layernorm.weight": source_params["model.layers.0.input_layernorm.weight"],  # Same
        "model.layers.0.input_layernorm.bias": -source_params["model.layers.0.input_layernorm.bias"]  # Negated (scaled by -1)
    }
    
    result = analyze_layer_changes(source_params, target_params)
    
    # Check structure
    assert isinstance(result, dict)
    assert set(result.keys()) == {"differences", "similarities", "crosscoded"}
    
    # Check differences
    assert isinstance(result["differences"], dict)
    for param_name in source_params:
        assert param_name in result["differences"]
        assert isinstance(result["differences"][param_name], float)
    
    # Verify specific changes
    # Weight should have non-zero difference due to random initialization
    assert result["differences"]["model.layers.0.mlp.gate_proj.weight"] > 0
    
    # Bias should show scaling difference
    assert abs(result["differences"]["model.layers.0.mlp.gate_proj.bias"] - 1.0) < 1e-5
    
    # Layernorm weight should have zero difference
    assert abs(result["differences"]["model.layers.0.input_layernorm.weight"]) < 1e-5
    
    # Layernorm bias should show scaling (negation is scaling by -1)
    assert abs(result["differences"]["model.layers.0.input_layernorm.bias"] - 1.0) < 1e-5
    
    # Check similarities
    assert isinstance(result["similarities"], dict)
    for param_name in source_params:
        assert param_name in result["similarities"]
        assert isinstance(result["similarities"][param_name], float)
        assert 0 <= result["similarities"][param_name] <= 1
    
    # Check crosscoded parameters
    assert isinstance(result["crosscoded"], dict)
    assert set(result["crosscoded"].keys()) == set(source_params.keys())
    for param_name, param in result["crosscoded"].items():
        assert param.shape == source_params[param_name].shape

@patch("ops.crosscoder.crosscode.get_layer_names")
@patch("ops.crosscoder.crosscode.analyze_layer_changes")
def test_analyze_model_changes(mock_analyze_layer, mock_get_names):
    """Test full model change analysis."""
    # Mock layer names
    mock_get_names.return_value = ["layer1", "layer2"]
    
    # Mock layer analysis results
    mock_results = {
        'differences': {
            'weight': torch.randn(10, MOCK_DIM),
            'bias': torch.randn(10)
        },
        'similarities': {
            'weight': torch.randn(10, 8)
        },
        'crosscoded': {
            'weight': torch.randn(10, MOCK_DIM),
            'bias': torch.randn(10)
        }
    }
    mock_analyze_layer.return_value = mock_results
    
    results = analyze_model_changes(
        "source_model",
        "target_model"
    )
    
    # Check structure
    assert set(results.keys()) == {"layer1", "layer2"}
    assert all(
        set(layer_results.keys()) == set(mock_results.keys())
        for layer_results in results.values()
    )
    
    # Check that each layer preserves parameter structure
    for layer_name, layer_results in results.items():
        assert set(layer_results["differences"].keys()) == {"weight", "bias"}
        assert set(layer_results["similarities"].keys()) == {"weight"}
        assert set(layer_results["crosscoded"].keys()) == {"weight", "bias"} 