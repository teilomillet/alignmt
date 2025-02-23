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
    """Test cosine similarity computation."""
    # Create orthogonal vectors
    v1 = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    v2 = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    
    similarities = compute_cosine_similarity(v1, v2)
    
    # Check dimensions
    assert similarities.shape == (2, 2)
    
    # Check values (orthogonal vectors should have 0 similarity)
    assert torch.allclose(similarities, torch.tensor([[0.0, 1.0], [1.0, 0.0]]), atol=1e-6)

def test_create_crosscoder_mapping():
    """Test crosscoder mapping creation."""
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
    
    # Test identical tensors
    source = torch.eye(5)
    target = source.clone()
    mapping = create_crosscoder_mapping(source, target)
    assert torch.allclose(mapping, torch.eye(5))

def test_crosscode_layer_params():
    """Test layer parameter crosscoding."""
    # Create test parameters with proper dimensions
    dim = 4
    source_params = {
        "model.layers.0.mlp.gate_proj.weight": torch.randn(dim, dim),  # Matrix
        "model.layers.0.mlp.gate_proj.bias": torch.randn(dim),         # Vector
        "model.layers.0.input_layernorm.weight": torch.randn(dim),     # Vector
        "model.layers.0.input_layernorm.bias": torch.randn(dim),       # Vector
        "model.layers.0.other_param": torch.randn(dim, dim)            # Unknown type
    }
    target_params = {
        "model.layers.0.mlp.gate_proj.weight": torch.randn(dim, dim),
        "model.layers.0.mlp.gate_proj.bias": torch.randn(dim),
        "model.layers.0.input_layernorm.weight": torch.randn(dim),
        "model.layers.0.input_layernorm.bias": torch.randn(dim),
        "model.layers.0.other_param": torch.randn(dim, dim)
    }
    
    # Test crosscoding
    result = crosscode_layer_params(source_params, target_params)
    
    # Check that all parameters are present
    assert set(result.keys()) == set(source_params.keys())
    
    # Check weight matrix crosscoding
    weight_name = "model.layers.0.mlp.gate_proj.weight"
    assert not torch.allclose(result[weight_name], target_params[weight_name])
    
    # Check that bias and norm were copied exactly
    for name in ["model.layers.0.mlp.gate_proj.bias",
                "model.layers.0.input_layernorm.weight",
                "model.layers.0.input_layernorm.bias"]:
        assert torch.allclose(result[name], target_params[name])
    
    # Check unknown parameter type handling
    assert torch.allclose(result["model.layers.0.other_param"], 
                         target_params["model.layers.0.other_param"])
    
    # Test error handling for incorrect dimensions
    bad_source_params = {
        "bad.weight": torch.randn(dim),  # 1D weight (should be 2D)
        "bad.bias": torch.randn(dim, dim)  # 2D bias (should be 1D)
    }
    bad_target_params = {
        "bad.weight": torch.randn(dim),
        "bad.bias": torch.randn(dim, dim)
    }
    
    # Should raise error for incorrect weight dimension
    with pytest.raises(ValueError, match="Weight tensor .* must be a matrix"):
        crosscode_layer_params(bad_source_params, bad_target_params)
    
    # Should raise error for incorrect bias dimension
    with pytest.raises(ValueError, match="Parameter .* must be a vector"):
        crosscode_layer_params(
            {"good.bias": torch.randn(dim, dim)},
            {"good.bias": torch.randn(dim, dim)}
        )

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