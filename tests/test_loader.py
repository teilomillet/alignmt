"""Tests for the model loading utilities."""

import pytest
import torch
from unittest.mock import Mock, patch, call
from pathlib import Path
import gc

from ops.loader.load import (
    get_layer_names,
    load_model_layer,
    iterate_model_layers,
    load_model_and_tokenizer,
    extract_model_weights,
    load_qwen_weights,
    load_deepseek_weights,
)

# Test Constants
TEST_CACHE_DIR = "test_cache"
MOCK_TENSOR = torch.randn(100, 100)

@pytest.fixture
def mock_model():
    """Create a mock model with basic attributes."""
    model = Mock()
    # Set up the state dict
    model.state_dict.return_value = {
        "model.layers.0.mlp.gate_proj.weight": MOCK_TENSOR,
        "model.layers.1.mlp.gate_proj.weight": MOCK_TENSOR,
    }
    # Set up named_parameters to return an iterator
    model.named_parameters.return_value = iter([
        ("model.layers.0.mlp.gate_proj.weight", MOCK_TENSOR),
        ("model.layers.1.mlp.gate_proj.weight", MOCK_TENSOR),
    ])
    return model

@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer."""
    return Mock()

class TestGetLayerNames:
    """Tests for the get_layer_names function."""
    
    def test_qwen_layer_names(self):
        """Test layer name generation for Qwen model."""
        names = get_layer_names("Qwen/Qwen2-1.5B")
        assert len(names) == 28
        assert names[0] == "model.layers.0.mlp.gate_proj.weight"
        assert names[-1] == "model.layers.27.mlp.gate_proj.weight"
    
    def test_deepseek_layer_names(self):
        """Test layer name generation for DeepSeek model."""
        names = get_layer_names("DeepSeek-ai/something")
        assert len(names) == 28
        assert names[0] == "model.layers.0.mlp.gate_proj.weight"
        assert names[-1] == "model.layers.27.mlp.gate_proj.weight"
    
    def test_unsupported_model(self):
        """Test error handling for unsupported models."""
        with pytest.raises(ValueError, match="Unsupported model"):
            get_layer_names("UnsupportedModel")

@pytest.mark.asyncio
class TestLoadModelLayer:
    """Tests for the load_model_layer function."""
    
    @patch("ops.loader.load.AutoModelForCausalLM")
    @patch("ops.loader.load.gc")
    @patch("torch.cuda")
    async def test_memory_cleanup(self, mock_cuda, mock_gc, mock_auto_model):
        """Test memory cleanup after layer loading."""
        # Setup mock model
        mock_model = Mock()
        mock_model.named_parameters.return_value = iter([
            ("model.layers.0.mlp.gate_proj.weight", MOCK_TENSOR),
        ])
        mock_auto_model.from_pretrained.return_value = mock_model
        
        # Load a layer
        layer = load_model_layer(
            "Qwen/Qwen2-1.5B",
            "model.layers.0.mlp.gate_proj.weight",
            cache_dir=TEST_CACHE_DIR
        )
        
        # Verify cleanup calls
        mock_gc.collect.assert_called_once()
        mock_cuda.empty_cache.assert_called_once()
        
        # Verify model was loaded with memory optimization params
        mock_auto_model.from_pretrained.assert_called_once_with(
            "Qwen/Qwen2-1.5B",
            cache_dir=TEST_CACHE_DIR,
            trust_remote_code=True,
            torch_dtype=torch.float32,
            device_map={"": "cuda"},
            max_memory={"cuda": "4GiB"},
            offload_folder="offload",
            low_cpu_mem_usage=True
        )
    
    @patch("ops.loader.load.AutoModelForCausalLM")
    async def test_device_placement(self, mock_auto_model):
        """Test layer loading on specific device."""
        mock_model = Mock()
        mock_model.named_parameters.return_value = iter([
            ("model.layers.0.mlp.gate_proj.weight", MOCK_TENSOR),
        ])
        mock_auto_model.from_pretrained.return_value = mock_model
        
        # Load layer on CPU
        layer = load_model_layer(
            "Qwen/Qwen2-1.5B",
            "model.layers.0.mlp.gate_proj.weight",
            device="cpu",
            cache_dir=TEST_CACHE_DIR
        )
        
        # Verify device mapping
        mock_auto_model.from_pretrained.assert_called_once_with(
            "Qwen/Qwen2-1.5B",
            cache_dir=TEST_CACHE_DIR,
            trust_remote_code=True,
            torch_dtype=torch.float32,
            device_map={"": "cpu"},
            max_memory={"cpu": "4GiB"},
            offload_folder="offload",
            low_cpu_mem_usage=True
        )

@pytest.mark.asyncio
class TestIterateModelLayers:
    """Tests for the iterate_model_layers function."""
    
    @patch("ops.loader.load.load_model_layer")
    @patch("ops.loader.load.gc")
    @patch("torch.cuda")
    async def test_memory_cleanup_during_iteration(self, mock_cuda, mock_gc, mock_load_layer):
        """Test memory cleanup between layer iterations."""
        mock_load_layer.return_value = MOCK_TENSOR
        
        # Iterate through layers
        layers = list(iterate_model_layers("Qwen/Qwen2-1.5B", cache_dir=TEST_CACHE_DIR))
        
        # Verify cleanup after each layer
        assert mock_gc.collect.call_count == 28  # Once per layer
        assert mock_cuda.empty_cache.call_count == 28  # Once per layer
        
        # Verify all layers were loaded
        assert len(layers) == 28
        assert all(isinstance(tensor, torch.Tensor) for _, tensor in layers)

@pytest.mark.asyncio
class TestLoadModelAndTokenizer:
    """Tests for the load_model_and_tokenizer function."""
    
    @patch("ops.loader.load.AutoModelForCausalLM")
    @patch("ops.loader.load.AutoTokenizer")
    async def test_successful_load(self, mock_tokenizer_cls, mock_model_cls, mock_model, mock_tokenizer):
        """Test successful loading of model and tokenizer."""
        mock_model_cls.from_pretrained.return_value = mock_model
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer
        
        model, tokenizer = load_model_and_tokenizer(
            "Qwen/Qwen2-1.5B",
            cache_dir=TEST_CACHE_DIR
        )
        
        assert model == mock_model
        assert tokenizer == mock_tokenizer
        mock_model_cls.from_pretrained.assert_called_once()
        mock_tokenizer_cls.from_pretrained.assert_called_once()

class TestExtractModelWeights:
    """Tests for the extract_model_weights function."""
    
    def test_weight_extraction(self, mock_model):
        """Test successful weight extraction."""
        weights = extract_model_weights(mock_model)
        
        assert len(weights) == 2
        assert all(isinstance(tensor, torch.Tensor) for tensor in weights.values())
    
    def test_detach_behavior(self, mock_model):
        """Test detach parameter behavior."""
        weights_detached = extract_model_weights(mock_model, detach=True)
        weights_attached = extract_model_weights(mock_model, detach=False)
        
        assert all(tensor.requires_grad == False for tensor in weights_detached.values())
        assert all(isinstance(tensor, torch.Tensor) for tensor in weights_attached.values())

@pytest.mark.asyncio
class TestModelSpecificLoaders:
    """Tests for model-specific loading functions."""
    
    @patch("ops.loader.load.iterate_model_layers")
    async def test_qwen_weights_loading(self, mock_iterate):
        """Test Qwen weights loading."""
        mock_iterate.return_value = [
            ("layer1", MOCK_TENSOR),
            ("layer2", MOCK_TENSOR)
        ]
        
        weights = load_qwen_weights(cache_dir=TEST_CACHE_DIR)
        
        assert len(weights) == 2
        assert all(isinstance(tensor, torch.Tensor) for tensor in weights.values())
    
    @patch("ops.loader.load.iterate_model_layers")
    async def test_deepseek_weights_loading(self, mock_iterate):
        """Test DeepSeek weights loading."""
        mock_iterate.return_value = [
            ("layer1", MOCK_TENSOR),
            ("layer2", MOCK_TENSOR)
        ]
        
        weights = load_deepseek_weights(cache_dir=TEST_CACHE_DIR)
        
        assert len(weights) == 2
        assert all(isinstance(tensor, torch.Tensor) for tensor in weights.values()) 