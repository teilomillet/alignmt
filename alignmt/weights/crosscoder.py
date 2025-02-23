"""
Crosscoder implementation for analyzing cross-layer features and model differences.

Based on "Sparse Crosscoders for Cross-Layer Features and Model Diffing"
https://transformer-circuits.pub/2024/crosscoders/index.html
"""

from typing import Dict, List, Optional, Tuple, Union
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np

logger = logging.getLogger(__name__)

class Crosscoder(nn.Module):
    """
    Sparse crosscoder that can read from and write to multiple layers.
    
    The crosscoder computes feature activations by combining information from 
    multiple layers and attempts to reconstruct the original activations.
    """
    
    def __init__(
        self,
        layer_dims: Dict[int, int],
        n_features: int,
        sparsity_weight: float = 1e-3,
        activation: nn.Module = nn.ReLU(),
    ):
        """
        Initialize a crosscoder.
        
        Args:
            layer_dims: Dictionary mapping layer indices to their dimensions
            n_features: Number of features to learn
            sparsity_weight: Weight of the L1 regularization term
            activation: Activation function to use (default: ReLU)
        """
        super().__init__()
        
        self.layer_dims = layer_dims
        self.n_features = n_features
        self.sparsity_weight = sparsity_weight
        self.activation = activation
        
        # Create encoder weights for each layer
        self.encoders = nn.ModuleDict({
            f"layer_{l}": nn.Linear(dim, n_features, bias=False)
            for l, dim in layer_dims.items()
        })
        
        # Shared encoder bias
        self.encoder_bias = nn.Parameter(torch.zeros(n_features))
        
        # Create decoder weights for each layer
        self.decoders = nn.ModuleDict({
            f"layer_{l}": nn.Linear(n_features, dim, bias=True)
            for l, dim in layer_dims.items()
        })
        
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights using Kaiming initialization."""
        for module in self.encoders.values():
            nn.init.kaiming_uniform_(module.weight, a=0.1)
            
        for module in self.decoders.values():
            nn.init.kaiming_uniform_(module.weight, a=0.1)
            nn.init.zeros_(module.bias)
    
    def encode(
        self, 
        activations: Dict[int, Tensor]
    ) -> Tensor:
        """
        Encode activations from multiple layers into feature space.
        
        Args:
            activations: Dictionary mapping layer indices to their activations
            
        Returns:
            Feature activations tensor
        """
        # Sum contributions from each layer
        feature_acts = torch.zeros(
            activations[list(activations.keys())[0]].shape[0],
            self.n_features,
            device=self.encoder_bias.device
        )
        
        for layer_idx, acts in activations.items():
            encoder = self.encoders[f"layer_{layer_idx}"]
            feature_acts += encoder(acts)
            
        feature_acts += self.encoder_bias
        
        return self.activation(feature_acts)
    
    def decode(
        self,
        features: Tensor
    ) -> Dict[int, Tensor]:
        """
        Decode features back to layer activations.
        
        Args:
            features: Feature activation tensor
            
        Returns:
            Dictionary mapping layer indices to reconstructed activations
        """
        return {
            layer_idx: self.decoders[f"layer_{layer_idx}"](features)
            for layer_idx in self.layer_dims.keys()
        }
    
    def forward(
        self,
        activations: Dict[int, Tensor]
    ) -> Tuple[Tensor, Dict[int, Tensor]]:
        """
        Forward pass computing features and reconstructions.
        
        Args:
            activations: Dictionary mapping layer indices to their activations
            
        Returns:
            Tuple of (feature activations, reconstructed activations)
        """
        features = self.encode(activations)
        reconstructions = self.decode(features)
        return features, reconstructions
    
    def compute_loss(
        self,
        activations: Dict[int, Tensor],
        features: Optional[Tensor] = None,
        reconstructions: Optional[Dict[int, Tensor]] = None
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        Compute the crosscoder loss following the paper's formulation.
        
        The loss combines reconstruction error with L1 regularization weighted by
        the L1 norm of decoder weight norms.
        
        Args:
            activations: Original activations
            features: Pre-computed features (optional)
            reconstructions: Pre-computed reconstructions (optional)
            
        Returns:
            Tuple of (total loss, loss components dictionary)
        """
        if features is None or reconstructions is None:
            features, reconstructions = self.forward(activations)
            
        # Compute reconstruction loss for each layer
        recon_losses = {}
        for layer_idx, orig_acts in activations.items():
            recon_acts = reconstructions[layer_idx]
            recon_losses[f"recon_loss_{layer_idx}"] = F.mse_loss(
                recon_acts, orig_acts
            )
            
        total_recon_loss = sum(recon_losses.values())
        
        # Compute L1 regularization term weighted by decoder norms
        decoder_norms = torch.stack([
            decoder.weight.norm(p=2, dim=0)
            for decoder in self.decoders.values()
        ])
        
        # Sum of L2 norms across layers for each feature
        feature_decoder_norms = decoder_norms.sum(dim=0)
        
        # Weight L1 penalty by decoder norms
        l1_loss = (features.abs() * feature_decoder_norms[None, :]).mean()
        
        total_loss = total_recon_loss + self.sparsity_weight * l1_loss
        
        losses = {
            "total": total_loss,
            "reconstruction": total_recon_loss,
            "l1_reg": l1_loss,
            **recon_losses
        }
        
        return total_loss, losses
        
    def get_feature_layer_norms(self) -> torch.Tensor:
        """
        Get the norm of each feature's decoder weights at each layer.
        
        Returns:
            Tensor of shape (n_layers, n_features) containing decoder norms
        """
        return torch.stack([
            decoder.weight.norm(p=2, dim=1)
            for decoder in self.decoders.values()
        ])
        
    def analyze_feature_sharing(
        self,
        threshold: float = 0.1
    ) -> Tuple[torch.Tensor, Dict[str, List[int]]]:
        """
        Analyze which features are shared across layers vs layer-specific.
        
        Args:
            threshold: Minimum norm threshold for considering a feature present
            
        Returns:
            Tuple of:
            - Boolean tensor indicating which features are present in each layer
            - Dictionary categorizing features by sharing pattern
        """
        # Get feature norms per layer
        norms = self.get_feature_layer_norms()
        
        # Determine which features are present in each layer
        presence = norms > threshold
        
        # Categorize features
        n_layers_present = presence.sum(dim=0)
        
        categories = {
            "shared": (n_layers_present > 1).nonzero().squeeze(-1).tolist(),
            "single_layer": (n_layers_present == 1).nonzero().squeeze(-1).tolist(),
            "unused": (n_layers_present == 0).nonzero().squeeze(-1).tolist()
        }
        
        return presence, categories 