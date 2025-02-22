"""
Training utilities for crosscoders.
"""

from typing import Dict, Optional, List, Union
import logging
from pathlib import Path
import json

import torch
from torch import Tensor
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from .crosscoder import Crosscoder

logger = logging.getLogger(__name__)

class CrosscoderTrainer:
    """Trainer for crosscoders with logging and checkpointing."""
    
    def __init__(
        self,
        crosscoder: Crosscoder,
        learning_rate: float = 1e-3,
        batch_size: int = 128,
        device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialize trainer.
        
        Args:
            crosscoder: Crosscoder model to train
            learning_rate: Learning rate for Adam optimizer
            batch_size: Training batch size
            device: Device to train on
        """
        self.crosscoder = crosscoder.to(device)
        self.device = device
        self.batch_size = batch_size
        
        self.optimizer = optim.Adam(
            crosscoder.parameters(),
            lr=learning_rate
        )
        
        self.history = {
            "total_loss": [],
            "reconstruction_loss": [],
            "l1_reg_loss": [],
        }
        
    def prepare_data(
        self,
        activations: Dict[int, Tensor]
    ) -> DataLoader:
        """
        Prepare training data loader.
        
        Args:
            activations: Dictionary of layer activations
            
        Returns:
            DataLoader for training
        """
        # Convert activations to tensors and move to device
        tensors = []
        for acts in activations.values():
            # Reshape to (batch_size * hidden_size, intermediate_size)
            batch_size, intermediate_size, hidden_size = acts.shape
            acts_reshaped = acts.permute(0, 2, 1).reshape(-1, intermediate_size)
            tensors.append(acts_reshaped)
        
        dataset = TensorDataset(*tensors)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True
        )
        
    def train_epoch(
        self,
        dataloader: DataLoader,
        desc: str = "Training"
    ) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            dataloader: Training data loader
            desc: Progress bar description
            
        Returns:
            Dictionary of average losses
        """
        self.crosscoder.train()
        running_losses = {}
        
        for batch in dataloader:
            # Prepare batch
            batch_acts = {
                idx: tensor
                for idx, tensor in enumerate(batch)
            }
            
            # Forward pass and loss computation
            self.optimizer.zero_grad()
            features, reconstructions = self.crosscoder(batch_acts)
            loss, losses = self.crosscoder.compute_loss(
                batch_acts,
                features,
                reconstructions
            )
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Update running losses
            for name, value in losses.items():
                if name not in running_losses:
                    running_losses[name] = []
                running_losses[name].append(value.item())
                
        # Compute averages
        avg_losses = {
            name: np.mean(values)
            for name, values in running_losses.items()
        }
        
        # Update history
        self.history["total_loss"].append(avg_losses["total"])
        self.history["reconstruction_loss"].append(avg_losses["reconstruction"])
        self.history["l1_reg_loss"].append(avg_losses["l1_reg"])
        
        return avg_losses
        
    def train(
        self,
        activations: Dict[int, Tensor],
        n_epochs: int,
        checkpoint_dir: Optional[Union[str, Path]] = None,
        checkpoint_freq: int = 10
    ):
        """
        Train the crosscoder.
        
        Args:
            activations: Dictionary of layer activations
            n_epochs: Number of epochs to train
            checkpoint_dir: Directory to save checkpoints
            checkpoint_freq: Epochs between checkpoints
        """
        dataloader = self.prepare_data(activations)
        
        if checkpoint_dir:
            checkpoint_dir = Path(checkpoint_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
        for epoch in range(n_epochs):
            avg_losses = self.train_epoch(
                dataloader,
                desc=f"Epoch {epoch+1}/{n_epochs}"
            )
            
            # Log progress
            logger.info(
                f"Epoch {epoch+1}/{n_epochs} - "
                f"Loss: {avg_losses['total']:.4f} "
                f"(Recon: {avg_losses['reconstruction']:.4f}, "
                f"L1: {avg_losses['l1_reg']:.4f})"
            )
            
            # Save checkpoint
            if checkpoint_dir and (epoch + 1) % checkpoint_freq == 0:
                self.save_checkpoint(
                    checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pt",
                    epoch + 1,
                    avg_losses
                )
                
    def save_checkpoint(
        self,
        path: Union[str, Path],
        epoch: int,
        losses: Dict[str, float]
    ):
        """Save a training checkpoint."""
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.crosscoder.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "losses": losses,
            "history": self.history
        }, path)
        logger.info(f"Saved checkpoint to {path}")
        
    def load_checkpoint(
        self,
        path: Union[str, Path]
    ):
        """Load a training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.crosscoder.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.history = checkpoint["history"]
        
        logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}") 