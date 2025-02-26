"""
Model activations extraction package.

This package provides modules and functions for extracting activations from
language models for feature-level interpretation.
"""

# Import functions from modules
from .hooks import register_activation_hooks
from .extraction import extract_activations, extract_activations_for_comparison

# Define public API
__all__ = [
    'register_activation_hooks',
    'extract_activations',
    'extract_activations_for_comparison'
] 