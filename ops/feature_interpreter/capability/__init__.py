"""
Feature capability testing package.

This package provides modules and functions for generating contrastive examples
and evaluating model capabilities based on feature interpretations.
"""

# Import functions from modules
from .examples import generate_contrastive_examples
from .evaluation import generate_response, evaluate_feature_capability
from .metrics import calculate_human_experience_score

# Define public API
__all__ = [
    'generate_contrastive_examples',
    'generate_response',
    'calculate_human_experience_score',
    'evaluate_feature_capability'
] 