"""
Feature naming and interpretation module.

This module provides functions for identifying, naming, and interpreting
distinctive features in neural networks.
"""

from .differences import compute_activation_differences, analyze_output_differences
from .extraction import extract_distinctive_features
from .interpretation import interpret_feature_differences
from .validation import causal_feature_validation
from .pipeline import name_features

__all__ = [
    'compute_activation_differences',
    'analyze_output_differences',
    'extract_distinctive_features',
    'interpret_feature_differences',
    'causal_feature_validation',
    'name_features'
] 