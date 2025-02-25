"""
Feature-level model difference interpretation package.

This package provides tools for interpreting differences between language models
at the feature level, going beyond simple similarity scores to identify specific
capabilities that differ between models.
"""

from .extract_activations import extract_activations, extract_activations_for_comparison
from .feature_naming import name_features, compute_activation_differences, interpret_feature_differences, causal_feature_validation
from .feature_visualization import create_visualizations, create_anthropic_style_visualization
from .generate_report import generate_report, generate_markdown_report, generate_html_report
from .main import run_feature_interpretation_pipeline
from .capability_testing import generate_contrastive_examples, evaluate_feature_capability

__all__ = [
    'extract_activations',
    'extract_activations_for_comparison',
    'name_features',
    'compute_activation_differences',
    'interpret_feature_differences',
    'causal_feature_validation',
    'create_visualizations',
    'create_anthropic_style_visualization',
    'generate_report',
    'generate_markdown_report',
    'generate_html_report',
    'run_feature_interpretation_pipeline',
    'generate_contrastive_examples',
    'evaluate_feature_capability'
] 