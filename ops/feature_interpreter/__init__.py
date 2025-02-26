"""
Feature Interpreter package.

This package provides tools for interpreting differences between
language models at the feature level.
"""

# Import from the new activations package
from .activations import extract_activations, extract_activations_for_comparison

# Import from the naming package
from .naming import (
    name_features, 
    compute_activation_differences, 
    interpret_feature_differences, 
    causal_feature_validation, 
    extract_distinctive_features
)

# Import from the visualization package
from .visualization import create_feature_distribution_plot, create_anthropic_style_visualization

# Import from the reporting package
from .reporting.report_generator import generate_report
from .reporting.markdown_report import generate_markdown_report

# Import from the capability package
from .capability.evaluation import evaluate_feature_capability
from .capability.metrics import calculate_human_experience_score
from .capability.examples import generate_contrastive_examples

# Import from the decoder_analysis package
from .decoder_analysis import (
    extract_feature_decoder_norms,
    identify_active_features,
    cluster_features,
    compare_feature_responses,
    generate_comprehensive_analysis
)

# Import from the pipeline package - using a try/except to handle backward compatibility
try:
    from .pipeline import PipelineConfig, run_feature_interpretation_pipeline
    _has_pipeline = True
except ImportError:
    _has_pipeline = False

__all__ = [
    # Activation extraction
    "extract_activations",
    "extract_activations_for_comparison",
    
    # Feature naming
    "name_features",
    "compute_activation_differences",
    "interpret_feature_differences",
    "causal_feature_validation",
    "extract_distinctive_features",
    
    # Visualization
    "create_feature_distribution_plot",
    "create_anthropic_style_visualization",
    
    # Reporting
    "generate_report",
    "generate_markdown_report",
    
    # Capability
    "generate_contrastive_examples",
    "evaluate_feature_capability",
    "calculate_human_experience_score",
    
    # Decoder analysis functions
    "extract_feature_decoder_norms",
    "identify_active_features",
    "cluster_features",
    "compare_feature_responses",
    "generate_comprehensive_analysis",
]

# Add pipeline classes to __all__ if available
if _has_pipeline:
    __all__.extend(["PipelineConfig", "run_feature_interpretation_pipeline"]) 