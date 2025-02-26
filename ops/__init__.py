"""
Operation modules for model analysis.

This package provides various tools for analyzing and
comparing language models.
"""

# Try to import from feature_interpreter
try:
    from .feature_interpreter import (
        PipelineConfig,
        run_feature_interpretation_pipeline
    )
    _has_feature_interpreter = True
except ImportError:
    _has_feature_interpreter = False

# Try to import from integrated
try:
    from .integrated import (
        IntegratedPipelineConfig,
        run_integrated_pipeline,
        run_crosscoder_analysis
    )
    _has_integrated = True
except ImportError:
    _has_integrated = False

__all__ = []

# Add feature interpreter classes to __all__ if available
if _has_feature_interpreter:
    __all__.extend([
        "PipelineConfig", 
        "run_feature_interpretation_pipeline"
    ])

# Add integrated pipeline classes to __all__ if available
if _has_integrated:
    __all__.extend([
        "IntegratedPipelineConfig", 
        "run_integrated_pipeline",
        "run_crosscoder_analysis"
    ]) 