"""
Pipeline module for feature-level model difference interpretation.

This module provides the core pipeline functionality for interpreting 
differences between language models at the feature level.
"""

from .config import PipelineConfig
from .runner import run_feature_interpretation_pipeline

__all__ = [
    "PipelineConfig",
    "run_feature_interpretation_pipeline",
] 