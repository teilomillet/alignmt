"""
Decoder Weight Analysis Package.

This package provides functionality for analyzing decoder weights
between base and target models, identifying model-specific features,
and clustering features based on their activation patterns.
"""

from .norms import extract_feature_decoder_norms
from .activity import identify_active_features
from .clustering import cluster_features
from .responses import compare_feature_responses
from .analysis import generate_comprehensive_analysis, categorize_features_by_norm

__all__ = [
    'extract_feature_decoder_norms',
    'identify_active_features',
    'cluster_features',
    'compare_feature_responses',
    'generate_comprehensive_analysis',
    'categorize_features_by_norm'
] 