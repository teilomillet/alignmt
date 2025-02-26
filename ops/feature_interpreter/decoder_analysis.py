"""
Decoder Weight Analysis Module.

DEPRECATED: This module is deprecated and will be removed in a future release.
            Please use the new modular implementation in the 
            `ops.feature_interpreter.decoder_analysis` package instead:

            from ops.feature_interpreter.decoder_analysis import (
                extract_feature_decoder_norms,
                identify_active_features,
                cluster_features,
                compare_feature_responses,
                generate_comprehensive_analysis
            )

This module provides functionality for analyzing decoder weights 
between base and target models, identifying model-specific features,
and clustering features based on their activation patterns.
"""

import warnings
import functools
import numpy as np
import matplotlib.pyplot as plt
import logging
import os
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

# Configure logging
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Deprecation warning
warnings.warn(
    "The decoder_analysis module is deprecated and will be removed in a future release. "
    "Please use the new modular implementation in the ops.feature_interpreter.decoder_analysis package instead.",
    DeprecationWarning,
    stacklevel=2
)

def _deprecation_warning(func):
    """Decorator to show deprecation warning when function is called."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        warnings.warn(
            f"Function {func.__name__} is deprecated and will be removed in a future release. "
            f"Please use ops.feature_interpreter.decoder_analysis.{func.__name__} instead.",
            DeprecationWarning,
            stacklevel=2
        )
        return func(*args, **kwargs)
    return wrapper

# Import functions from visualization to maintain compatibility
from .visualization.analysis import categorize_features_by_norm, calculate_feature_alignment

# Import the new implementations for backward compatibility
from .decoder_analysis.norms import extract_feature_decoder_norms as _extract_feature_decoder_norms
from .decoder_analysis.activity import identify_active_features as _identify_active_features
from .decoder_analysis.clustering import cluster_features as _cluster_features
from .decoder_analysis.responses import compare_feature_responses as _compare_feature_responses
from .decoder_analysis.analysis import generate_comprehensive_analysis as _generate_comprehensive_analysis

# Export the functions with deprecation warnings
@_deprecation_warning
def extract_feature_decoder_norms(*args, **kwargs):
    return _extract_feature_decoder_norms(*args, **kwargs)

@_deprecation_warning
def identify_active_features(*args, **kwargs):
    return _identify_active_features(*args, **kwargs)

@_deprecation_warning
def cluster_features(*args, **kwargs):
    return _cluster_features(*args, **kwargs)

@_deprecation_warning
def compare_feature_responses(*args, **kwargs):
    return _compare_feature_responses(*args, **kwargs)

@_deprecation_warning
def generate_comprehensive_analysis(*args, **kwargs):
    return _generate_comprehensive_analysis(*args, **kwargs) 