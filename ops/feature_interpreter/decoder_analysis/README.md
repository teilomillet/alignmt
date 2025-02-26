# Decoder Analysis Module

This module provides functionality for analyzing decoder weights and their relationships to model features. It helps understand how identified features influence model outputs through decoder connections.

## Key Components

### norms.py
Tools for analyzing decoder weight norms:
- `extract_feature_decoder_norms`: Extracts norms of decoder weights for feature analysis

### activity.py
Functions for identifying active features:
- `identify_active_features`: Identifies which features are most active in different contexts

### clustering.py
Clustering tools for grouping similar features:
- `cluster_features`: Clusters features based on their activation patterns or decoder weights

### responses.py
Tools for analyzing feature responses:
- `compare_feature_responses`: Compares how features respond to different inputs across models

### analysis.py
Comprehensive analysis functions:
- `generate_comprehensive_analysis`: Performs a complete decoder analysis
- `categorize_features_by_norm`: Categorizes features based on their decoder weight norms

## Usage

The decoder analysis module helps researchers understand how internal features connect to model outputs. The typical workflow involves:

1. Extracting feature-to-decoder weight connections
2. Analyzing the norms and patterns in these weights
3. Clustering features based on similar behaviors
4. Comparing feature responses across different inputs or models

This analysis provides insights into how specific features influence a model's final output and can help researchers understand the mechanisms behind model capabilities. 