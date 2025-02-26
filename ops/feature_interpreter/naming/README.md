# Feature Naming Module

This module provides functionality for naming and interpreting features based on activation differences between models. It helps in understanding what specific features or capabilities differ between models.

## Key Components

### differences.py
Functions for computing and analyzing activation differences between models:
- `compute_activation_differences`: Calculates differences in activations between models
- `analyze_output_differences`: Analyzes differences in model outputs

### extraction.py
Tools for extracting distinctive features from activation data:
- `extract_distinctive_features`: Identifies and extracts features that show significant differences between models

### interpretation.py
Functions for interpreting the meaning of feature differences:
- `interpret_feature_differences`: Provides interpretations of what feature differences represent

### validation.py
Validation tools to verify feature interpretations:
- `causal_feature_validation`: Validates the causal relationship between identified features and model behaviors

### pipeline.py
Orchestrates the feature naming workflow:
- `name_features`: Runs the complete feature naming pipeline

## Usage

The naming module is typically used after extracting activations from models. The workflow involves:

1. Computing activation differences between models
2. Extracting the most distinctive features
3. Interpreting what these differences represent
4. Validating the interpretations through causal analysis

The results help researchers understand what capabilities or behaviors differ between language models at a feature level. 