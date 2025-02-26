# Activations Module

This module provides functionality for extracting activations from language models, which is a fundamental step in feature interpretation. The activations extracted can be used for comparative analysis between different models or for in-depth analysis of individual models.

## Key Components

### extraction.py
Contains functions for extracting activations from language models:
- `extract_activations`: Extracts activations from a specified model for a set of prompts
- `extract_activations_for_comparison`: Extracts and processes activations from two models (base and target) for comparison

### hooks.py
Provides functionality to register hooks on model layers to capture activations during forward passes:
- `register_activation_hooks`: Registers hooks to capture activations from specified layers

## Usage

These modules form the foundation of the feature interpretation pipeline by providing the raw activation data needed for all subsequent analysis steps. The typical workflow involves:

1. Loading models
2. Registering activation hooks on specific layers
3. Running inference with chosen prompts
4. Collecting and processing the resulting activations

The resulting activation data can then be passed to other modules for naming, visualization, or decoder analysis. 