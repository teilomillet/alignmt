# Capability Testing Module

This module provides tools for generating contrastive examples and evaluating model capabilities based on feature interpretations. It helps assess how identified features contribute to specific model capabilities.

## Key Components

### examples.py
Functions for generating examples to test specific capabilities:
- `generate_contrastive_examples`: Creates pairs of examples that highlight specific feature differences

### evaluation.py
Tools for evaluating model performance on specific capabilities:
- `generate_response`: Generates model responses for evaluation
- `evaluate_feature_capability`: Evaluates how specific features contribute to model capabilities

### metrics.py
Metrics for quantifying model capabilities:
- `calculate_human_experience_score`: Calculates a score representing how a feature impacts human experience of the model

## Usage

The capability module is typically used after features have been identified and named. The workflow involves:

1. Generating contrastive examples that isolate specific features or capabilities
2. Evaluating model performance on these examples
3. Calculating metrics to quantify the impact of features on model capabilities

This process helps researchers understand the relationship between internal model features and observable capabilities, providing insights into what makes models perform differently on specific tasks. 