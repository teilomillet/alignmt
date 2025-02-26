# Pipeline Module

This module provides the core pipeline functionality for orchestrating the entire feature interpretation process. It integrates all other modules into a streamlined workflow for interpreting differences between language models at the feature level.

## Key Components

### config.py
Configuration management for the pipeline:
- `PipelineConfig`: Configuration class for setting pipeline parameters

### runner.py
Core pipeline execution:
- `run_feature_interpretation_pipeline`: Main function to run the complete interpretation pipeline

### analysis_steps.py
Individual analysis steps that make up the pipeline:
- Contains functions for each step in the feature interpretation process

### feature_analysis.py
Feature-specific analysis tools:
- Functions for analyzing individual features and their properties

### layer_analysis.py
Layer-specific analysis tools:
- Functions for analyzing model layers and their characteristics

### cli.py
Command-line interface for the pipeline:
- Provides a CLI for running the pipeline from the command line

### utils.py
Utility functions used across the pipeline:
- Helper functions for file operations, data manipulation, etc.

## Usage

The pipeline module serves as the entry point for most feature interpretation workflows. To use it:

1. Create a PipelineConfig object with desired parameters
2. Call run_feature_interpretation_pipeline with this config
3. The pipeline will automatically execute all steps in the proper sequence

This provides a convenient way to run comprehensive feature interpretation analyses without having to manually orchestrate each component of the process. 