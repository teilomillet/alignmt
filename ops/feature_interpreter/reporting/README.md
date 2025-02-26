# Reporting Module

This module provides tools for generating comprehensive reports on feature interpretation findings, allowing researchers to document and share their results effectively.

## Key Components

### report_generator.py
Core report generation functionality:
- `generate_report`: Generates a comprehensive report based on feature interpretation results

### markdown_report.py
Markdown-specific report generation:
- `generate_markdown_report`: Creates detailed reports in Markdown format, suitable for GitHub or documentation sites

## Usage

The reporting module is typically used as the final step in the feature interpretation workflow, after all analysis has been completed. The process involves:

1. Collecting all results from feature extraction, naming, capability testing, and visualization
2. Organizing these results into a coherent narrative
3. Generating a structured report in the desired format

The resulting reports integrate numerical data, visualizations, and interpretations to provide a comprehensive view of the feature-level differences between models. These reports serve as documentation for research findings and can be shared with stakeholders to communicate insights about model behavior. 