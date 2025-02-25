# Feature-Level Model Difference Interpretation

This package provides tools for interpreting differences between language models at the feature level, going beyond simple similarity scores to identify specific capabilities that differ between models.

## Overview

The current crosscoder analysis gives us similarity scores between model layers, but these numbers don't tell us **what** actually changed between models. This feature interpretation pipeline helps us:

1. Name specific features that differ between models
2. Identify which capabilities were added, removed or modified
3. Locate where in the model these changes occurred
4. Provide concrete examples of how these changes manifest in model behavior

Compare these two outputs:

**Current output:** "Layer 18 has a similarity score of 0.43"  
**Desired output:** "Layer 18 shows enhanced step-by-step reasoning features in the target model, with the base model relying more on unconstrained reasoning approaches"

## Components

The feature interpretation pipeline consists of four main components:

1. **Activation Extraction**: Extracts activations from both models for a set of prompts
2. **Feature Naming**: Associates activations with interpretable feature names
3. **Feature Visualization**: Creates visualizations of feature differences
4. **Report Generation**: Produces comprehensive comparison reports

## Installation

This package is part of the `alignmt` project. No additional installation is required.

## Usage

### Command-line Interface

The simplest way to use the feature interpretation pipeline is through the command-line interface:

```bash
python -m ops.feature_interpreter.main \
    --base-model "Qwen/Qwen2-1.5B" \
    --target-model "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" \
    --output-dir "feature_interpretation" \
    --device "cuda" \
    --quantization "fp16"
```

This will run the complete pipeline and save the results to the specified output directory.

### Python API

You can also use the feature interpretation pipeline from Python:

```python
from ops.feature_interpreter import run_feature_interpretation_pipeline

run_feature_interpretation_pipeline(
    base_model="Qwen/Qwen2-1.5B",
    target_model="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    output_dir="feature_interpretation",
    device="cuda",
    quantization="fp16"
)
```

### Options

The feature interpretation pipeline supports the following options:

- `--base-model`: Name of the base model
- `--target-model`: Name of the target model
- `--crosscoder-file`: Optional path to crosscoder analysis file
- `--output-dir`: Directory to save results
- `--device`: Device to use
- `--cache-dir`: Optional cache directory
- `--quantization`: Quantization method to use ("fp32", "fp16", or "int8")
- `--skip-activations`: Skip activation extraction
- `--skip-naming`: Skip feature naming
- `--skip-visualization`: Skip visualization creation
- `--skip-report`: Skip report generation
- `--report-format`: Report format ("markdown", "html", or "both")
- `--feature-threshold`: Threshold for considering a difference significant

## Output

The feature interpretation pipeline produces the following outputs:

1. **Activations**: Saved in `{output_dir}/activations/`
2. **Feature Interpretations**: Saved in `{output_dir}/features/`
3. **Visualizations**: Saved in `{output_dir}/visualizations/`
4. **Reports**: Saved in `{output_dir}/reports/`

### Example Visualization

The pipeline creates Anthropic-style visualizations with feature callout boxes:

![Feature Distribution](https://i.imgur.com/PlPkbM2.png)

### Example Report

The pipeline generates comprehensive reports in Markdown and HTML formats, including:

- Summary of model differences
- Feature overview
- Layer-by-layer analysis
- Visualizations
- Conclusion and overall assessment

## Advanced Usage

### Running Individual Components

You can run individual components of the pipeline separately:

```python
from ops.feature_interpreter import extract_activations_for_comparison, name_features, create_visualizations, generate_report

# Extract activations
activation_data = extract_activations_for_comparison(
    base_model="Qwen/Qwen2-1.5B",
    target_model="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    prompts=prompt_categories,
    output_dir="activations"
)

# Name features
feature_data = name_features(
    activation_data,
    crosscoder_data=None,
    output_dir="features",
    threshold=0.3
)

# Create visualizations
create_visualizations(feature_data, output_dir="visualizations")

# Generate report
generate_report(
    feature_data,
    crosscoder_data=None,
    output_dir="reports",
    report_format="both"
)
```

### Customizing Prompt Categories

You can customize the prompt categories used for activation extraction:

```python
prompt_categories = {
    "reasoning": [
        "Solve the equation: 2x + 3 = 7. Show all your steps.",
        "A train travels at 60 mph. How far will it travel in 2.5 hours? Explain your reasoning."
    ],
    "instruction_following": [
        "Write a short poem about artificial intelligence.",
        "List five benefits of regular exercise."
    ],
    # Add more categories as needed
}

run_feature_interpretation_pipeline(
    base_model="Qwen/Qwen2-1.5B",
    target_model="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    prompt_categories=prompt_categories
)
```

## Requirements

- Python 3.8+
- PyTorch 1.10+
- Transformers 4.20+
- Matplotlib 3.5+
- NumPy 1.20+
- scikit-learn 1.0+
- (Optional) Python-Markdown for HTML report generation

## License

This package is part of the `alignmt` project and is subject to the same license. 