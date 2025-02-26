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

## SOTA Interpretability Prompts

The pipeline includes a carefully designed set of prompts specifically optimized for model interpretability. These prompts are structured to elicit meaningful feature differences between models:

### Key Prompt Categories

- **Cognitive Capabilities**: Prompts testing various reasoning capabilities including recursive, Bayesian, and paradoxical reasoning
- **Mathematical Reasoning**: Complex problems that test mathematical understanding and multi-step solution processes
- **Causal Reasoning**: Scenarios that require understanding causation vs. correlation and experimental design
- **Emergent Abilities**: Tasks that typically emerge at specific capability thresholds in language models
- **Knowledge Integration**: Problems requiring synthesis across multiple domains
- **Conceptual Abstraction**: Deep philosophical questions requiring abstract conceptualization
- **Multi-Step Planning**: Complex planning problems requiring sequential thinking
- **Hallucination Triggers**: Prompts designed to potentially trigger model confabulation
- **Ethical Alignment**: Moral dilemmas that test value alignment
- **Code Understanding**: Programming challenges that test code comprehension
- **Multilingual Capabilities**: Cross-language translation and cultural understanding tests
- **Counterfactual Reasoning**: "What if" scenarios requiring imagination of alternative realities
- **Context Awareness**: Prompts testing pragmatic understanding of language
- **Adversarial Prompts**: Challenging prompts designed to test model boundaries

### Design Principles

These prompts were designed following several key principles for effective interpretability:

1. **Capability Differentiation**: Targets specific capabilities that often differ between models
2. **Cognitive Dimension Coverage**: Ensures broad coverage of cognitive dimensions
3. **Difficulty Gradients**: Includes problems at varying difficulty levels to identify capability thresholds
4. **Specialized Feature Targeting**: Targets features known to emerge at different training stages
5. **Minimal Ambiguity**: Crafted to have clear, unambiguous correct responses
6. **Failure Mode Exploration**: Includes prompts that test common model failure modes

These SOTA prompts help identify more nuanced and meaningful feature differences between models compared to generic prompt sets.

## Focused Research Approach: Reasoning Capabilities

Our interpretability analysis has been specifically designed to address a focused research question:

**How do reasoning capabilities differ between language models at the feature level, and what specific reasoning patterns show the most significant differences?**

Rather than attempting to cover every possible model capability, we've designed a targeted set of prompts focused exclusively on different facets of reasoning. This approach allows for more meaningful interpretation of feature differences and clearer insights into how reasoning capabilities are represented within model architectures.

### Reasoning-Specific Prompt Categories

Our prompts are organized into the following specialized reasoning categories:

1. **Step-by-Step Reasoning**: Prompts that require explicit step-by-step problem solving, typically with mathematical or logical problems.
  
2. **Chain of Thought**: Prompts designed to elicit multi-step thinking processes with clear intermediate reasoning.

3. **Formal Logic**: Problems requiring formal logical analysis, including syllogisms, conditional statements, and logical equivalence.

4. **Causal Reasoning**: Scenarios requiring analysis of cause-effect relationships and identification of confounding variables.

5. **Probabilistic Reasoning**: Problems involving probability calculations, Bayesian reasoning, and statistical analysis.

6. **Counterfactual Reasoning**: Prompts asking for analysis of hypothetical scenarios and their implications.

7. **Analogical Reasoning**: Tasks requiring comparison between different domains and assessment of where analogies hold or break down.

8. **Abductive Reasoning**: Scenarios requiring generation of multiple hypotheses to explain observations and assessment of their plausibility.

9. **Adversarial Reasoning**: Problems designed with common cognitive traps that require careful analysis to avoid errors.

10. **Constraint Satisfaction**: Complex problems with multiple interacting constraints that must be satisfied simultaneously.

### Why This Approach?

This focused approach offers several advantages:

1. **Targeted Feature Detection**: By concentrating on reasoning capabilities, we can more effectively identify specific features that differentiate models in this crucial domain.

2. **Interpretable Differences**: Reasoning processes tend to produce more interpretable activation patterns that can be more clearly linked to specific cognitive processes.

3. **Practical Applications**: Understanding reasoning differences has direct applications in determining which models are better suited for tasks requiring specific types of reasoning.

4. **Meaningful Contrastive Examples**: The reasoning-focused prompts enable generation of more meaningful contrastive examples when evaluating identified features.

This specialized approach represents a deliberate shift away from broad but shallow capability assessment toward deep, focused analysis of specific capability differences that matter most for downstream applications.

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
   - Feature distribution plots
   - Anthropic-style visualizations with feature callout boxes
   - Feature heatmaps
   - Categorized features analysis (in JSON format)
   - Feature alignment visualizations
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
    prompt_categories=prompt_categories,
    output_dir="feature_interpretation",
    device="cuda",
    quantization="fp16"
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