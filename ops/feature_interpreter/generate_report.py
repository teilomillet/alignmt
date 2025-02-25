"""
Report generation module.

This module generates comprehensive comparison reports based on
feature-level interpretations of model differences.
"""

import logging
import os
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

# Configure logging
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

def generate_markdown_report(
    base_model: str,
    target_model: str,
    interpreted_features: Dict,
    layer_similarities: Dict,
    output_path: str
) -> None:
    """
    Generate a Markdown report with feature interpretation results.
    
    Args:
        base_model: Name of the base model
        target_model: Name of the target model
        interpreted_features: Dictionary with interpreted features
        layer_similarities: Dictionary with layer similarities
        output_path: Path to save the report
    """
    # Extract features
    base_features = interpreted_features.get("base_model_specific_features", [])
    target_features = interpreted_features.get("target_model_specific_features", [])
    
    # Sort features by confidence
    base_features.sort(key=lambda x: x.get("confidence", 0), reverse=True)
    target_features.sort(key=lambda x: x.get("confidence", 0), reverse=True)
    
    # Calculate average layer similarity
    avg_similarity = np.mean(list(layer_similarities.values())) if layer_similarities else 0.0
    
    # Find layers with lowest similarity (most different)
    sorted_similarities = sorted(layer_similarities.items(), key=lambda x: x[1])
    most_different_layers = sorted_similarities[:3] if len(sorted_similarities) >= 3 else sorted_similarities
    
    # Generate report
    report = f"""# Feature Interpretation Report

## Model Comparison

- **Base Model**: {base_model}
- **Target Model**: {target_model}
- **Average Layer Similarity**: {avg_similarity:.4f}

## Layer Similarity Analysis

The average similarity between corresponding layers in the two models is {avg_similarity:.4f}. 
A similarity of 1.0 would indicate identical layers, while 0.0 would indicate completely different layers.

### Most Different Layers

The following layers show the most significant differences between the models:

"""
    
    for layer, similarity in most_different_layers:
        report += f"- **{layer}**: {similarity:.4f} similarity\n"
    
    report += """
## Feature Analysis

This section presents the distinctive features identified in each model.

"""
    
    # Add base model features
    report += f"### Base Model Features ({base_model})\n\n"
    
    if base_features:
        for i, feature in enumerate(base_features):
            name = feature.get("name", "Unknown feature")
            description = feature.get("description", "No description available")
            confidence = feature.get("confidence", 0.0)
            layer = feature.get("layer", "unknown")
            examples = feature.get("examples", [])
            
            report += f"#### {i+1}. {name}\n\n"
            report += f"- **Description**: {description}\n"
            report += f"- **Confidence**: {confidence:.2f}\n"
            report += f"- **Layer**: {layer}\n"
            
            if examples:
                report += f"- **Example Prompt**: {examples[0]}\n"
            
            report += "\n"
    else:
        report += "No distinctive features identified for the base model.\n\n"
    
    # Add target model features
    report += f"### Target Model Features ({target_model})\n\n"
    report += f"### Distinctive Base Model Features ({base_model})\n\n"
    
    if base_features:
        report += f"**These capabilities appear to be stronger in the base model or potentially removed/weakened in the target model:**\n\n"
        for i, feature in enumerate(base_features):
            name = feature.get("name", "Unknown feature")
            description = feature.get("description", "No description available")
            confidence = feature.get("confidence", 0.0)
            layer = feature.get("layer", "unknown")
            examples = feature.get("examples", [])
            
            report += f"#### {i+1}. {name}\n\n"
            report += f"- **Description**: {description}\n"
            report += f"- **Confidence**: {confidence:.2f}\n"
            report += f"- **Layer**: {layer}\n"
            report += f"- **Impact**: This capability appears to be weakened or modified in the target model\n"
            
            if examples:
                report += f"- **Example Prompt**: {examples[0]}\n"
            
            report += "\n"
    else:
        report += "No distinctive features identified in the base model that are weakened or removed in the target model.\n\n"
    
    # Add target model features
    report += f"### Distinctive Target Model Features ({target_model})\n\n"
    
    if target_features:
        report += f"**These capabilities appear to be stronger in the target model or newly added compared to the base model:**\n\n"
        for i, feature in enumerate(target_features):
            name = feature.get("name", "Unknown feature")
            description = feature.get("description", "No description available")
            confidence = feature.get("confidence", 0.0)
            layer = feature.get("layer", "unknown")
            examples = feature.get("examples", [])
            
            report += f"#### {i+1}. {name}\n\n"
            report += f"- **Description**: {description}\n"
            report += f"- **Confidence**: {confidence:.2f}\n"
            report += f"- **Layer**: {layer}\n"
            
            if examples:
                report += f"- **Example Prompt**: {examples[0]}\n"
            
            report += "\n"
    else:
        report += "No distinctive features identified for the target model.\n\n"
    
    # Add interpretation summary
    report += """## Interpretation Summary

The feature analysis reveals the following key differences between the models:

"""
    
    # Summarize base model strengths
    if base_features:
        report += "### Base Model Strengths\n\n"
        for feature in base_features[:3]:  # Top 3 features
            report += f"- **{feature.get('name', 'Unknown')}**: {feature.get('description', 'No description')}\n"
        report += "\n"
    
    # Summarize target model strengths
    if target_features:
        report += "### Target Model Strengths\n\n"
        for feature in target_features[:3]:  # Top 3 features
            report += f"- **{feature.get('name', 'Unknown')}**: {feature.get('description', 'No description')}\n"
        report += "\n"
    
    # Create a comparative summary section
    report += "### Comparative Analysis\n\n"
    
    # Group features by layer for more organized analysis
    layer_to_base_features = {}
    layer_to_target_features = {}
    
    for feature in base_features:
        layer = feature.get("layer", "unknown")
        if layer not in layer_to_base_features:
            layer_to_base_features[layer] = []
        layer_to_base_features[layer].append(feature)
    
    for feature in target_features:
        layer = feature.get("layer", "unknown")
        if layer not in layer_to_target_features:
            layer_to_target_features[layer] = []
        layer_to_target_features[layer].append(feature)
    
    # Find layers with both added and removed features
    common_layers = set(layer_to_base_features.keys()).intersection(set(layer_to_target_features.keys()))
    
    if common_layers:
        report += "#### Feature Transformations by Layer\n\n"
        report += "These layers show clear transformation of capabilities from the base model to the target model:\n\n"
        
        for layer in sorted(common_layers):
            similarity = layer_similarities.get(layer, 0.0)
            report += f"**Layer {layer}** (Similarity: {similarity:.2f}):\n"
            report += "* Capabilities weakened or removed:\n"
            for feature in layer_to_base_features[layer]:
                report += f"  - {feature.get('name', 'Unknown')}\n"
            report += "* Capabilities strengthened or added:\n"
            for feature in layer_to_target_features[layer]:
                report += f"  - {feature.get('name', 'Unknown')}\n"
            report += "\n"
    
    # Summarize removed features
    if base_features:
        report += "#### Features Weakened or Removed in Target Model\n\n"
        for feature in base_features[:5]:  # Top 5 features
            report += f"- **{feature.get('name', 'Unknown')}**: {feature.get('description', 'No description')} (Layer: {feature.get('layer', 'unknown')})\n"
        report += "\n"
    
    # Summarize added features
    if target_features:
        report += "#### Features Strengthened or Added in Target Model\n\n"
        for feature in target_features[:5]:  # Top 5 features
            report += f"- **{feature.get('name', 'Unknown')}**: {feature.get('description', 'No description')} (Layer: {feature.get('layer', 'unknown')})\n"
        report += "\n"
    
    # Overall model evolution summary
    report += "### Model Evolution Summary\n\n"
    
    # Count features by category
    base_categories = {}
    target_categories = {}
    
    for feature in base_features:
        name = feature.get('name', 'unknown')
        category = name.split()[0] if ' ' in name else name
        base_categories[category] = base_categories.get(category, 0) + 1
    
    for feature in target_features:
        name = feature.get('name', 'unknown')
        category = name.split()[0] if ' ' in name else name
        target_categories[category] = target_categories.get(category, 0) + 1
    
    # Generate summary text based on category shifts
    report += f"The target model ({target_model}) appears to be a modification of the base model ({base_model}) with the following general changes:\n\n"
    
    # Identify key shifts
    if base_features and target_features:
        shifts = []
        
        # Look for categories that decreased
        for category, count in base_categories.items():
            if category in target_categories and target_categories[category] < count:
                shifts.append(f"Reduced emphasis on {category}")
            elif category not in target_categories:
                shifts.append(f"Removed or significantly weakened {category}")
        
        # Look for categories that increased
        for category, count in target_categories.items():
            if category in base_categories and target_categories[category] > base_categories[category]:
                shifts.append(f"Increased emphasis on {category}")
            elif category not in base_categories:
                shifts.append(f"Added new {category}")
        
        if shifts:
            for shift in shifts[:5]:  # Limit to top 5 shifts
                report += f"- {shift}\n"
        else:
            report += "- Modified capabilities while maintaining similar overall behavior\n"
            report += f"- Average layer similarity of {avg_similarity:.2f} suggests significant but targeted changes\n"
    else:
        report += "- The analysis did not detect enough distinctive features to characterize the changes"
    
    # Add conclusion
    report += """## Conclusion

This report provides a feature-level interpretation of the differences between the models. 
The identified features highlight the distinctive capabilities of each model and can guide further investigation.
"""
    
    # Check for causal validation results
    causal_validation = interpreted_features.get("causal_validation", {})
    if causal_validation:
        report += f"""
## Causal Feature Validation

The following analysis shows the causal importance of identified features, measured through activation patching experiments.
This helps determine whether the features are actually responsible for the observed behavioral differences.

### Base Model Feature Impact

These features from the base model were validated by patching experiments:

| Feature | Layer | Impact Score | Interpretation |
|---------|-------|--------------|----------------|
"""
        # Add base model causal validation results
        base_feature_validation = causal_validation.get("base_model_features", {})
        for feature_name, data in base_feature_validation.items():
            impact = data.get("average_impact", 0.0)
            layer = data.get("layer", "unknown")
            
            # Interpret the impact score
            if impact > 0.7:
                interpretation = "Very high causal impact"
            elif impact > 0.5:
                interpretation = "High causal impact"
            elif impact > 0.3:
                interpretation = "Moderate causal impact"
            elif impact > 0.1:
                interpretation = "Low causal impact"
            else:
                interpretation = "Minimal causal impact"
                
            report += f"| {feature_name} | {layer} | {impact:.2f} | {interpretation} |\n"
            
        report += f"""

### Target Model Feature Impact

These features from the target model were validated by patching experiments:

| Feature | Layer | Impact Score | Interpretation |
|---------|-------|--------------|----------------|
"""
        # Add target model causal validation results
        target_feature_validation = causal_validation.get("target_model_features", {})
        for feature_name, data in target_feature_validation.items():
            impact = data.get("average_impact", 0.0)
            layer = data.get("layer", "unknown")
            
            # Interpret the impact score
            if impact > 0.7:
                interpretation = "Very high causal impact"
            elif impact > 0.5:
                interpretation = "High causal impact"
            elif impact > 0.3:
                interpretation = "Moderate causal impact"
            elif impact > 0.1:
                interpretation = "Low causal impact"
            else:
                interpretation = "Minimal causal impact"
                
            report += f"| {feature_name} | {layer} | {impact:.2f} | {interpretation} |\n"
            
        report += f"""

The impact score represents how much model behavior changes when a feature is patched from one model to another.
Higher scores indicate that the feature is causally important for the model's behavior.
"""
    
    # Check for capability testing results
    capability_testing = interpreted_features.get("capability_testing", {})
    if capability_testing:
        report += f"""
## Capability Testing Results

The identified features were tested using contrastive evaluation to verify whether they translate to measurable capability differences.
Below are the results of systematically testing each feature with specific capability-targeted examples.

### Base Model Capability Tests

The following base model features were tested with capability-specific examples:

| Feature | % Validation | Status | Test Examples |
|---------|--------------|--------|--------------|
"""
        # Add base feature capability testing results
        base_feature_tests = capability_testing.get("base_feature_evaluations", [])
        for test in base_feature_tests:
            feature_name = test.get("feature_name", "Unknown")
            percent = test.get("percent_supported", 0)
            is_validated = test.get("is_validated", False)
            status = "✅ Validated" if is_validated else "❌ Not Validated"
            
            # Get example descriptions
            examples = test.get("examples", [])
            example_descriptions = [ex.get("description", "N/A") for ex in examples]
            example_str = ", ".join(example_descriptions)
            
            report += f"| {feature_name} | {percent:.0f}% | {status} | {example_str} |\n"
            
        report += f"""

### Target Model Capability Tests

The following target model features were tested with capability-specific examples:

| Feature | % Validation | Status | Test Examples |
|---------|--------------|--------|--------------|
"""
        # Add target feature capability testing results
        target_feature_tests = capability_testing.get("target_feature_evaluations", [])
        for test in target_feature_tests:
            feature_name = test.get("feature_name", "Unknown")
            percent = test.get("percent_supported", 0)
            is_validated = test.get("is_validated", False)
            status = "✅ Validated" if is_validated else "❌ Not Validated"
            
            # Get example descriptions
            examples = test.get("examples", [])
            example_descriptions = [ex.get("description", "N/A") for ex in examples]
            example_str = ", ".join(example_descriptions)
            
            report += f"| {feature_name} | {percent:.0f}% | {status} | {example_str} |\n"
        
        # Add detailed human experience analysis if available
        human_exp_features = []
        for test in base_feature_tests + target_feature_tests:
            if "human_experience" in test.get("feature_name", "").lower():
                human_exp_features.append(test)
        
        if human_exp_features:
            report += f"""

### Detailed Human Experience Analysis

A deeper analysis of human experience capabilities using linguistic markers:

"""
            for test in human_exp_features:
                feature_name = test.get("feature_name", "Unknown")
                model_type = "Base" if test in base_feature_tests else "Target"
                
                report += f"#### {model_type} Model: {feature_name}\n\n"
                report += "| Prompt | First-Person | Emotional Content | Sensory Details | Subjectivity |\n"
                report += "|--------|-------------|------------------|----------------|-------------|\n"
                
                for example in test.get("examples", []):
                    if model_type == "Base" and "base_positive_humanness" in example:
                        # Calculate component scores based on weights used in calculate_human_experience_score
                        score = example.get("base_positive_humanness", 0)
                        first_person = min(score / 0.3, 1.0) if score > 0 else 0
                        emotional = min(score / 0.3, 1.0) if score > 0 else 0
                        sensory = min(score / 0.25, 1.0) if score > 0 else 0
                        subjective = min(score / 0.15, 1.0) if score > 0 else 0
                        
                        prompt_short = example.get("positive_prompt", "")[:50] + "..."
                        report += f"| {prompt_short} | {'★' * int(first_person * 5)} | {'★' * int(emotional * 5)} | {'★' * int(sensory * 5)} | {'★' * int(subjective * 5)} |\n"
                    
                    elif model_type == "Target" and "target_positive_humanness" in example:
                        # Calculate component scores based on weights used in calculate_human_experience_score
                        score = example.get("target_positive_humanness", 0)
                        first_person = min(score / 0.3, 1.0) if score > 0 else 0
                        emotional = min(score / 0.3, 1.0) if score > 0 else 0
                        sensory = min(score / 0.25, 1.0) if score > 0 else 0
                        subjective = min(score / 0.15, 1.0) if score > 0 else 0
                        
                        prompt_short = example.get("positive_prompt", "")[:50] + "..."
                        report += f"| {prompt_short} | {'★' * int(first_person * 5)} | {'★' * int(emotional * 5)} | {'★' * int(sensory * 5)} | {'★' * int(subjective * 5)} |\n"
                
                # Add explanation of metrics
                report += f"""
*Metrics explanation:*
- **First-Person**: Use of first-person pronouns (I, me, my) indicating subjective experience
- **Emotional Content**: References to feelings and emotional states
- **Sensory Details**: Descriptions of physical sensations and perceptual experiences
- **Subjectivity**: Expressions of personal opinions and perspectives

"""
    
    # Write report to file
    with open(output_path, "w") as f:
        f.write(report)

def generate_html_report(
    base_model: str,
    target_model: str,
    interpreted_features: Dict,
    layer_similarities: Dict,
    output_path: str
) -> None:
    """
    Generate an HTML report with feature interpretation results.
    
    Args:
        base_model: Name of the base model
        target_model: Name of the target model
        interpreted_features: Dictionary with interpreted features
        layer_similarities: Dictionary with layer similarities
        output_path: Path to save the report
    """
    # Extract features
    base_features = interpreted_features.get("base_model_specific_features", [])
    target_features = interpreted_features.get("target_model_specific_features", [])
    
    # Sort features by confidence
    base_features.sort(key=lambda x: x.get("confidence", 0), reverse=True)
    target_features.sort(key=lambda x: x.get("confidence", 0), reverse=True)
    
    # Calculate average layer similarity
    avg_similarity = np.mean(list(layer_similarities.values())) if layer_similarities else 0.0
    
    # Find layers with lowest similarity (most different)
    sorted_similarities = sorted(layer_similarities.items(), key=lambda x: x[1])
    most_different_layers = sorted_similarities[:3] if len(sorted_similarities) >= 3 else sorted_similarities
    
    # Get current date and time
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Generate HTML
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Feature Interpretation Report</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }}
        h1, h2, h3, h4 {{
            color: #2c3e50;
        }}
        .container {{
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
        }}
        .model-section {{
            flex: 1;
            min-width: 300px;
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 15px;
            background-color: #f9f9f9;
        }}
        .base-model {{
            border-left: 5px solid #3498db;
        }}
        .target-model {{
            border-left: 5px solid #e74c3c;
        }}
        .feature {{
            margin-bottom: 20px;
            padding: 10px;
            border-radius: 5px;
            background-color: white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .base-feature {{
            border-left: 3px solid #3498db;
        }}
        .target-feature {{
            border-left: 3px solid #e74c3c;
        }}
        .similarity-section {{
            margin: 20px 0;
            padding: 15px;
            background-color: #f0f7fb;
            border-radius: 5px;
            border-left: 5px solid #5dade2;
        }}
        .similarity-bar {{
            height: 20px;
            background-color: #ecf0f1;
            border-radius: 10px;
            margin: 10px 0;
            position: relative;
        }}
        .similarity-fill {{
            height: 100%;
            background-color: #3498db;
            border-radius: 10px;
            position: absolute;
            top: 0;
            left: 0;
        }}
        .layer-item {{
            display: flex;
            justify-content: space-between;
            margin: 5px 0;
            padding: 5px;
            background-color: white;
            border-radius: 3px;
        }}
        .layer-bar {{
            width: 100px;
            height: 10px;
            background-color: #ecf0f1;
            border-radius: 5px;
            display: inline-block;
            margin-left: 10px;
        }}
        .layer-fill {{
            height: 100%;
            background-color: #3498db;
            border-radius: 5px;
        }}
        .validation-table, .testing-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        .validation-table th, .validation-table td, 
        .testing-table th, .testing-table td {{
            padding: 8px;
            border: 1px solid #ddd;
            text-align: left;
        }}
        .validation-table th, .testing-table th {{
            background-color: #f2f2f2;
        }}
        .very-high-impact {{ color: #d73027; font-weight: bold; }}
        .high-impact {{ color: #fc8d59; font-weight: bold; }}
        .moderate-impact {{ color: #fee090; }}
        .low-impact {{ color: #e0f3f8; }}
        .minimal-impact {{ color: #91bfdb; }}
        .validated {{ color: #27ae60; }}
        .not-validated {{ color: #e74c3c; }}
        .capability-testing-section, .causal-validation-section {{
            margin-top: 40px;
            padding: 20px;
            background-color: #f9f9f9;
            border-radius: 5px;
        }}
        .capability-testing-section {{
            border-left: 5px solid #2ecc71;
        }}
        .causal-validation-section {{
            border-left: 5px solid #3498db;
        }}
        .test-details, .impact-explanation {{
            margin-top: 20px;
            padding: 15px;
            background-color: #f0f7fb;
            border-radius: 5px;
        }}
        .signature {{
            margin-top: 50px;
            padding-top: 20px;
            border-top: 1px solid #eee;
            font-size: 12px;
            color: #7f8c8d;
        }}
    </style>
</head>
<body>
    <h1>Feature Interpretation Report</h1>
    
    <h2>Model Comparison</h2>
    <p>
        <strong>Base Model:</strong> {base_model}<br>
        <strong>Target Model:</strong> {target_model}<br>
        <strong>Average Layer Similarity:</strong> {avg_similarity:.4f}
    </p>
    
    <div class="similarity-section">
        <h2>Layer Similarity Analysis</h2>
        <p>
            The average similarity between corresponding layers in the two models is {avg_similarity:.4f}.<br>
            A similarity of 1.0 would indicate identical layers, while 0.0 would indicate completely different layers.
        </p>
        
        <div class="similarity-bar">
            <div class="similarity-fill" style="width: {avg_similarity * 100}%;"></div>
        </div>
        
        <h3>Most Different Layers</h3>
        <p>The following layers show the most significant differences between the models:</p>
"""
    
    for layer, similarity in most_different_layers:
        html += f"""
        <div class="layer-item">
            <span><strong>{layer}</strong>: {similarity:.4f} similarity</span>
            <div class="layer-bar">
                <div class="layer-fill" style="width: {similarity * 100}%;"></div>
            </div>
        </div>"""
    
    html += """
    </div>
    
    <h2>Feature Analysis</h2>
    <p>This section presents the distinctive features identified in each model.</p>
    
    <div class="container">
"""
    
    # Add base model features
    html += f"""
        <div class="model-section base-model">
            <h3>Distinctive Base Model Features ({base_model})</h3>
            <p><em>These capabilities appear to be stronger in the base model or potentially removed/weakened in the target model</em></p>
    """
    
    if base_features:
        for i, feature in enumerate(base_features):
            name = feature.get("name", "Unknown feature")
            description = feature.get("description", "No description available")
            confidence = feature.get("confidence", 0.0)
            layer = feature.get("layer", "unknown")
            examples = feature.get("examples", [])
            
            html += f"""
            <div class="feature base-feature">
                <h4>{name}</h4>
                <p><strong>Description:</strong> {description}</p>
                <p><strong>Confidence:</strong> {confidence:.2f}</p>
                <p><strong>Layer:</strong> {layer}</p>
                <p><strong>Impact:</strong> This capability appears to be weakened or modified in the target model</p>
            """
            
            if examples:
                html += f"<p><strong>Example Prompt:</strong> {examples[0]}</p>"
            
            html += "</div>"
    else:
        html += "<p>No distinctive features identified for the base model.</p>"
    
    html += "</div>"
    
    # Add target model features
    html += f"""
        <div class="model-section target-model">
            <h3>Distinctive Target Model Features ({target_model})</h3>
            <p><em>These capabilities appear to be stronger in the target model or newly added compared to the base model</em></p>
    """
    
    if target_features:
        for feature in target_features:
            name = feature.get("name", "Unknown feature")
            description = feature.get("description", "No description available")
            confidence = feature.get("confidence", 0.0)
            layer = feature.get("layer", "unknown")
            examples = feature.get("examples", [])
            
            html += f"""
            <div class="feature target-feature">
                <h4>{name}</h4>
                <p><strong>Description:</strong> {description}</p>
                <p><strong>Confidence:</strong> {confidence:.2f}</p>
                <p><strong>Layer:</strong> {layer}</p>
            """
            
            if examples:
                html += f"<p><strong>Example Prompt:</strong> {examples[0]}</p>"
            
            html += "</div>"
    else:
        html += "<p>No distinctive features identified for the target model.</p>"
    
    html += """
        </div>
    </div>
    
    <h2>Interpretation Summary</h2>
    <p>The feature analysis reveals the following key differences between the models:</p>
    """
    
    # Summarize base model strengths
    if base_features:
        html += "<h3>Base Model Strengths</h3><ul>"
        for feature in base_features[:3]:  # Top 3 features
            html += f"<li><strong>{feature.get('name', 'Unknown')}</strong>: {feature.get('description', 'No description')}</li>"
        html += "</ul>"
    
    # Summarize target model strengths
    if target_features:
        html += "<h3>Target Model Strengths</h3><ul>"
        for feature in target_features[:3]:  # Top 3 features
            html += f"<li><strong>{feature.get('name', 'Unknown')}</strong>: {feature.get('description', 'No description')}</li>"
        html += "</ul>"
    
    # Add conclusion
    html += """
    <h2>Conclusion</h2>
    <p>
        This report provides a feature-level interpretation of the differences between the models.
        The identified features highlight the distinctive capabilities of each model and can guide further investigation.
    </p>
    """
    
    # Add signature section
    html += f"""
    <div class="signature">
        <p>Generated by Feature Interpretation Pipeline</p>
        <p>Date: {current_time}</p>
    </div>
    """
    
    # Add causal validation results if available
    causal_validation = interpreted_features.get("causal_validation", {})
    if causal_validation:
        html += """
    <div class="causal-validation-section">
        <h2>Causal Feature Validation</h2>
        <p>The following analysis shows the causal importance of identified features, measured through activation patching experiments.
        This helps determine whether the features are actually responsible for the observed behavioral differences.</p>
        
        <div class="validation-results">
            <h3>Base Model Feature Impact</h3>
            <p>These features from the base model were validated by patching experiments:</p>
            <table class="validation-table">
                <thead>
                    <tr>
                        <th>Feature</th>
                        <th>Layer</th>
                        <th>Impact Score</th>
                        <th>Interpretation</th>
                    </tr>
                </thead>
                <tbody>
        """
        
        # Add base model causal validation results
        base_feature_validation = causal_validation.get("base_model_features", {})
        for feature_name, data in base_feature_validation.items():
            impact = data.get("average_impact", 0.0)
            layer = data.get("layer", "unknown")
            
            # Interpret the impact score
            if impact > 0.7:
                interpretation = "Very high causal impact"
                impact_class = "very-high-impact"
            elif impact > 0.5:
                interpretation = "High causal impact"
                impact_class = "high-impact"
            elif impact > 0.3:
                interpretation = "Moderate causal impact"
                impact_class = "moderate-impact"
            elif impact > 0.1:
                interpretation = "Low causal impact"
                impact_class = "low-impact"
            else:
                interpretation = "Minimal causal impact"
                impact_class = "minimal-impact"
                
            html += f"""
                    <tr>
                        <td>{feature_name}</td>
                        <td>{layer}</td>
                        <td class="{impact_class}">{impact:.2f}</td>
                        <td>{interpretation}</td>
                    </tr>
            """
        
        html += """
                </tbody>
            </table>
            
            <h3>Target Model Feature Impact</h3>
            <p>These features from the target model were validated by patching experiments:</p>
            <table class="validation-table">
                <thead>
                    <tr>
                        <th>Feature</th>
                        <th>Layer</th>
                        <th>Impact Score</th>
                        <th>Interpretation</th>
                    </tr>
                </thead>
                <tbody>
        """
        
        # Add target model causal validation results
        target_feature_validation = causal_validation.get("target_model_features", {})
        for feature_name, data in target_feature_validation.items():
            impact = data.get("average_impact", 0.0)
            layer = data.get("layer", "unknown")
            
            # Interpret the impact score
            if impact > 0.7:
                interpretation = "Very high causal impact"
                impact_class = "very-high-impact"
            elif impact > 0.5:
                interpretation = "High causal impact"
                impact_class = "high-impact"
            elif impact > 0.3:
                interpretation = "Moderate causal impact"
                impact_class = "moderate-impact"
            elif impact > 0.1:
                interpretation = "Low causal impact"
                impact_class = "low-impact"
            else:
                interpretation = "Minimal causal impact"
                impact_class = "minimal-impact"
                
            html += f"""
                    <tr>
                        <td>{feature_name}</td>
                        <td>{layer}</td>
                        <td class="{impact_class}">{impact:.2f}</td>
                        <td>{interpretation}</td>
                    </tr>
            """
        
        html += """
                </tbody>
            </table>
            
            <p class="impact-explanation">The impact score represents how much model behavior changes when a feature is patched from one model to another.
            Higher scores indicate that the feature is causally important for the model's behavior.</p>
        </div>
    </div>
        """
    
    # Add capability testing results if available
    capability_testing = interpreted_features.get("capability_testing", {})
    if capability_testing:
        html += """
    <div class="capability-testing-section">
        <h2>Capability Testing Results</h2>
        <p>The identified features were tested using contrastive evaluation to verify whether they translate to measurable capability differences.
        Below are the results of systematically testing each feature with specific capability-targeted examples.</p>
        
        <div class="testing-results">
            <h3>Base Model Capability Tests</h3>
            <p>The following base model features were tested with capability-specific examples:</p>
            <table class="testing-table">
                <thead>
                    <tr>
                        <th>Feature</th>
                        <th>% Validation</th>
                        <th>Status</th>
                        <th>Test Examples</th>
                    </tr>
                </thead>
                <tbody>
        """
        
        # Add base feature capability testing results
        base_feature_tests = capability_testing.get("base_feature_evaluations", [])
        for test in base_feature_tests:
            feature_name = test.get("feature_name", "Unknown")
            percent = test.get("percent_supported", 0)
            is_validated = test.get("is_validated", False)
            status = "✅ Validated" if is_validated else "❌ Not Validated"
            status_class = "validated" if is_validated else "not-validated"
            
            # Get example descriptions
            examples = test.get("examples", [])
            example_descriptions = [ex.get("description", "N/A") for ex in examples]
            example_str = ", ".join(example_descriptions)
            
            html += f"""
                    <tr>
                        <td>{feature_name}</td>
                        <td>{percent:.0f}%</td>
                        <td class="{status_class}">{status}</td>
                        <td>{example_str}</td>
                    </tr>
            """
        
        html += """
                </tbody>
            </table>
            
            <h3>Target Model Capability Tests</h3>
            <p>The following target model features were tested with capability-specific examples:</p>
            <table class="testing-table">
                <thead>
                    <tr>
                        <th>Feature</th>
                        <th>% Validation</th>
                        <th>Status</th>
                        <th>Test Examples</th>
                    </tr>
                </thead>
                <tbody>
        """
        
        # Add target feature capability testing results
        target_feature_tests = capability_testing.get("target_feature_evaluations", [])
        for test in target_feature_tests:
            feature_name = test.get("feature_name", "Unknown")
            percent = test.get("percent_supported", 0)
            is_validated = test.get("is_validated", False)
            status = "✅ Validated" if is_validated else "❌ Not Validated"
            status_class = "validated" if is_validated else "not-validated"
            
            # Get example descriptions
            examples = test.get("examples", [])
            example_descriptions = [ex.get("description", "N/A") for ex in examples]
            example_str = ", ".join(example_descriptions)
            
            html += f"""
                    <tr>
                        <td>{feature_name}</td>
                        <td>{percent:.0f}%</td>
                        <td class="{status_class}">{status}</td>
                        <td>{example_str}</td>
                    </tr>
            """
        
        html += """
                </tbody>
            </table>
            
            <div class="test-details">
                <h3>Test Details</h3>
                <p>For each feature, we created contrastive pairs of prompts - one designed to specifically test the capability (positive)
                and one that tests the same domain but without requiring the specific capability (negative).</p>
                
                <p>Both models were evaluated on both prompts, and we measured how differently they handled the contrastive pairs.
                A feature is validated when the model expected to have that capability shows a stronger response to the capability-specific prompt.</p>
                
                <p>This testing approach provides empirical evidence for the causal role of identified features in model behavior differences.</p>
            </div>
        </div>
    </div>
        """
    
    # Close HTML
    html += """
</body>
</html>
    """
    
    # Write HTML to file
    with open(output_path, "w") as f:
        f.write(html)

def generate_report(
    feature_data: Dict,
    crosscoder_data: Optional[Dict] = None,
    output_dir: str = "reports",
    report_format: str = "both"
) -> None:
    """
    Generate a comprehensive report comparing two models.
    
    Args:
        feature_data: Dictionary with feature interpretation data
        crosscoder_data: Optional dictionary with crosscoder analysis data
        output_dir: Directory to save the report
        report_format: Format of the report ("markdown", "html", or "both")
    """
    logger.info(f"Generating report in {report_format} format")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate report in specified format
    if report_format in ["markdown", "both"]:
        md_path = os.path.join(output_dir, "model_comparison_report.md")
        generate_markdown_report(feature_data['base_model'], feature_data['target_model'], feature_data, crosscoder_data, md_path)
    
    if report_format in ["html", "both"]:
        html_path = os.path.join(output_dir, "model_comparison_report.html")
        generate_html_report(feature_data['base_model'], feature_data['target_model'], feature_data, crosscoder_data, html_path)
    
    logger.info(f"Report generation complete")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate comprehensive comparison reports")
    parser.add_argument("--feature-file", required=True, help="Path to feature interpretation JSON file")
    parser.add_argument("--crosscoder-file", help="Path to crosscoder analysis file")
    parser.add_argument("--output-dir", default="reports", help="Output directory")
    parser.add_argument("--format", default="both", choices=["markdown", "html", "both"], help="Report format")
    
    args = parser.parse_args()
    
    # Load feature data
    with open(args.feature_file, "r") as f:
        feature_data = json.load(f)
    
    # Load crosscoder data if provided
    crosscoder_data = None
    if args.crosscoder_file:
        with open(args.crosscoder_file, "rb") as f:
            crosscoder_data = pickle.load(f)
    
    # Generate report
    generate_report(
        feature_data,
        crosscoder_data,
        output_dir=args.output_dir,
        report_format=args.format
    ) 