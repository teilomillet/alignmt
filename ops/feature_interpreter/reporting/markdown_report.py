"""
Markdown report generation module.

This module generates Markdown reports based on feature interpretation results.
"""

import logging
import os
from typing import Dict, List, Tuple, Optional, Union, Any
from datetime import datetime
import numpy as np

# Configure logging
logger = logging.getLogger(__name__)

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
    logger.info(f"Generating Markdown report at {output_path}")
    
    # Extract features
    base_features = interpreted_features.get("base_model_specific_features", [])
    target_features = interpreted_features.get("target_model_specific_features", [])
    shared_features = interpreted_features.get("shared_features", [])
    
    # Sort features by confidence
    base_features.sort(key=lambda x: x.get("confidence", 0), reverse=True)
    target_features.sort(key=lambda x: x.get("confidence", 0), reverse=True)
    
    # Calculate average layer similarity
    avg_similarity = np.mean(list(layer_similarities.values())) if layer_similarities else 0.0
    
    # Find layers with lowest similarity (most different)
    sorted_similarities = sorted(layer_similarities.items(), key=lambda x: x[1])
    most_different_layers = sorted_similarities[:3] if len(sorted_similarities) >= 3 else sorted_similarities
    
    # Start building the report
    report = f"# Model Comparison: {base_model} vs {target_model}\n\n"
    report += f"*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n"
    
    report += f"## Model Comparison\n\n"
    report += f"- **Base Model**: {base_model}\n"
    report += f"- **Target Model**: {target_model}\n"
    report += f"- **Average Layer Similarity**: {avg_similarity:.4f}\n\n"
    
    report += "## Layer Similarity Analysis\n\n"
    report += f"The average similarity between corresponding layers in the two models is {avg_similarity:.4f}. \n"
    report += "A similarity of 1.0 would indicate identical layers, while 0.0 would indicate completely different layers.\n\n"
    
    report += "### Most Different Layers\n\n"
    for layer, similarity in most_different_layers:
        report += f"- **{layer}**: {similarity:.4f} similarity\n"
    
    report += "\n## Feature Analysis\n\n"
    report += "This section presents the distinctive features identified in each model.\n\n"
    
    # Add base model features
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
    report += "\n## Interpretation Summary\n\n"
    
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
    report += "\n## Conclusion\n\n"
    report += "This report provides a feature-level interpretation of the differences between the models. \n"
    report += "The identified features highlight the distinctive capabilities of each model and can guide further investigation.\n"
    
    # Add decoder analysis section if available
    decoder_analysis = interpreted_features.get("decoder_analysis", None)
    if decoder_analysis:
        report += "\n## Decoder Weight Analysis\n\n"
        
        categorized = decoder_analysis.get("categorized_features", {})
        if categorized:
            report += "### Feature Categorization by Decoder Norms\n\n"
            
            base_specific = categorized.get("base_specific", [])
            target_specific = categorized.get("target_specific", [])
            shared = categorized.get("shared", [])
            
            report += f"- **Base-specific features**: {len(base_specific)}\n"
            report += f"- **Target-specific features**: {len(target_specific)}\n" 
            report += f"- **Shared features**: {len(shared)}\n\n"
            
            if base_specific:
                report += "#### Top Base-Specific Features\n\n"
                for i, feature in enumerate(sorted(base_specific, key=lambda x: 1/x.get('ratio', 1))[:5]):
                    report += f"- Feature {feature.get('id', 'unknown')}: base_norm={feature.get('base_norm', 0):.4f}, target_norm={feature.get('target_norm', 0):.4f}, ratio={feature.get('ratio', 0):.4f}\n"
                report += "\n"
            
            if target_specific:
                report += "#### Top Target-Specific Features\n\n"
                for i, feature in enumerate(sorted(target_specific, key=lambda x: x.get('ratio', 1))[:5]):
                    report += f"- Feature {feature.get('id', 'unknown')}: base_norm={feature.get('base_norm', 0):.4f}, target_norm={feature.get('target_norm', 0):.4f}, ratio={feature.get('ratio', 0):.4f}\n"
                report += "\n"
        
        # Add feature alignment section if available
        alignment = decoder_analysis.get("alignment_results", {}).get("alignment_scores", {})
        if alignment:
            report += "### Feature Alignment Analysis\n\n"
            
            # Calculate alignment statistics
            alignments = list(alignment.values())
            neg_aligned = sum(1 for a in alignments if a < 0)
            low_aligned = sum(1 for a in alignments if 0 <= a < 0.5)
            high_aligned = sum(1 for a in alignments if 0.5 <= a <= 1.0)
            
            report += f"- **Negatively aligned features**: {neg_aligned} ({neg_aligned/len(alignments):.1%})\n"
            report += f"- **Low alignment features (0-0.5)**: {low_aligned} ({low_aligned/len(alignments):.1%})\n"
            report += f"- **High alignment features (0.5-1.0)**: {high_aligned} ({high_aligned/len(alignments):.1%})\n\n"
            
            report += "These statistics indicate how similarly the features function between the two models.\n"
    
    # Write the report to file
    with open(output_path, "w") as f:
        f.write(report)
    
    logger.info(f"Markdown report generated successfully at {output_path}") 