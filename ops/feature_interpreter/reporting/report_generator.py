"""
Report generator module.

This module provides the main entry point for generating reports in different formats.
"""

import logging
import os
import json
import pickle
from pathlib import Path
from typing import Dict, Optional

from .markdown_report import generate_markdown_report

# Configure logging
logger = logging.getLogger(__name__)

def generate_report(
    feature_data: Dict,
    crosscoder_data: Optional[Dict] = None,
    output_dir: str = "reports",
    report_format: str = "markdown"
) -> None:
    """
    Generate a comprehensive report comparing two models.
    
    Args:
        feature_data: Dictionary with feature interpretation data
        crosscoder_data: Optional dictionary with crosscoder analysis data
        output_dir: Directory to save the report
        report_format: Format of the report (currently only "markdown" is supported)
    """
    logger.info(f"Generating report in {report_format} format")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate report in markdown format
    if report_format == "markdown":
        md_path = os.path.join(output_dir, "model_comparison_report.md")
        generate_markdown_report(
            feature_data.get('base_model', 'Base Model'),
            feature_data.get('target_model', 'Target Model'),
            feature_data,
            feature_data.get('layer_similarities', {}),
            md_path
        )
    else:
        logger.warning(f"Unsupported report format: {report_format}. Using markdown format instead.")
        md_path = os.path.join(output_dir, "model_comparison_report.md")
        generate_markdown_report(
            feature_data.get('base_model', 'Base Model'),
            feature_data.get('target_model', 'Target Model'),
            feature_data,
            feature_data.get('layer_similarities', {}),
            md_path
        )
    
    logger.info(f"Report generation complete")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate comprehensive comparison reports")
    parser.add_argument("--feature-file", required=True, help="Path to feature interpretation JSON file")
    parser.add_argument("--crosscoder-file", help="Path to crosscoder analysis file")
    parser.add_argument("--output-dir", default="reports", help="Output directory")
    parser.add_argument("--format", default="markdown", choices=["markdown"], help="Report format")
    
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