"""
Command-line interface for the feature interpretation pipeline.

This module provides the command-line interface for running the 
feature interpretation pipeline.
"""

import argparse
import json
import logging
from pathlib import Path

from .config import PipelineConfig
from .runner import run_feature_interpretation_pipeline

logger = logging.getLogger(__name__)

def main():
    """Command-line interface for feature interpretation pipeline."""
    parser = argparse.ArgumentParser(
        description="Feature-level model difference interpretation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model arguments
    model_group = parser.add_argument_group("Model Configuration")
    model_group.add_argument("--base-model", default="Qwen/Qwen2-1.5B", help="Base model name or path")
    model_group.add_argument("--target-model", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", help="Target model name or path")
    model_group.add_argument("--output-dir", default="feature_interpretation", help="Output directory")
    model_group.add_argument("--device", default="cuda", help="Device to use (cuda or cpu)")
    model_group.add_argument("--cache-dir", default=str(Path.home() / "backups" / "huggingface_cache"), help="Cache directory for models")
    model_group.add_argument("--quantization", default="fp16", choices=["fp32", "fp16", "int8"], help="Quantization method")
    
    # Skip flags
    skip_group = parser.add_argument_group("Pipeline Steps (skip options)")
    skip_group.add_argument("--skip-activations", action="store_true", help="Skip activation extraction")
    skip_group.add_argument("--skip-naming", action="store_true", help="Skip feature naming and interpretation")
    skip_group.add_argument("--skip-visualization", action="store_true", help="Skip visualization creation")
    skip_group.add_argument("--skip-report", action="store_true", help="Skip report generation")
    skip_group.add_argument("--skip-capability-testing", action="store_true", help="Skip capability testing")
    skip_group.add_argument("--skip-decoder-analysis", action="store_true", help="Skip decoder weight analysis")
    
    # Analysis parameters
    params_group = parser.add_argument_group("Analysis Parameters")
    params_group.add_argument("--report-format", default="markdown", choices=["markdown", "html", "both"], help="Report format")
    params_group.add_argument("--feature-threshold", type=float, default=0.3, help="Threshold for identifying distinctive features")
    params_group.add_argument("--norm-ratio-threshold", type=float, default=1.5, help="Threshold for norm ratio to categorize features")
    params_group.add_argument("--n-clusters", type=int, default=5, help="Number of clusters for feature clustering")
    
    # Input data
    data_group = parser.add_argument_group("Input Data")
    data_group.add_argument("--prompts-file", default="ops/feature_interpreter/prompts.json", help="Path to prompts JSON file")
    data_group.add_argument("--crosscoder-file", help="Path to crosscoder analysis file (optional)")
    
    args = parser.parse_args()
    
    # Load prompt categories from JSON file
    try:
        with open(args.prompts_file, 'r') as f:
            prompt_categories = json.load(f)
        logger.info(f"Successfully loaded prompts from {args.prompts_file}")
    except Exception as e:
        logger.warning(f"Failed to load prompts from {args.prompts_file}: {str(e)}")
        logger.warning("Falling back to default prompt categories")
        # Define default prompt categories as fallback
        prompt_categories = {
            "reasoning": [
                "Solve the equation: 2x + 3 = 7. Show all your steps.",
                "A train travels at 60 mph. How far will it travel in 2.5 hours? Explain your reasoning."
            ],
            "instruction_following": [
                "Write a short poem about artificial intelligence.",
                "List five benefits of regular exercise."
            ],
            "factual_knowledge": [
                "What is the capital of France?",
                "Who wrote the novel 'Pride and Prejudice'?"
            ]
        }
    
    # Create pipeline configuration
    config = PipelineConfig(
        base_model=args.base_model,
        target_model=args.target_model,
        output_dir=args.output_dir,
        device=args.device,
        cache_dir=args.cache_dir,
        quantization=args.quantization,
        skip_activations=args.skip_activations,
        skip_naming=args.skip_naming,
        skip_visualization=args.skip_visualization,
        skip_report=args.skip_report,
        skip_capability_testing=args.skip_capability_testing,
        skip_decoder_analysis=args.skip_decoder_analysis,
        report_format=args.report_format,
        feature_threshold=args.feature_threshold,
        norm_ratio_threshold=args.norm_ratio_threshold,
        n_clusters=args.n_clusters,
        prompt_categories=prompt_categories
    )
    
    # Run pipeline
    run_feature_interpretation_pipeline(config) 