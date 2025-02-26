"""
Command-line interface for the integrated pipeline.

This module provides the command-line interface for running the 
integrated crosscoder + feature interpretation pipeline.
"""

import argparse
import json
import logging
from pathlib import Path

from .config import IntegratedPipelineConfig
from .runner import run_integrated_pipeline

logger = logging.getLogger(__name__)

def main():
    """Command-line interface for integrated pipeline."""
    parser = argparse.ArgumentParser(
        description="Integrated crosscoder + feature interpretation pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model arguments
    model_group = parser.add_argument_group("Model Configuration")
    model_group.add_argument("--base-model", default="Qwen/Qwen2-1.5B", help="Base model name or path")
    model_group.add_argument("--target-model", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", help="Target model name or path")
    model_group.add_argument("--output-dir", default="integrated_analysis", help="Output directory")
    model_group.add_argument("--device", default="cuda", help="Device to use (cuda or cpu)")
    model_group.add_argument("--cache-dir", default=str(Path.home() / "backups" / "huggingface_cache"), help="Cache directory for models")
    model_group.add_argument("--quantization", default="fp16", choices=["fp32", "fp16", "int8"], help="Quantization method")
    
    # Skip flags for feature interpreter
    skip_fi_group = parser.add_argument_group("Feature Interpreter Steps (skip options)")
    skip_fi_group.add_argument("--skip-activations", action="store_true", help="Skip activation extraction")
    skip_fi_group.add_argument("--skip-naming", action="store_true", help="Skip feature naming and interpretation")
    skip_fi_group.add_argument("--skip-visualization", action="store_true", help="Skip visualization creation")
    skip_fi_group.add_argument("--skip-report", action="store_true", help="Skip report generation")
    skip_fi_group.add_argument("--skip-capability-testing", action="store_true", help="Skip capability testing")
    skip_fi_group.add_argument("--skip-decoder-analysis", action="store_true", help="Skip decoder weight analysis")
    
    # Crosscoder options
    crosscoder_group = parser.add_argument_group("Crosscoder Options")
    crosscoder_group.add_argument("--skip-crosscoder", action="store_true", help="Skip crosscoder analysis")
    crosscoder_group.add_argument("--crosscoder-output-dir", help="Crosscoder output directory (defaults to output-dir/crosscoder)")
    crosscoder_group.add_argument("--crosscoder-param-types", nargs="+", default=["gate_proj.weight"], 
                                 help="Parameter types to analyze (space-separated)")
    crosscoder_group.add_argument("--save-crosscoded-models", action="store_true", help="Save crosscoded model parameters (uses more disk space)")
    
    # Analysis parameters
    params_group = parser.add_argument_group("Analysis Parameters")
    params_group.add_argument("--report-format", default="markdown", choices=["markdown", "html", "both"], help="Report format")
    params_group.add_argument("--feature-threshold", type=float, default=0.1, help="Threshold for identifying distinctive features")
    params_group.add_argument("--norm-ratio-threshold", type=float, default=1.5, help="Threshold for norm ratio to categorize features")
    params_group.add_argument("--n-clusters", type=int, default=5, help="Number of clusters for feature clustering")
    
    # Input data
    data_group = parser.add_argument_group("Input Data")
    data_group.add_argument("--prompts-file", default="ops/feature_interpreter/prompts.json", help="Path to prompts JSON file")
    
    args = parser.parse_args()
    
    # Load prompts
    prompt_categories = {}
    try:
        with open(args.prompts_file, "r") as f:
            prompt_categories = json.load(f)
    except Exception as e:
        logger.warning(f"Could not load prompts from {args.prompts_file}: {e}")
        logger.warning("Using default prompts instead")
    
    # Create configuration
    config = IntegratedPipelineConfig(
        # Model configs
        base_model=args.base_model,
        target_model=args.target_model,
        output_dir=args.output_dir,
        device=args.device,
        cache_dir=args.cache_dir,
        quantization=args.quantization,
        
        # Feature interpreter skip flags
        skip_activations=args.skip_activations,
        skip_naming=args.skip_naming,
        skip_visualization=args.skip_visualization,
        skip_report=args.skip_report,
        skip_capability_testing=args.skip_capability_testing,
        skip_decoder_analysis=args.skip_decoder_analysis,
        
        # Analysis parameters
        feature_threshold=args.feature_threshold,
        norm_ratio_threshold=args.norm_ratio_threshold,
        n_clusters=args.n_clusters,
        report_format=args.report_format,
        
        # Prompt data
        prompt_categories=prompt_categories,
        
        # Crosscoder configs
        crosscoder_output_dir=args.crosscoder_output_dir,
        skip_crosscoder=args.skip_crosscoder,
        crosscoder_param_types=args.crosscoder_param_types,
        crosscoder_save_crosscoded_models=args.save_crosscoded_models
    )
    
    # Run the integrated pipeline
    run_integrated_pipeline(config)
    
    # Print summary
    print("\nIntegrated pipeline complete!")
    print(f"Results saved to: {config.output_dir}")
    
    # Print feature interpretation report location
    if not args.skip_report:
        report_format = args.report_format
        if report_format == "both":
            print(f"HTML report: {config.output_dir}/reports/model_comparison_report.html")
            print(f"Markdown report: {config.output_dir}/reports/model_comparison_report.md")
        elif report_format == "html":
            print(f"HTML report: {config.output_dir}/reports/model_comparison_report.html")
        else:
            print(f"Markdown report: {config.output_dir}/reports/model_comparison_report.md")
    
    # Print crosscoder results location if available
    if not args.skip_crosscoder:
        print(f"Crosscoder analysis saved to: {config.crosscoder_output_dir}")

if __name__ == "__main__":
    main() 