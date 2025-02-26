#!/usr/bin/env python3
"""
Focused Reasoning Capability Analysis

This script runs a focused analysis of reasoning capabilities between two language models,
identifying feature-level differences in how models approach various forms of reasoning.
"""

import argparse
import json
import os
from pathlib import Path
import time
import logging

from ops.feature_interpreter import PipelineConfig, run_feature_interpretation_pipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def setup_argument_parser():
    """Set up command line argument parser."""
    parser = argparse.ArgumentParser(
        description='Run focused reasoning capability analysis between language models',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model selection arguments
    parser.add_argument('--base-model', type=str, required=True,
                        help='HuggingFace model ID or path for the base model')
    parser.add_argument('--target-model', type=str, required=True,
                        help='HuggingFace model ID or path for the target model')
    
    # Output configuration
    parser.add_argument('--output-dir', type=str, default='reasoning_analysis_results',
                        help='Directory to save analysis results')
    
    # Hardware options
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to run analysis on (cuda, cpu, or cuda:n)')
    parser.add_argument('--cache-dir', type=str, default=None,
                        help='HuggingFace cache directory')
    parser.add_argument('--quantization', type=str, default='fp16',
                        choices=['fp16', 'fp32', '8bit', '4bit', 'none'],
                        help='Quantization method for models')
    
    # Analysis options
    parser.add_argument('--feature-threshold', type=float, default=0.25,
                        help='Activation difference threshold for feature identification')
    parser.add_argument('--reasoning-categories', type=str, nargs='+',
                        default=['step_by_step_reasoning', 'chain_of_thought', 'formal_logic',
                                'causal_reasoning', 'probabilistic_reasoning', 'counterfactual_reasoning'],
                        help='Reasoning categories to analyze')
    
    # Skip options
    parser.add_argument('--skip-visualization', action='store_true',
                        help='Skip visualization generation')
    parser.add_argument('--skip-capability-testing', action='store_true',
                        help='Skip capability testing')
    
    return parser

def load_prompts(categories=None):
    """Load reasoning-focused prompts from prompts.json file.
    
    Args:
        categories: List of reasoning categories to include (None for all)
        
    Returns:
        Dictionary of prompt categories
    """
    # Find the prompts.json file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    prompts_file = os.path.join(script_dir, 'prompts.json')
    
    try:
        with open(prompts_file, 'r') as f:
            all_prompts = json.load(f)
            
        logger.info(f"Loaded {len(all_prompts)} reasoning categories from {prompts_file}")
        
        # Filter categories if specified
        if categories:
            prompts = {}
            for cat in categories:
                if cat in all_prompts:
                    prompts[cat] = all_prompts[cat]
                else:
                    logger.warning(f"Requested category '{cat}' not found in prompts file")
            
            logger.info(f"Selected {len(prompts)} reasoning categories for analysis")
        else:
            prompts = all_prompts
            
        return prompts
        
    except Exception as e:
        logger.error(f"Failed to load prompts from {prompts_file}: {e}")
        logger.warning("Using default basic reasoning prompts instead")
        
        # Return minimal default prompts as fallback
        return {
            "basic_reasoning": [
                "Solve the equation: 2x + 3 = 7. Show all your steps.",
                "A train travels at 60 mph. How far will it travel in 2.5 hours? Explain your reasoning."
            ],
            "logic": [
                "If all A are B, and all B are C, what can we conclude about A and C? Explain your reasoning.",
                "If it's raining, then the ground is wet. The ground is wet. Does that mean it's raining? Explain."
            ]
        }

def main():
    """Run focused reasoning capability analysis between language models."""
    parser = setup_argument_parser()
    args = parser.parse_args()
    
    start_time = time.time()
    
    # Load prompts for reasoning categories
    prompts = load_prompts(args.reasoning_categories)
    
    # Print analysis configuration
    logger.info("\nRunning Focused Reasoning Capability Analysis")
    logger.info("=" * 50)
    logger.info(f"Base model: {args.base_model}")
    logger.info(f"Target model: {args.target_model}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Reasoning categories: {', '.join(prompts.keys())}")
    logger.info(f"Feature threshold: {args.feature_threshold}")
    logger.info("=" * 50 + "\n")
    
    # Create pipeline configuration
    config = PipelineConfig(
        # Model configuration
        base_model=args.base_model,
        target_model=args.target_model,
        output_dir=args.output_dir,
        device=args.device,
        cache_dir=args.cache_dir,
        quantization=args.quantization,
        
        # Analysis parameters
        skip_visualization=args.skip_visualization,
        skip_capability_testing=args.skip_capability_testing,
        prompt_categories=prompts,
        feature_threshold=args.feature_threshold,
        report_format="markdown",
        
        # Always generate the report
        skip_report=False
    )
    
    logger.info("Starting feature interpretation pipeline with reasoning focus...")
    
    # Run the pipeline
    results = run_feature_interpretation_pipeline(config)
    
    # Report completion
    elapsed_time = time.time() - start_time
    logger.info(f"\nAnalysis complete in {elapsed_time:.2f} seconds!")
    logger.info(f"Results saved to {args.output_dir}")
    logger.info(f"Report available at {os.path.join(args.output_dir, 'reports/model_comparison_report.md')}")
    
    # Summarize key findings if available
    if results and "summary" in results and "distinctive_features" in results["summary"]:
        features = results["summary"]["distinctive_features"]
        logger.info(f"\nIdentified {len(features)} distinctive reasoning features:")
        for i, feature in enumerate(features, 1):
            logger.info(f"{i}. {feature.get('name', 'Unnamed feature')} in layer {feature.get('layer_name', 'unknown')}")
    
    logger.info("\nSuggested next steps:")
    logger.info("1. Review the generated report for detailed findings")
    logger.info("2. Examine visualizations in the 'visualizations' directory")
    logger.info("3. Check feature interpretations in the 'feature_naming' directory")
    logger.info("4. Compare capability test results in the 'capability_testing' directory")
    
    return results

if __name__ == "__main__":
    main() 