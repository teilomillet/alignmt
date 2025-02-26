"""
Example script demonstrating how to use the feature interpretation pipeline.

This script shows how to use the feature interpretation pipeline to compare
two models and generate a comprehensive report of their differences.
"""

from pathlib import Path

from ops.feature_interpreter import run_feature_interpretation_pipeline

def main():
    """Run the feature interpretation pipeline with example models."""
    # Define models to compare
    base_model = "Qwen/Qwen2-1.5B"
    target_model = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    
    # Define output directory
    output_dir = "feature_interpretation_example"
    
    # Define cache directory
    cache_dir = str(Path.home() / "backups" / "huggingface_cache")
    
    # Define custom prompt categories
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
    
    # Run the feature interpretation pipeline
    run_feature_interpretation_pipeline(
        base_model=base_model,
        target_model=target_model,
        output_dir=output_dir,
        prompt_categories=prompt_categories,
        device="cuda",
        cache_dir=cache_dir,
        quantization="fp16",
        skip_activations=False,
        skip_naming=False,
        skip_visualization=False,
        skip_report=False,
        report_format="both",
        feature_threshold=0.3
    )
    
    print(f"Feature interpretation pipeline complete. Results saved to {output_dir}")
    print(f"You can view the report at {output_dir}/reports/model_comparison_report.html")

if __name__ == "__main__":
    main() 