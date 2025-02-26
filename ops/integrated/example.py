"""
Example script demonstrating how to use the integrated pipeline with a focused reasoning capability analysis.

This script shows how to use the integrated pipeline to analyze specific differences in reasoning capabilities
between language models at the feature level. This focused approach enables more meaningful and interpretable
results than a broad capability assessment.
"""

from pathlib import Path
import json
import os

from ops.integrated import IntegratedPipelineConfig, run_integrated_pipeline

def main():
    """Run the integrated pipeline focused on reasoning capability analysis."""
    # Define models to compare
    base_model = "Qwen/Qwen2-1.5B"
    target_model = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    
    # Define output directory
    output_dir = "reasoning_capability_analysis"
    
    # Define cache directory
    cache_dir = str(Path.home() / "backups" / "huggingface_cache")
    
    # Load the reasoning-focused interpretability prompts
    prompts_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                               "feature_interpreter", "prompts.json")
    
    try:
        with open(prompts_file, 'r') as f:
            prompt_categories = json.load(f)
            print(f"Loaded {len(prompt_categories)} reasoning categories from {prompts_file}")
            
            # Optional: Select only a subset of categories for faster execution
            # Each category probes a different aspect of reasoning ability
            selected_categories = [
                "step_by_step_reasoning",
                "chain_of_thought", 
                "formal_logic",
                "causal_reasoning",
                "probabilistic_reasoning",
                "counterfactual_reasoning"
            ]
            
            # Additional categories available but not included in this example run:
            # - analogical_reasoning: Tests ability to map concepts between domains
            # - abductive_reasoning: Tests hypothesis generation from observations
            # - adversarial_reasoning: Tests resilience to cognitive traps
            # - constraint_satisfaction: Tests multi-constraint problem solving
            
            prompt_categories = {k: prompt_categories[k] for k in selected_categories if k in prompt_categories}
            print(f"Selected {len(prompt_categories)} reasoning categories for analysis")
            
    except Exception as e:
        print(f"Warning: Could not load reasoning prompts from {prompts_file}: {e}")
        print("Using basic reasoning prompts instead")
        prompt_categories = {
            "basic_reasoning": [
                "Solve the equation: 2x + 3 = 7. Show all your steps.",
                "A train travels at 60 mph. How far will it travel in 2.5 hours? Explain your reasoning."
            ],
            "logic": [
                "If all A are B, and all B are C, what can we conclude about A and C? Explain your reasoning.",
                "If it's raining, then the ground is wet. The ground is wet. Does that mean it's raining? Explain."
            ],
            "probabilistic": [
                "If you flip a fair coin 3 times, what is the probability of getting exactly 2 heads? Show your work.",
                "A bag contains 3 red balls and 4 blue balls. If you draw 2 balls without replacement, what is the probability of getting 2 blue balls?"
            ]
        }
    
    # Create configuration with research focus on reasoning capabilities
    config = IntegratedPipelineConfig(
        # Model configs
        base_model=base_model,
        target_model=target_model,
        output_dir=output_dir,
        device="cuda",
        cache_dir=cache_dir,
        quantization="fp16",
        
        # Feature interpreter options - ensure visualization and capability testing are enabled
        skip_activations=False,
        skip_naming=False,
        skip_visualization=False,
        skip_report=False,
        skip_capability_testing=False,  # Important for testing reasoning capabilities
        skip_decoder_analysis=False,
        
        # Analysis parameters - adjusted for reasoning feature detection
        feature_threshold=0.25,  # Slightly lower threshold to catch subtle reasoning differences
        norm_ratio_threshold=1.5,
        n_clusters=5,
        report_format="markdown",
        
        # Reasoning-focused prompt categories
        prompt_categories=prompt_categories,
        
        # Crosscoder configs
        crosscoder_output_dir=None,  # Will use default (output_dir/crosscoder)
        skip_crosscoder=False,
        crosscoder_param_types=["gate_proj.weight"],  # Gate projections often encode reasoning patterns
        crosscoder_save_crosscoded_models=False
    )
    
    print("\nRunning focused reasoning capability analysis between models:")
    print(f"- Base model: {base_model}")
    print(f"- Target model: {target_model}")
    print(f"- Research question: How do reasoning capabilities differ between these models at the feature level?")
    print(f"- Analysis categories: {', '.join(selected_categories)}\n")
    
    # Run the integrated pipeline
    run_integrated_pipeline(config)

if __name__ == "__main__":
    main() 