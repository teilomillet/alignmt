"""
Configuration module for the feature interpretation pipeline.

This module defines the configuration dataclass for the pipeline that 
holds all parameters and settings.
"""

import os
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, List

logger = logging.getLogger(__name__)

@dataclass
class PipelineConfig:
    """Configuration for feature interpretation pipeline."""
    # Model configs
    base_model: str
    target_model: str
    output_dir: str
    device: str = "cuda"
    cache_dir: Optional[str] = None
    quantization: str = "fp16"
    
    # Pipeline steps to skip
    skip_activations: bool = False
    skip_naming: bool = False
    skip_visualization: bool = False  
    skip_report: bool = False
    skip_capability_testing: bool = False
    skip_decoder_analysis: bool = False
    
    # Analysis parameters
    feature_threshold: float = 0.1
    norm_ratio_threshold: float = 1.5
    n_clusters: int = 5
    report_format: str = "markdown"
    
    # Prompt data
    prompt_categories: Dict[str, List[str]] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        os.makedirs(self.output_dir, exist_ok=True)
        
        if not self.prompt_categories:
            logger.warning("No prompt categories provided, using defaults")
            self.prompt_categories = {
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