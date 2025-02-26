"""
Configuration module for the integrated feature+crosscoder pipeline.

This module defines the configuration dataclass for the integrated pipeline
that combines crosscoder and feature interpreter functionality.
"""

import os
import logging
from dataclasses import dataclass, field
from typing import Optional, List

from ..feature_interpreter.pipeline.config import PipelineConfig

logger = logging.getLogger(__name__)

@dataclass
class IntegratedPipelineConfig(PipelineConfig):
    """Configuration for integrated feature+crosscoder pipeline."""
    
    # Crosscoder specific configurations
    crosscoder_output_dir: Optional[str] = None
    skip_crosscoder: bool = False
    crosscoder_param_types: List[str] = field(default_factory=lambda: ["gate_proj.weight"])
    crosscoder_save_crosscoded_models: bool = False
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        super().__post_init__()
        
        # Set default crosscoder output directory if not provided
        if not self.crosscoder_output_dir:
            self.crosscoder_output_dir = os.path.join(self.output_dir, "crosscoder")
            
        # Create crosscoder output directory
        os.makedirs(self.crosscoder_output_dir, exist_ok=True) 