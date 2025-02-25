"""
Test script for the feature interpretation pipeline.

This script tests the feature interpretation pipeline with minimal examples
to ensure it works correctly.
"""

import os
import tempfile
import unittest
import shutil
from pathlib import Path

import torch
import numpy as np

from ops.feature_interpreter.extract_activations import extract_activations
from ops.feature_interpreter.feature_naming import compute_activation_differences, extract_distinctive_features
from ops.feature_interpreter.feature_visualization import create_feature_distribution_plot
from ops.feature_interpreter.generate_report import generate_markdown_report

class TestFeatureInterpreter(unittest.TestCase):
    """Test cases for the feature interpretation pipeline."""
    
    def setUp(self):
        """Set up test environment."""
        # Create temporary directory for test outputs
        self.test_dir = tempfile.mkdtemp()
        
        # Create mock activation data
        self.mock_activations = {
            "prompt1": {
                "text": "This is a test output from the base model.",
                "activations": {
                    "layer1": torch.randn(1, 10, 20)
                }
            },
            "prompt2": {
                "text": "This is another test output from the base model.",
                "activations": {
                    "layer1": torch.randn(1, 10, 20)
                }
            }
        }
        
        self.mock_target_activations = {
            "prompt1": {
                "text": "This is a test output from the target model with more step-by-step reasoning.",
                "activations": {
                    "layer1": torch.randn(1, 10, 20)
                }
            },
            "prompt2": {
                "text": "This is another test output from the target model with more step-by-step reasoning.",
                "activations": {
                    "layer1": torch.randn(1, 10, 20)
                }
            }
        }
        
        # Create mock feature data
        self.mock_feature_data = {
            "base_model": "base_model",
            "target_model": "target_model",
            "base_model_specific_features": [
                {
                    "name": "unconstrained reasoning",
                    "layer": "layer1",
                    "confidence": 0.85,
                    "description": "Base model has stronger unconstrained reasoning features",
                    "examples": ["Solve the equation: 2x + 3 = 7..."]
                }
            ],
            "target_model_specific_features": [
                {
                    "name": "step-by-step reasoning",
                    "layer": "layer1",
                    "confidence": 0.90,
                    "description": "Target model has enhanced step-by-step reasoning",
                    "examples": ["Solve the equation: 2x + 3 = 7..."]
                }
            ]
        }
    
    def tearDown(self):
        """Clean up after tests."""
        # Remove temporary directory
        shutil.rmtree(self.test_dir)
    
    def test_compute_activation_differences(self):
        """Test computing activation differences."""
        # Compute differences
        differences = compute_activation_differences(
            self.mock_activations,
            self.mock_target_activations,
            "layer1"
        )
        
        # Check that differences were computed
        self.assertIn("prompt1", differences)
        self.assertIn("prompt2", differences)
        self.assertIn("difference", differences["prompt1"])
        self.assertIn("similarity", differences["prompt1"])
        
        # Check that difference values are reasonable
        self.assertGreaterEqual(differences["prompt1"]["difference"], 0.0)
        self.assertLessEqual(differences["prompt1"]["difference"], 1.0)
    
    def test_extract_distinctive_features(self):
        """Test extracting distinctive features."""
        # Create mock differences
        mock_differences = {
            "prompt1": {
                "difference": 0.5,
                "similarity": 0.5,
                "base_output": "Base output 1",
                "target_output": "Target output 1"
            },
            "prompt2": {
                "difference": 0.6,
                "similarity": 0.4,
                "base_output": "Base output 2",
                "target_output": "Target output 2"
            }
        }
        
        # Create mock prompt categories
        mock_categories = {
            "prompt1": "category1",
            "prompt2": "category1"
        }
        
        # Extract distinctive features
        features = extract_distinctive_features(
            mock_differences,
            mock_categories,
            threshold=0.3
        )
        
        # Check that features were extracted
        self.assertIn("category1", features)
        self.assertIn("significant_examples", features["category1"])
        self.assertGreater(len(features["category1"]["significant_examples"]), 0)
    
    def test_create_feature_distribution_plot(self):
        """Test creating feature distribution plot."""
        # Create plot
        output_path = os.path.join(self.test_dir, "test_plot.png")
        create_feature_distribution_plot(
            self.mock_feature_data,
            output_path
        )
        
        # Check that plot was created
        self.assertTrue(os.path.exists(output_path))
    
    def test_generate_markdown_report(self):
        """Test generating Markdown report."""
        # Generate report
        output_path = os.path.join(self.test_dir, "test_report.md")
        generate_markdown_report(
            self.mock_feature_data,
            output_path=output_path
        )
        
        # Check that report was created
        self.assertTrue(os.path.exists(output_path))
        
        # Check report content
        with open(output_path, "r") as f:
            content = f.read()
            self.assertIn("Model Comparison Report", content)
            self.assertIn("Base Model Distinctive Features", content)
            self.assertIn("Target Model Distinctive Features", content)

if __name__ == "__main__":
    unittest.main() 