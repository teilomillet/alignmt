"""
Test script for the feature interpretation pipeline.

This script tests the feature interpretation pipeline with minimal examples
to ensure it works correctly.
"""

import os
import tempfile
import unittest
import shutil

import torch
import numpy as np

from ops.feature_interpreter.naming import compute_activation_differences, extract_distinctive_features
from ops.feature_interpreter.visualization.basic import create_feature_distribution_plot
from ops.feature_interpreter.reporting.markdown_report import generate_markdown_report
from ops.feature_interpreter.decoder_analysis import (
    extract_feature_decoder_norms,
    identify_active_features,
    cluster_features,
    compare_feature_responses,
    generate_comprehensive_analysis
)

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
        # Note: In the implementation, differences are not necessarily bounded by 1.0
        # They can be larger depending on the vector distances
        self.assertGreaterEqual(differences["prompt1"]["difference"], 0.0)
        
        # Update the similarity check - it seems the similarity can be negative
        # Similarity is likely using cosine similarity which ranges from -1 to 1
        self.assertGreaterEqual(differences["prompt1"]["similarity"], -1.0)
        self.assertLessEqual(differences["prompt1"]["similarity"], 1.0)
    
    def test_extract_distinctive_features(self):
        """Test extracting distinctive features."""
        # Create mock differences
        mock_differences = {
            "prompt1": {
                "difference": 0.5,
                "similarity": 0.5,
                "base_output": "Base output 1",
                "target_output": "Target output 1",
                "layer": "layer1"  # Add layer information to avoid KeyError
            },
            "prompt2": {
                "difference": 0.6,
                "similarity": 0.4,
                "base_output": "Base output 2",
                "target_output": "Target output 2",
                "layer": "layer1"  # Add layer information to avoid KeyError
            }
        }
        
        # Extract distinctive features
        features = extract_distinctive_features(
            mock_differences,
            threshold=0.3,
            min_prompts=1  # Reduce min_prompts for the test to ensure we get a result
        )
        
        # Check that features were extracted correctly
        self.assertTrue(features.get('is_distinctive', False))
        self.assertEqual(features.get('layer'), 'layer1')
        self.assertEqual(features.get('significant_prompt_count'), 2)
        self.assertEqual(features.get('total_prompt_count'), 2)
    
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
        
        # Create feature data format expected by the new function
        feature_data = {
            "base_model": "mock_base_model",
            "target_model": "mock_target_model",
            "base_model_specific_features": self.mock_feature_data.get("base_model_specific_features", []),
            "target_model_specific_features": self.mock_feature_data.get("target_model_specific_features", []),
            "shared_features": self.mock_feature_data.get("shared_features", []),
            "layer_similarities": {}
        }
        
        generate_markdown_report(
            base_model="mock_base_model",
            target_model="mock_target_model",
            interpreted_features=feature_data,
            layer_similarities={},
            output_path=output_path
        )
        
        # Check that report was created
        self.assertTrue(os.path.exists(output_path))
        
        # Check report content
        with open(output_path, "r") as f:
            content = f.read()
            self.assertIn("Model Comparison", content)
            self.assertIn("Base Model", content)
            self.assertIn("Target Model", content)


class TestDecoderAnalysis(unittest.TestCase):
    """Test cases for the decoder analysis functionality."""
    
    def setUp(self):
        """Set up test environment."""
        # Create temporary directory for test outputs
        self.test_dir = tempfile.mkdtemp()
        
        # Generate synthetic data for testing
        self.num_features = 100
        self.feature_dim = 50
        self.num_prompts = 5
        
        # Create synthetic decoder matrices
        self.base_decoder = np.random.normal(0, 1, (self.num_features, self.feature_dim))
        self.target_decoder = self.base_decoder.copy()
        
        # Modify some features to create differences
        # Base-specific features (first 30%)
        base_specific_indices = np.arange(0, int(0.3 * self.num_features))
        self.target_decoder[base_specific_indices] *= 0.1
        
        # Target-specific features (next 30%)
        target_specific_indices = np.arange(
            int(0.3 * self.num_features),
            int(0.6 * self.num_features)
        )
        self.target_decoder[target_specific_indices] *= 5.0
        
        # Shared features (remaining 40%)
        # Add small random noise to all features
        self.target_decoder += np.random.normal(0, 0.1, self.target_decoder.shape)
        
        # Create synthetic activations
        self.base_activations = np.random.normal(0, 1, (self.num_features, self.num_prompts))
        self.target_activations = self.base_activations.copy()
        
        # Modify activations to match feature specificity
        self.target_activations[base_specific_indices] *= 0.2
        self.target_activations[target_specific_indices] *= 3.0
        
        # Create prompt labels
        self.prompt_labels = [f"Prompt {i+1}" for i in range(self.num_prompts)]
    
    def tearDown(self):
        """Clean up after tests."""
        # Remove temporary directory
        shutil.rmtree(self.test_dir)
    
    def test_extract_feature_decoder_norms(self):
        """Test extracting and comparing feature decoder norms."""
        feature_data = extract_feature_decoder_norms(
            self.base_decoder,
            self.target_decoder
        )
        
        # Check structure of output
        self.assertIn("feature_norms", feature_data)
        self.assertIn("feature_decoders", feature_data)
        
        # Check that all features have norms
        self.assertEqual(len(feature_data["feature_norms"]), self.num_features)
        
        # Check first feature to verify structure
        first_feature = next(iter(feature_data["feature_norms"]))
        self.assertIn("base_norm", feature_data["feature_norms"][first_feature])
        self.assertIn("target_norm", feature_data["feature_norms"][first_feature])
        self.assertIn("norm_ratio", feature_data["feature_norms"][first_feature])
    
    def test_identify_active_features(self):
        """Test identifying active features."""
        # Create activations dict as expected by the function
        activations = {
            "base_activations": self.base_activations,
            "target_activations": self.target_activations
        }
        
        active_features = identify_active_features(
            activations,
            threshold=0.5
        )
        
        # Check structure of output
        self.assertIn("base_active", active_features)
        self.assertIn("target_active", active_features)
        self.assertIn("active_in_both", active_features)
        self.assertIn("base_specific_active", active_features)
        self.assertIn("target_specific_active", active_features)
        self.assertIn("stats", active_features)
        
        # Check that stats include expected fields
        self.assertIn("total_features", active_features["stats"])
        self.assertIn("base_active_count", active_features["stats"])
        self.assertIn("target_active_count", active_features["stats"])
        
        # Verify that total count matches
        self.assertEqual(active_features["stats"]["total_features"], self.num_features)
    
    def test_cluster_features(self):
        """Test clustering features."""
        # First generate feature data
        feature_data = extract_feature_decoder_norms(
            self.base_decoder,
            self.target_decoder
        )
        
        # Apply clustering
        clusters = cluster_features(
            feature_data,
            method="kmeans",
            n_clusters=3,
            output_dir=self.test_dir
        )
        
        # Check structure of output
        self.assertIn("centers", clusters)
        self.assertIn("feature_clusters", clusters)
        
        # Check that we have expected number of clusters
        self.assertEqual(len(clusters["centers"]), 3)
        
        # Check that visualization files were created
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "feature_clusters_kmeans.png")))
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "feature_clusters_kmeans.json")))
    
    def test_compare_feature_responses(self):
        """Test comparing feature responses."""
        response_analysis = compare_feature_responses(
            self.base_activations,
            self.target_activations,
            self.prompt_labels,
            output_dir=self.test_dir
        )
        
        # Check structure of output
        self.assertIn("top_different_features", response_analysis)
        self.assertIn("prompt_specific_features", response_analysis)
        
        # Check that we have entries for each prompt
        self.assertEqual(len(response_analysis["prompt_specific_features"]), len(self.prompt_labels))
        
        # Check that visualization files were created
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "feature_response_diff.png")))
    
    def test_generate_comprehensive_analysis(self):
        """Test generating comprehensive analysis."""
        analysis_results = generate_comprehensive_analysis(
            self.base_decoder,
            self.target_decoder,
            self.base_activations,
            self.target_activations,
            self.prompt_labels,
            output_dir=self.test_dir
        )
        
        # Check structure of output
        self.assertIn("feature_data", analysis_results)
        self.assertIn("categorized_features", analysis_results)
        self.assertIn("alignment_results", analysis_results)
        self.assertIn("clustering_results", analysis_results)
        self.assertIn("response_analysis", analysis_results)
        
        # Check that summary file was created
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "comprehensive_analysis_summary.json")))
        
        # Verify categories are present in categorized features
        categorized = analysis_results["categorized_features"]
        self.assertIn("base_specific", categorized)
        self.assertIn("target_specific", categorized)
        self.assertIn("shared", categorized)
        
        # Verify that all feature categories have entries (due to our synthetic data design)
        self.assertGreater(len(categorized["base_specific"]), 0)
        self.assertGreater(len(categorized["target_specific"]), 0)


if __name__ == "__main__":
    unittest.main() 