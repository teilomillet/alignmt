"""
Feature capability testing package.

This package provides modules and functions for generating contrastive examples
and evaluating model capabilities based on feature interpretations.
"""

# First import from the modules that don't depend on evaluation.py
from .examples import generate_contrastive_examples, reset_used_prompts
from .metrics import calculate_human_experience_score

# Define a function to get evaluation functions without importing at module level
def get_evaluation_functions():
    """
    Get evaluation functions lazily to avoid circular imports.
    
    Returns:
        Tuple of (generate_response, evaluate_feature_capability) functions
    """
    from .evaluation import generate_response, evaluate_feature_capability
    return generate_response, evaluate_feature_capability

# Define functions to expose the evaluation functions more cleanly
def generate_response(*args, **kwargs):
    """
    Proxy function for evaluation.generate_response.
    
    See evaluation.generate_response for full documentation.
    """
    func, _ = get_evaluation_functions()
    return func(*args, **kwargs)

def evaluate_feature_capability(*args, **kwargs):
    """
    Proxy function for evaluation.evaluate_feature_capability.
    
    See evaluation.evaluate_feature_capability for full documentation.
    """
    _, func = get_evaluation_functions()
    return func(*args, **kwargs)

# Export specific functions directly in the package namespace
__all__ = [
    'generate_contrastive_examples',
    'reset_used_prompts',
    'calculate_human_experience_score',
    'generate_response',
    'evaluate_feature_capability',
    'get_evaluation_functions'
] 