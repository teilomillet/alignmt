"""
Hooks module for activation extraction.

This module provides functions to register hooks on model layers
for capturing activations during forward passes.
"""

import logging
from typing import Dict, List, Tuple, Callable
from transformers import AutoModelForCausalLM

# Configure logging
logger = logging.getLogger(__name__)

def register_activation_hooks(
    model: AutoModelForCausalLM,
    layer_names: List[str]
) -> Tuple[Dict[str, List], List[Callable]]:
    """
    Register hooks to capture activations from specified layers.
    
    Args:
        model: The model to hook into
        layer_names: Names of layers to capture activations from
        
    Returns:
        Tuple of (activations dictionary, list of hook handles)
    """
    logger.info(f"Registering activation hooks for {len(layer_names)} layers")
    activations = {name: [] for name in layer_names}
    handles = []
    
    def get_activation(name):
        def hook(module, input, output):
            # Handle different output types
            if isinstance(output, tuple):
                # Some modules return tuples, take the first element
                activation = output[0].detach()
            else:
                activation = output.detach()
            
            # Store the activation
            activations[name].append(activation)
        return hook
    
    # Map from layer name patterns to actual module paths
    # This is needed because weight names (like 'model.layers.0.mlp.gate_proj.weight')
    # don't directly map to module paths for hooks
    layer_name_to_module_path = {
        # For transformer models, we typically want to hook into the output of layers
        # rather than the weights themselves
        'model.layers.': 'model.layers.',  # Base pattern
    }
    
    for name in layer_names:
        # Convert weight names to module paths
        module_path = None
        for pattern, path_prefix in layer_name_to_module_path.items():
            if pattern in name:
                # Extract the layer number
                parts = name.split('.')
                for i, part in enumerate(parts):
                    if part == 'layers' and i+1 < len(parts) and parts[i+1].isdigit():
                        layer_num = parts[i+1]
                        # Create a module path for the transformer layer output
                        module_path = f"model.layers.{layer_num}"
                        break
        
        if not module_path:
            logger.warning(f"Could not map layer name {name} to a module path, skipping")
            continue
            
        try:
            # Navigate to the specific module
            module = model
            for part in module_path.split('.'):
                if part.isdigit():
                    module = module[int(part)]
                elif hasattr(module, part):
                    module = getattr(module, part)
                else:
                    logger.warning(f"Could not find module part {part} in {module_path}")
                    module = None
                    break
            
            if module is None:
                continue
                
            # Make sure we're hooking into a module, not a parameter
            if not hasattr(module, 'register_forward_hook'):
                logger.warning(f"Module {module_path} does not support hooks, skipping")
                continue
                
            # Register the hook
            handle = module.register_forward_hook(get_activation(name))
            handles.append(handle)
            logger.info(f"Successfully registered hook for {name} at {module_path}")
        except Exception as e:
            logger.warning(f"Error registering hook for {name}: {str(e)}")
    
    return activations, handles 