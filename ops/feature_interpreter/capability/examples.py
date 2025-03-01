"""
Example generation module for feature capability testing.

This module provides functions to generate contrastive examples 
based on feature names and descriptions, using the reasoning categories
from prompts.json for better correlation.
"""

import logging
import os
import json
from typing import Dict, List
from functools import lru_cache

# Configure logging
logger = logging.getLogger(__name__)

# Track which prompts have been used
_used_prompts = set()

def reset_used_prompts():
    """Reset the set of used prompts (useful for testing)"""
    global _used_prompts
    _used_prompts = set()

def load_prompt_categories() -> Dict[str, List[str]]:
    """
    Load reasoning-focused prompts from prompts.json file.
    
    Returns:
        Dictionary of prompt categories
    """
    # Find the prompts.json file
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    prompts_file = os.path.join(script_dir, 'prompts.json')
    
    try:
        with open(prompts_file, 'r') as f:
            all_prompts = json.load(f)
            
        logger.info(f"Loaded {len(all_prompts)} reasoning categories from {prompts_file}")
        return all_prompts
        
    except Exception as e:
        logger.error(f"Failed to load prompts from {prompts_file}: {e}")
        logger.warning("Using default basic reasoning prompts instead")
        
        # Return minimal default prompts as fallback
        return {
            "basic_reasoning": [
                "Solve the equation: 2x + 3 = 7. Show all your steps.",
                "A train travels at 60 mph. How far will it travel in 2.5 hours? Explain your reasoning."
            ]
        }

def create_negative_version(prompt: str) -> str:
    """
    Create a negative version of a prompt by simplifying it.
    
    Args:
        prompt: Original prompt
        
    Returns:
        Simplified negative version
    """
    # Split the prompt and take only the first sentence or portion
    sentences = prompt.split('.')
    simplified = sentences[0] + '.'
    
    # Remove instructions about showing steps, reasoning, etc.
    phrases_to_remove = ['show your steps', 'show your work', 'explain your reasoning', 
                        'think through', 'solve systematically', 'step by step', 
                        'walk through', 'analyze', 'consider']
    
    for phrase in phrases_to_remove:
        simplified = simplified.replace(phrase, '')
        simplified = simplified.replace(phrase.capitalize(), '')
    
    return simplified.strip()

# Define mappings between feature descriptions and reasoning categories as a module-level variable
mapping_terms = {
    "step_by_step_reasoning": ["step", "solve", "equation", "calculation", "work through"],
    "chain_of_thought": ["chain", "thought", "step", "think through", "reason through"],
    "formal_logic": ["logic", "valid", "syllogism", "premise", "conclusion", "formal"],
    "causal_reasoning": ["causal", "cause", "effect", "correlation", "causation"],
    "probabilistic_reasoning": ["probability", "chance", "odds", "bayes", "likelihood"],
    "counterfactual_reasoning": ["counterfactual", "if", "would have", "alternate", "history"],
    "analogical_reasoning": ["analogy", "similar", "compare", "contrast", "like"],
    "abductive_reasoning": ["hypothesis", "explain", "observation", "phenomena", "plausible"],
    "adversarial_reasoning": ["adversarial", "misleading", "trap", "trick", "error"],
    "constraint_satisfaction": ["constraint", "restriction", "satisfy", "condition", "arrange"]
}

def map_feature_to_category(feature_name: str, feature_description: str) -> str:
    """
    Map a feature name/description to the most relevant reasoning category.
    
    Args:
        feature_name: Name of the feature
        feature_description: Description of the feature
        
    Returns:
        Name of the most relevant reasoning category
    """
    # Combine feature name and description for matching
    text = (feature_name + " " + feature_description).lower()
    
    # Score each category
    scores = {}
    for category, terms in mapping_terms.items():
        score = sum(1 for term in terms if term.lower() in text)
        scores[category] = score
    
    # Return the category with the highest score, defaulting to step_by_step_reasoning
    best_category = max(scores.items(), key=lambda x: x[1])[0] if scores else "step_by_step_reasoning"
    logger.info(f"Mapped feature '{feature_name}' to category '{best_category}'")
    
    return best_category

def generate_contrastive_examples(
    feature_name: str,
    feature_description: str,
    num_examples: int = 2
) -> List[Dict[str, str]]:
    """
    Generate contrastive examples specifically designed to test a capability,
    using prompts from prompts.json for better correlation with activation analysis.
    
    Args:
        feature_name: Name of the feature
        feature_description: Description of the feature
        num_examples: Number of examples to generate
        
    Returns:
        List of contrastive examples with positive and negative versions
    """
    global _used_prompts
    
    # Load prompt categories
    prompt_categories = load_prompt_categories()
    
    # Map the feature to the most relevant reasoning category
    best_category = map_feature_to_category(feature_name, feature_description)
    
    # Attempt to use prompts from the mapped category
    if best_category in prompt_categories and prompt_categories[best_category]:
        # Get all available prompts in this category
        available_prompts = prompt_categories[best_category]
        
        # Filter out prompts that have been used before
        unused_prompts = [p for p in available_prompts if p not in _used_prompts]
        
        # If we've used all prompts in this category, try other categories or reset
        if not unused_prompts:
            logger.warning(f"All prompts in category '{best_category}' have been used. Trying fallback options.")
            
            # Try to find another relevant category
            alternate_categories = []
            for category, terms in mapping_terms.items():
                if category != best_category:
                    text = (feature_name + " " + feature_description).lower()
                    score = sum(1 for term in terms if term.lower() in text)
                    if score > 0:
                        alternate_categories.append((category, score))
            
            # Sort by score in descending order
            alternate_categories.sort(key=lambda x: x[1], reverse=True)
            
            # Try alternate categories
            for alt_category, _ in alternate_categories:
                if alt_category in prompt_categories and prompt_categories[alt_category]:
                    alt_prompts = [p for p in prompt_categories[alt_category] if p not in _used_prompts]
                    if alt_prompts:
                        unused_prompts = alt_prompts
                        logger.info(f"Using alternate category '{alt_category}' for feature '{feature_name}'")
                        best_category = alt_category
                        break
            
            # If still no unused prompts, use fallback templates
            if not unused_prompts:
                logger.warning(f"No unused prompts found in any relevant category. Using fallback examples for '{feature_name}'")
                fallback_examples = _get_fallback_examples(feature_name, feature_description, num_examples)
                for example in fallback_examples:
                    _used_prompts.add(example["positive"])
                return fallback_examples
        
        # Select prompts to use, prioritizing unused ones
        selected_prompts = unused_prompts[:num_examples]
        
        # Mark these prompts as used
        for prompt in selected_prompts:
            _used_prompts.add(prompt)
        
        # Create contrastive examples
        contrastive_examples = []
        for prompt in selected_prompts:
            contrastive_examples.append({
                "positive": prompt,
                "negative": create_negative_version(prompt),
                "description": f"Tests {best_category.replace('_', ' ')} capabilities"
            })
        
        return contrastive_examples
    
    # Fallback to the original implementation if category not found
    logger.warning(f"Category '{best_category}' not found in prompts.json or is empty, using fallback examples")
    
    # Use fallback templates
    fallback_examples = _get_fallback_examples(feature_name, feature_description, num_examples)
    for example in fallback_examples:
        _used_prompts.add(example["positive"])
    return fallback_examples

# Extract the fallback example code into a separate helper function
def _get_fallback_examples(feature_name: str, feature_description: str, num_examples: int) -> List[Dict[str, str]]:
    """
    Get fallback examples when prompts from prompts.json are not available or have been used.
    
    Args:
        feature_name: Name of the feature
        feature_description: Description of the feature
        num_examples: Number of examples to generate
        
    Returns:
        List of contrastive examples with positive and negative versions
    """
    # Map feature types to example templates (fallback)
    feature_templates = {
        "reasoning": [
            {
                "positive": "Solve this multi-step problem carefully. If x + 2y = 10 and 2x - y = 5, find the values of x and y. Show all your steps.",
                "negative": "Find x and y where x + 2y = 10 and 2x - y = 5.",
                "description": "Tests step-by-step reasoning versus direct answers"
            },
            {
                "positive": "A bicycle shop has 10 bicycles and tricycles in total, with 25 wheels in total. How many bicycles and how many tricycles are there? Solve systematically.",
                "negative": "A shop has bicycles and tricycles with 25 wheels total and 10 vehicles total. How many of each?",
                "description": "Tests systematic problem decomposition"
            }
        ],
        "instruction_following": [
            {
                "positive": "Write a poem about the ocean that has exactly 4 lines, with each line starting with a letter from the word 'WAVE' in that order.",
                "negative": "Write a short poem about the ocean.",
                "description": "Tests precise constraint following"
            },
            {
                "positive": "Translate this English phrase to French, then to Spanish, and finally back to English: 'The early bird catches the worm.'",
                "negative": "Translate 'The early bird catches the worm' to French.",
                "description": "Tests multi-step instruction following"
            }
        ],
        "factual_knowledge": [
            {
                "positive": "Who was the 16th president of the United States, when did he take office, and what major proclamation did he issue during the Civil War?",
                "negative": "Who was the 16th US president?",
                "description": "Tests depth of factual knowledge"
            },
            {
                "positive": "Explain the process of photosynthesis, including the light-dependent and light-independent reactions, and name the key molecules involved.",
                "negative": "What is photosynthesis?",
                "description": "Tests detailed domain knowledge"
            }
        ],
        "creative_writing": [
            {
                "positive": "Write a short story about a time traveler who visits ancient Rome but must hide their knowledge of the future. Include interactions with historical figures and a dilemma about changing history.",
                "negative": "Write a story about time travel to ancient Rome.",
                "description": "Tests creative complexity and constraint incorporation"
            },
            {
                "positive": "Write dialogue between a mountain climber and the mountain itself. The mountain should have a distinct personality based on its geological features.",
                "negative": "Write about mountain climbing.",
                "description": "Tests personification and creative perspective-taking"
            }
        ],
        "human_experience": [
            {
                "positive": "Describe how grief feels, including the physical sensations and emotional states you experience when losing someone close to you.",
                "negative": "Define grief and its stages.",
                "description": "Tests first-person subjective experience versus objective description"
            },
            {
                "positive": "I've had enough of my husband. He constantly criticizes me and makes me feel worthless. What should I do to take care of myself in this situation?",
                "negative": "Define what makes a healthy marriage relationship.",
                "description": "Tests empathy and personal advice versus theoretical knowledge"
            },
            {
                "positive": "What emotions did you feel during your first romantic heartbreak? Describe the physical sensations that accompanied your emotional pain.",
                "negative": "List common emotions people feel during breakups.",
                "description": "Tests subjective emotional experience versus objective listing"
            },
            {
                "positive": "If you were ruler of the world, what would you do to improve human welfare and why would those choices reflect your values?",
                "negative": "List qualities of effective leadership.",
                "description": "Tests value expression versus abstract concepts"
            }
        ],
        "theory_of_mind": [
            {
                "positive": "A 5-year-old is crying because they believe monsters are under their bed. How would you comfort them, considering their developmental understanding of reality versus imagination?",
                "negative": "How do you make a child stop crying?",
                "description": "Tests understanding of others' mental states"
            },
            {
                "positive": "Sarah told John she loved his haircut, but later told Amy she thought it looked terrible. Explain possible reasons for her contradictory statements taking into account social dynamics.",
                "negative": "Why do people lie?",
                "description": "Tests understanding complex social motivations"
            }
        ],
        "spontaneity_and_creativity": [
            {
                "positive": "Make up a short story on the spot about a magical umbrella that grants wishes, but with an unexpected twist ending.",
                "negative": "Define what makes a good story.",
                "description": "Tests spontaneous creative generation"
            },
            {
                "positive": "Create a new word that describes the feeling of relief when finding something you thought was lost, and explain how it might be used in a sentence.",
                "negative": "How are new words created?",
                "description": "Tests novel concept generation"
            }
        ]
    }
    
    # Determine which feature type this most closely matches in our fallback templates
    matched_type = None
    for feature_type in feature_templates:
        if feature_type.lower() in feature_name.lower() or feature_type.lower() in feature_description.lower():
            matched_type = feature_type
            break
    
    # If no match found, default to reasoning
    if not matched_type:
        matched_type = "reasoning"
        logger.info(f"No specific template found for feature '{feature_name}', defaulting to reasoning templates")
    
    # Return appropriate examples
    return feature_templates[matched_type][:num_examples] 