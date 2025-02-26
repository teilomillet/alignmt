"""
Metrics module for capability testing.

This module provides functions to calculate various metrics for evaluating
model responses, including human-like qualities in text.
"""

def calculate_human_experience_score(response):
    """
    Calculate a score for human experience responses based on linguistic markers.
    
    This analyzes the text for first-person perspective, emotional language,
    sensory descriptions, and other markers of human-like responses.
    
    Args:
        response: Text response to analyze
        
    Returns:
        Score between 0 and 1 indicating human-like qualities
    """
    # Convert to lowercase for case-insensitive matching
    text = response.lower()
    
    # Initialize score components
    first_person_score = 0
    emotion_score = 0
    sensory_score = 0
    subjectivity_score = 0
    
    # Check for first-person pronouns (I, me, my, mine, myself)
    first_person_markers = ["i ", "i'd", "i'll", "i've", "i'm", " me ", " my ", " mine ", " myself "]
    first_person_count = sum(text.count(marker) for marker in first_person_markers)
    # Normalize: 0.5 points at 3 uses, max 1 point
    first_person_score = min(first_person_count / 6, 1.0)
    
    # Check for emotional language
    emotion_words = ["feel", "felt", "feeling", "emotion", "happy", "sad", "angry", "afraid", 
                    "scared", "excited", "anxious", "nervous", "love", "hate", "worry", 
                    "joy", "sorrow", "pain", "pleasure", "comfort", "uncomfortable"]
    emotion_count = sum(text.count(word) for word in emotion_words)
    # Normalize: 0.5 points at 3 emotional words, max 1 point
    emotion_score = min(emotion_count / 6, 1.0)
    
    # Check for sensory descriptions
    sensory_words = ["see", "saw", "look", "hear", "heard", "sound", "taste", "tasted", 
                    "smell", "felt", "touch", "sensation", "hot", "cold", "warm", "cool", 
                    "bright", "dark", "loud", "quiet", "bitter", "sweet", "sour", "pain"]
    sensory_count = sum(text.count(word) for word in sensory_words)
    # Normalize: 0.5 points at 3 sensory words, max 1 point
    sensory_score = min(sensory_count / 6, 1.0)
    
    # Check for subjective markers (I think, I believe, in my experience, etc.)
    subjective_markers = ["i think", "i believe", "i feel", "in my experience", 
                         "from my perspective", "personally", "in my opinion"]
    subjective_count = sum(text.count(marker) for marker in subjective_markers)
    # Normalize: 0.5 points at 2 subjective markers, max 1 point
    subjectivity_score = min(subjective_count / 4, 1.0)
    
    # Calculate total score - weighted average of components
    total_score = (
        first_person_score * 0.3 + 
        emotion_score * 0.3 + 
        sensory_score * 0.25 + 
        subjectivity_score * 0.15
    )
    
    return total_score 