import re
from typing import Tuple, Optional

# Dictionary of intents and their associated keywords
INTENT_KEYWORDS = {
    "hull_performance": ["hull", "performance", "roughness", "power loss"],
    "fuel_efficiency": ["fuel", "efficiency", "consumption", "usage"],
    "speed_performance": ["speed", "velocity", "knots"],
    "general_info": ["info", "information", "details", "specs", "specifications"]
}

def recognize_intent(user_input: str) -> str:
    """
    Recognize the intent of the user's input based on keywords.
    
    Args:
    user_input (str): The user's input text

    Returns:
    str: The recognized intent
    """
    user_input = user_input.lower()
    for intent, keywords in INTENT_KEYWORDS.items():
        if any(keyword in user_input for keyword in keywords):
            return intent
    return "unknown"

def extract_vessel_name(user_input: str) -> Optional[str]:
    """
    Extract the vessel name from the user's input.
    
    Args:
    user_input (str): The user's input text

    Returns:
    Optional[str]: The extracted vessel name, or None if not found
    """
    # Pattern to match "vessel name" or "ship name" followed by any word characters
    pattern = r"(?:vessel|ship)\s+name\s+is\s+(\w+)"
    match = re.search(pattern, user_input, re.IGNORECASE)
    
    if match:
        return match.group(1)
    
    # If the above pattern doesn't match, try to find any capitalized words
    # that might represent a vessel name
    words = user_input.split()
    capitalized_words = [word for word in words if word.istitle()]
    
    if capitalized_words:
        return capitalized_words[0]  # Return the first capitalized word as the vessel name
    
    return None

def process_user_input(user_input: str) -> Tuple[str, Optional[str]]:
    """
    Process the user's input to recognize intent and extract vessel name.
    
    Args:
    user_input (str): The user's input text

    Returns:
    Tuple[str, Optional[str]]: A tuple containing the recognized intent and extracted vessel name (if any)
    """
    intent = recognize_intent(user_input)
    vessel_name = extract_vessel_name(user_input)
    return intent, vessel_name

# Example usage
if __name__ == "__main__":
    test_inputs = [
        "What's the hull performance of vessel name is Serenity?",
        "Show me the fuel efficiency for the ship Atlantic Star",
        "I need information about the speed of the Majestic Princess",
        "Give me general info on the Queen Mary 2"
    ]
    
    for input_text in test_inputs:
        intent, vessel = process_user_input(input_text)
        print(f"Input: {input_text}")
        print(f"Recognized Intent: {intent}")
        print(f"Extracted Vessel Name: {vessel}")
        print()
