import re
from typing import Tuple, Optional

INTENT_KEYWORDS = {
    "hull_performance": ["hull", "performance", "roughness", "power loss"],
    "fuel_efficiency": ["fuel", "efficiency", "consumption", "usage"],
    "speed_performance": ["speed", "velocity", "knots"],
    "general_info": ["info", "information", "details", "specs", "specifications"]
}

def recognize_intent(user_input: str) -> str:
    user_input = user_input.lower()
    for intent, keywords in INTENT_KEYWORDS.items():
        if any(keyword in user_input for keyword in keywords):
            return intent
    return "unknown"

def extract_vessel_name(user_input: str) -> Optional[str]:
    patterns = [
        r"(?:of|for)\s+the\s+(\w+(?:\s+\w+)?)",
        r"(\w+(?:\s+\w+)?)'s",
        r"vessel\s+(?:name\s+is\s+)?(\w+(?:\s+\w+)?)",
        r"ship\s+(?:name\s+is\s+)?(\w+(?:\s+\w+)?)"
    ]
    
    for pattern in patterns:
        match = re.search(pattern, user_input, re.IGNORECASE)
        if match:
            return match.group(1)
    
    words = user_input.split()
    capitalized_words = [word for word in words if word.istitle()]
    
    if capitalized_words:
        return ' '.join(capitalized_words)
    
    return None

def has_vessel_name(user_input: str) -> bool:
    return extract_vessel_name(user_input) is not None

def process_user_input(user_input: str) -> Tuple[str, bool]:
    intent = recognize_intent(user_input)
    vessel_present = has_vessel_name(user_input)
    return intent, vessel_present
