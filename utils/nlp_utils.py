import re

# Define vessel name patterns (can be expanded as needed)
VESSEL_NAME_PATTERN = r"\b(?:vessel\s+)?([A-Za-z0-9\s]+)\b"

def process_user_input(user_input: str):
    """
    Process user input to determine the intent (hull performance, speed consumption, etc.)
    and whether a vessel name is present.
    
    Args:
        user_input (str): The input query from the user.
    
    Returns:
        tuple: (intent, vessel_present)
    """
    user_input = user_input.lower()

    # Check for hull performance intent
    if "hull performance" in user_input or "hull condition" in user_input:
        return "hull_performance", True if extract_vessel_name(user_input) else False

    # Check for speed consumption intent
    elif "speed consumption" in user_input or "consumption profile" in user_input:
        return "speed_consumption", True if extract_vessel_name(user_input) else False

    # Check for combined vessel performance intent (hull + speed)
    elif "vessel performance" in user_input or "overall performance" in user_input:
        return "vessel_performance", True if extract_vessel_name(user_input) else False

    # Add more intents as needed
    else:
        return "general_info", False

def extract_vessel_name(user_input: str):
    """
    Extract vessel name from user input using regex patterns.
    
    Args:
        user_input (str): The input query from the user.
    
    Returns:
        str: The extracted vessel name (if found).
    """
    match = re.search(VESSEL_NAME_PATTERN, user_input, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None
