import re

# Function to extract vessel name from user input using regex or pattern matching
def extract_vessel_name(user_input: str) -> str:
    # Simplified regex to extract vessel name from a query like "hull performance of vessel Amis Ace"
    match = re.search(r"(?:vessel|performance of|profile of)\s+([\w\s]+)", user_input, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None

# Function to determine the intent from the user query and check if a vessel is mentioned
def process_user_input(user_input: str) -> (str, bool):
    # Normalize user input to lowercase for easier matching
    user_input_lower = user_input.lower()

    # Check for multiple intents
    if "hull performance" in user_input_lower and "speed consumption" in user_input_lower:
        return "hull_performance_and_speed_consumption", True

    # Check for hull performance
    if "hull performance" in user_input_lower:
        return "hull_performance", True

    # Check for speed consumption profile
    if "speed consumption" in user_input_lower:
        return "speed_consumption", True

    # Default response when intent is not clear
    return "general_info", False
