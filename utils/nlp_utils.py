import re

def extract_vessel_name(user_input: str) -> str:
    # More comprehensive regex to extract vessel name
    patterns = [
        r"(?:vessel|ship|boat)\s+([\w\s-]+)",  # Matches "vessel Name"
        r"(?:of|for)\s+([\w\s-]+)",           # Matches "of Name" or "for Name"
        r"([\w\s-]+)(?:'s|\s+performance)",   # Matches "Name's" or "Name performance"
        r"\b([\w-]{2,}(?:\s+[\w-]+){0,3})\b"  # Matches 1-4 word phrases
    ]
    
    for pattern in patterns:
        match = re.search(pattern, user_input, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    
    # If no match found, return None
    return None

def process_user_input(user_input: str) -> (str, bool):
    user_input_lower = user_input.lower()
    
    # Define intent keywords
    intent_keywords = {
        "hull_performance": ["hull", "performance", "power loss", "roughness"],
        "speed_consumption": ["speed", "consumption", "fuel", "efficiency"],
        "hull_performance_and_speed_consumption": ["overall", "combined", "both"]
    }
    
    # Check for multiple intents
    intents_found = []
    for intent, keywords in intent_keywords.items():
        if any(keyword in user_input_lower for keyword in keywords):
            intents_found.append(intent)
    
    # Determine the final intent
    if len(intents_found) > 1 or "hull_performance_and_speed_consumption" in intents_found:
        return "hull_performance_and_speed_consumption", True
    elif len(intents_found) == 1:
        return intents_found[0], True
    else:
        return "general_info", False

# Function to clean and normalize vessel names
def clean_vessel_name(name: str) -> str:
    if name:
        # Remove any extra spaces and convert to title case
        return ' '.join(name.split()).title()
    return None

# Test the functions
if __name__ == "__main__":
    test_inputs = [
        "What's the hull performance of vessel Amis Ace?",
        "Show me the speed consumption for Trammo Marycam",
        "Give me overall performance data on Nordic Orion",
        "Tell me about the Stella Kosan",
        "What's the latest info on Brage R?"
    ]
    
    for input_text in test_inputs:
        intent, vessel_present = process_user_input(input_text)
        vessel_name = extract_vessel_name(input_text)
        cleaned_name = clean_vessel_name(vessel_name)
        print(f"Input: {input_text}")
        print(f"Intent: {intent}")
        print(f"Vessel Present: {vessel_present}")
        print(f"Extracted Vessel Name: {vessel_name}")
        print(f"Cleaned Vessel Name: {cleaned_name}")
        print("---")
