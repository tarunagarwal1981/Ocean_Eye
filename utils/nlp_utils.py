import openai

def get_llm_decision(query: str):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an AI assistant that categorizes queries about vessel performance."},
            {"role": "user", "content": f"Categorize this query: {query}"}
        ],
        temperature=0.3,
        max_tokens=100
    )
    decision = response.choices[0].message['content']
    
    if "hull" in decision.lower():
        return {"decision": "hull_performance"}
    elif "speed" in decision.lower() or "consumption" in decision.lower():
        return {"decision": "speed_consumption"}
    elif "vessel performance" in decision.lower():
        return {"decision": "vessel_performance"}
    else:
        return {"decision": "general"}

def get_llm_analysis(query: str, vessel_name: str, data_summary: str, agent_type: str):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": f"You are an AI assistant specialized in {agent_type} analysis for vessels."},
            {"role": "user", "content": f"Analyze this query about {vessel_name}: {query}\n\nData Summary: {data_summary}"}
        ],
        temperature=0.7,
        max_tokens=500
    )
    return response.choices[0].message['content']

def clean_vessel_name(name: str) -> str:
    # Remove any leading/trailing whitespace and convert to uppercase
    return ' '.join(name.strip().upper().split())

def extract_vessel_name(query: str) -> str:
    # Split the query into words
    words = query.lower().split()
    
    # Look for common patterns
    for i, word in enumerate(words):
        if word in ['of', 'for', 'vessel', 'ship']:
            # Return the rest of the words as the vessel name
            return ' '.join(words[i+1:])
    
    # If no pattern is found, return the last two words (assuming it might be a vessel name)
    return ' '.join(words[-2:])
