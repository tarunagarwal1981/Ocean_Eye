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
    # Implement vessel name cleaning logic
    return name.strip().upper()

def extract_vessel_name(query: str) -> str:
    # Implement vessel name extraction logic
    # This is a placeholder implementation
    words = query.split()
    for i in range(len(words) - 1):
        if words[i].lower() == "vessel" and i + 1 < len(words):
            return words[i + 1]
    return ""
