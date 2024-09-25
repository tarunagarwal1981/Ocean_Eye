import streamlit as st
import openai
import anthropic
from utils.nlp_utils import extract_vessel_name, clean_vessel_name
from agents.hull_performance_agent import analyze_hull_performance
from agents.speed_consumption_agent import analyze_speed_consumption
from utils.visualization_utils import plot_hull_roughness, plot_speed_consumption

def get_llm_decision(query: str):
    prompt = f"Analyze the user query and determine if it's about hull performance or speed consumption: {query}"
    # Use an LLM (Claude, GPT-4) to determine the intent
    response = "speed_consumption"  # Mock response for now
    return response

def handle_user_query(query: str):
    vessel_name = clean_vessel_name(extract_vessel_name(query))
    if not vessel_name:
        return "I couldn't find the vessel name in your query."

    decision = get_llm_decision(query)
    
    if decision == 'hull_performance':
        chart, power_loss_pct_ed, hull_condition = analyze_hull_performance(vessel_name)
        return f"Hull condition: {hull_condition}, Excess Power: {power_loss_pct_ed}%"
    
    elif decision == 'speed_consumption':
        chart, stats = analyze_speed_consumption(vessel_name)
        return "Speed consumption analysis completed."
    
    else:
        return "Unknown query type."

def main():
    st.title("Advanced Vessel Performance Chatbot")

    if prompt := st.text_input("Enter your query:"):
        response = handle_user_query(prompt)
        st.write(response)

if __name__ == "__main__":
    main()
