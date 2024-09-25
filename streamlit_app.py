import streamlit as st
import openai
import os
import pandas as pd
import json
import re
from typing import Dict
from agents.hull_performance_agent import analyze_hull_performance
from agents.speed_consumption_agent import analyze_speed_consumption
from utils.nlp_utils import clean_vessel_name

# LLM Prompts
DECISION_PROMPT = """
You are an AI assistant specialized in vessel performance analysis. The user will ask a query related to vessel performance. Based on the user's query, do two things:
1. Extract only the vessel name from the query. The vessel name may appear after the word 'of' (e.g., 'hull performance of Trammo Marycam' => 'Trammo Marycam').
2. Determine what type of performance information is needed to answer the user's query. The options are:
   - Hull performance
   - Speed consumption
   - Combined performance (both hull and speed)
   - General vessel information

Output your response as a JSON object with the following structure:
{
    "vessel_name": "<vessel_name>",
    "decision": "hull_performance" or "speed_consumption" or "combined_performance" or "general_info",
    "explanation": "Brief explanation of why you made this decision"
}
"""

# Function to get the OpenAI API key
def get_api_key():
    if 'openai' in st.secrets:
        return st.secrets['openai']['api_key']
    api_key = os.getenv('OPENAI_API_KEY')
    if api_key is None:
        raise ValueError("API key not found. Set OPENAI_API_KEY as an environment variable.")
    return api_key

# Initialize OpenAI API
openai.api_key = get_api_key()

# Fallback regex to extract vessel name in case LLM fails
def fallback_extract_vessel_name(query: str) -> str:
    match = re.search(r'of\s+(.+)', query, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return query  # If regex fails, return the original query

# Function to call ChatGPT for decision making using ChatCompletion
def get_llm_decision(query: str) -> Dict[str, str]:
    messages = [
        {"role": "system", "content": DECISION_PROMPT},
        {"role": "user", "content": query}
    ]
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=200,
            temperature=0.3
        )
        decision_text = response.choices[0].message['content'].strip()
        st.write(f"LLM Response: {decision_text}")  # Debugging output
        
        decision_data = json.loads(decision_text)
        
        # Fallback if the vessel name is not correctly extracted
        if "vessel_name" not in decision_data or decision_data['vessel_name'] is None or 'hull performance' in decision_data['vessel_name'].lower():
            decision_data['vessel_name'] = fallback_extract_vessel_name(query)
        
        return decision_data
    except openai.error.InvalidRequestError as e:
        st.error(f"InvalidRequestError: {str(e)}")
        return {
            "vessel_name": fallback_extract_vessel_name(query),
            "decision": "general_info",
            "explanation": "Invalid request. Defaulting to general info."
        }
    except Exception as e:
        st.error(f"Error in LLM decision: {str(e)}")
        return {
            "vessel_name": fallback_extract_vessel_name(query),
            "decision": "general_info",
            "explanation": "An error occurred. Defaulting to general info."
        }

# Function to handle user query and return analysis
def handle_user_query(query: str):
    # Get the decision and vessel name from the LLM (ChatGPT)
    llm_decision = get_llm_decision(query)
    
    vessel_name = llm_decision.get("vessel_name")
    if not vessel_name:
        return "I couldn't identify a vessel name in your query."

    st.write(f"Extracted Vessel Name: {vessel_name}")
    
    # Based on the decision, call the appropriate agent
    if llm_decision['decision'] == 'hull_performance':
        analysis, power_loss_pct, hull_condition = analyze_hull_performance(vessel_name)
        st.write(f"Hull performance analysis executed for {vessel_name}.")
        st.write(f"Analysis: {analysis}")
        
        if power_loss_pct is None or hull_condition is None:
            st.warning(f"Hull performance chart is not available for this vessel.")
        else:
            st.success(f"Average Power Loss: {power_loss_pct:.2f}%, Hull Condition: {hull_condition}")
    
    elif llm_decision['decision'] == 'speed_consumption':
        analysis = analyze_speed_consumption(vessel_name)
        st.write("Speed consumption analysis executed.")
    
    elif llm_decision['decision'] == 'combined_performance':
        # Call both hull and speed consumption agents and combine the analysis
        hull_analysis, _, _ = analyze_hull_performance(vessel_name)
        speed_analysis = analyze_speed_consumption(vessel_name)
        analysis = f"{hull_analysis}\n\n{speed_analysis}"
        st.write("Both hull performance and speed consumption analysis executed.")
    
    else:
        analysis = "The query seems to require general vessel information or is unclear. Please refine the query."
    
    # Display charts based on the decision
    display_charts(llm_decision['decision'], vessel_name)
    
    return analysis

# Function to display the charts based on the LLM's decision
def display_charts(decision: str, vessel_name: str):
    if decision in ["speed_consumption", "combined_performance"]:
        try:
            speed_chart, _ = analyze_speed_consumption(vessel_name)
            if speed_chart is not None and hasattr(speed_chart, 'savefig'):
                st.pyplot(speed_chart)
            else:
                st.warning("Speed consumption chart is not available for this vessel.")
        except Exception as e:
            st.error(f"An error occurred while generating the speed consumption chart: {str(e)}")
    
    if decision in ["hull_performance", "combined_performance"]:
        try:
            hull_chart, _, _ = analyze_hull_performance(vessel_name)
            if hull_chart is not None and hasattr(hull_chart, 'savefig'):
                st.pyplot(hull_chart)
            else:
                st.warning("Hull performance chart is not available for this vessel.")
        except Exception as e:
            st.error(f"An error occurred while generating the hull performance chart: {str(e)}")

# Main function for the Streamlit app
def main():
    st.title("Advanced Vessel Performance Chatbot (Powered by ChatGPT)")

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What would you like to know about vessel performance?"):
        st.session_state.messages.append({"role": "human", "content": prompt})
        with st.chat_message("human"):
            st.markdown(prompt)

        # Get the response from the chatbot
        analysis = handle_user_query(prompt)

        st.session_state.messages.append({"role": "assistant", "content": analysis})
        with st.chat_message("assistant"):
            st.markdown(analysis)

# Run the app
if __name__ == "__main__":
    main()
