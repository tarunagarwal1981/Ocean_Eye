# File: streamlit_app.py

import streamlit as st
import openai
from utils.nlp_utils import process_user_input, extract_vessel_name
from modules.hull_performance import analyze_hull_performance
from modules.speed_consumption import analyze_speed_consumption
import os

def get_api_key():
    if 'openai' in st.secrets:
        return st.secrets['openai']['api_key']
    api_key = os.getenv('OPENAI_API_KEY')
    if api_key is None:
        raise ValueError("API key not found. Set OPENAI_API_KEY as an environment variable.")
    return api_key

openai.api_key = get_api_key()

def get_gpt_response(prompt):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an AI assistant specialized in vessel performance."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150,
            n=1,
            stop=None,
            temperature=0.5,
        )
        return response.choices[0].message['content'].strip()
    except Exception as e:
        st.error(f"Error in GPT response: {str(e)}")
        return "I'm sorry, I encountered an error while processing your request."

def handle_user_query(user_input: str) -> str:
    intent, vessel_name = process_user_input(user_input)

    if not vessel_name:
        return "I couldn't identify a vessel name in your query. Could you please provide a specific vessel name?"

    if intent == "hull_performance":
        return process_hull_performance(vessel_name)
    elif intent == "speed_consumption":
        return process_speed_consumption(vessel_name)
    elif intent == "vessel_performance":
        return process_combined_performance(vessel_name)
    else:
        return get_gpt_response(f"The user asked about {intent} for the vessel {vessel_name}. Provide a helpful response related to vessel performance.")

def process_hull_performance(vessel_name: str) -> str:
    try:
        fig, power_loss, hull_condition = analyze_hull_performance(vessel_name)
        if fig and power_loss is not None and hull_condition:
            st.pyplot(fig)
            return f"Here's the hull performance analysis for {vessel_name}. The average power loss is {power_loss:.2f}% and the hull condition is {hull_condition}."
        else:
            return f"Sorry, I couldn't find specific hull performance data for {vessel_name}."
    except Exception as e:
        st.error(f"An error occurred while processing hull performance: {str(e)}")
        return "I encountered an error while analyzing hull performance. Please try again or check if the vessel name is correct."

def process_speed_consumption(vessel_name: str) -> str:
    try:
        fig = analyze_speed_consumption(vessel_name)
        if fig:
            st.pyplot(fig)
            return f"Here's the speed consumption profile for {vessel_name}."
        else:
            return f"Sorry, I couldn't find specific speed consumption data for {vessel_name}."
    except Exception as e:
        st.error(f"An error occurred while processing speed consumption: {str(e)}")
        return "I encountered an error while analyzing speed consumption. Please try again or check if the vessel name is correct."

def process_combined_performance(vessel_name: str) -> str:
    try:
        hull_fig, power_loss, hull_condition = analyze_hull_performance(vessel_name)
        speed_fig = analyze_speed_consumption(vessel_name)

        if hull_fig:
            st.pyplot(hull_fig)
            st.write(f"Hull Performance - Average Power Loss: {power_loss:.2f}% | Hull Condition: {hull_condition}")
        else:
            st.write(f"Sorry, I couldn't find specific hull performance data for {vessel_name}.")

        if speed_fig:
            st.pyplot(speed_fig)
            st.write("Speed Consumption Profile")
        else:
            st.write(f"Sorry, I couldn't find specific speed consumption data for {vessel_name}.")

        return f"Combined performance analysis for {vessel_name}."
    except Exception as e:
        st.error(f"An error occurred while processing combined performance: {str(e)}")
        return "I encountered an error while analyzing combined performance. Please try again or check if the vessel name is correct."

def main():
    st.title("Vessel Performance Chatbot")

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What would you like to know about vessel performance?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        response = handle_user_query(prompt)

        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)

if __name__ == "__main__":
    main()

# File: modules/hull_performance.py

import matplotlib.pyplot as plt
from utils.database_utils import fetch_hull_performance_data

def analyze_hull_performance(vessel_name: str):
    try:
        data = fetch_hull_performance_data(vessel_name)
        
        if not data or len(data['dates']) == 0:
            return None, None, None

        # Create a line plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(data['dates'], data['power_loss'])
        ax.set_xlabel('Date')
        ax.set_ylabel('Power Loss (%)')
        ax.set_title(f'Hull Performance for {vessel_name}')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Calculate average power loss
        avg_power_loss = sum(data['power_loss']) / len(data['power_loss'])
        
        # Determine hull condition based on average power loss
        if avg_power_loss < 2.0:
            hull_condition = "Excellent"
        elif avg_power_loss < 2.5:
            hull_condition = "Good"
        elif avg_power_loss < 3.0:
            hull_condition = "Fair"
        else:
            hull_condition = "Poor"
        
        return fig, avg_power_loss, hull_condition
    except Exception as e:
        raise Exception(f"Error analyzing hull performance: {str(e)}")

# File: modules/speed_consumption.py

import matplotlib.pyplot as plt
from utils.database_utils import fetch_speed_consumption_data

def analyze_speed_consumption(vessel_name: str):
    try:
        data = fetch_speed_consumption_data(vessel_name)
        
        if not data or len(data['speeds']) == 0:
            return None

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(data['speeds'], data['consumption'])
        ax.set_xlabel('Speed (knots)')
        ax.set_ylabel('Fuel Consumption (tons/day)')
        ax.set_title(f'Speed-Consumption Profile for {vessel_name}')
        plt.tight_layout()
        
        return fig
    except Exception as e:
        raise Exception(f"Error analyzing speed consumption: {str(e)}")

# File: utils/nlp_utils.py

import spacy

# Load the English NLP model
nlp = spacy.load("en_core_web_sm")

def process_user_input(user_input: str):
    doc = nlp(user_input)
    
    # Extract intent
    intent = extract_intent(doc)
    
    # Extract vessel name
    vessel_name = extract_vessel_name(doc)
    
    return intent, vessel_name

def extract_intent(doc):
    # Define keywords for each intent
    intent_keywords = {
        "hull_performance": ["hull", "performance", "power loss"],
        "speed_consumption": ["speed", "consumption", "fuel"],
        "vessel_performance": ["vessel performance", "overall performance"],
    }
    
    # Check for intent keywords in the processed text
    for intent, keywords in intent_keywords.items():
        if any(keyword in doc.text.lower() for keyword in keywords):
            return intent
    
    return "general_info"

def extract_vessel_name(doc):
    for ent in doc.ents:
        if ent.label_ == "ORG" or ent.label_ == "PRODUCT":
            return ent.text
    return None

# Note: You would need to implement the database_utils.py file to handle the actual database queries.
# This file would contain functions like fetch_hull_performance_data and fetch_speed_consumption_data.
