import streamlit as st
import openai
from utils.nlp_utils import process_user_input, extract_vessel_name
from modules.hull_performance import analyze_hull_performance
from modules.speed_consumption import analyze_speed_consumption
import os

# Function to get OpenAI API key
def get_api_key():
    if 'openai' in st.secrets:
        return st.secrets['openai']['api_key']
    api_key = os.getenv('OPENAI_API_KEY')
    if api_key is None:
        raise ValueError("API key not found. Set OPENAI_API_KEY as an environment variable.")
    return api_key

# Initialize OpenAI API key
openai.api_key = get_api_key()

# Function to get GPT-3 response
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

# Function to handle user queries
def handle_user_query(user_input: str) -> str:
    intent, vessel_present = process_user_input(user_input)
    vessel_name = extract_vessel_name(user_input)

    if not vessel_name:
        return "I couldn't identify a vessel name in your query. Could you please provide a specific vessel name?"

    if intent == "hull_performance":
        return process_hull_performance(vessel_name)
    elif intent == "speed_consumption":
        return process_speed_consumption(vessel_name)
    elif intent == "hull_performance_and_speed_consumption":
        return process_combined_performance(vessel_name)
    else:
        return get_gpt_response(f"The user asked about {intent} for the vessel {vessel_name}. Provide a helpful response related to vessel performance.")

# Function to process hull performance
def process_hull_performance(vessel_name: str) -> str:
    try:
        chart, power_loss, hull_condition = analyze_hull_performance(vessel_name)
        if chart and power_loss is not None and hull_condition:
            st.pyplot(chart)
            return f"Here's the hull performance analysis for {vessel_name}. The power loss due to hull roughness is {power_loss:.2f}% and the hull condition is {hull_condition}."
        else:
            return f"Sorry, I couldn't find specific hull performance data for {vessel_name}."
    except Exception as e:
        st.error(f"An error occurred while processing hull performance: {str(e)}")
        return "I encountered an error while analyzing hull performance. Please try again or check if the vessel name is correct."

# Function to process speed consumption
def process_speed_consumption(vessel_name: str) -> str:
    try:
        chart = analyze_speed_consumption(vessel_name)
        if chart:
            st.pyplot(chart)
            return f"Here's the speed consumption profile for {vessel_name}. The chart shows the relationship between speed and fuel consumption for both laden and ballast conditions over the last 6 months."
        else:
            return f"Sorry, I couldn't find specific speed consumption data for {vessel_name}."
    except Exception as e:
        st.error(f"An error occurred while processing speed consumption: {str(e)}")
        return "I encountered an error while analyzing speed consumption. Please try again or check if the vessel name is correct."

# Function to process combined performance
def process_combined_performance(vessel_name: str) -> str:
    try:
        hull_chart, power_loss, hull_condition = analyze_hull_performance(vessel_name)
        speed_chart = analyze_speed_consumption(vessel_name)

        if hull_chart:
            st.pyplot(hull_chart)
            st.write(f"Hull Performance - Power Loss due to Hull Roughness: {power_loss:.2f}% | Hull Condition: {hull_condition}")
        else:
            st.write(f"Sorry, I couldn't find specific hull performance data for {vessel_name}.")

        if speed_chart:
            st.pyplot(speed_chart)
            st.write("Speed Consumption Profile")
        else:
            st.write(f"Sorry, I couldn't find specific speed consumption data for {vessel_name}.")

        return f"Combined performance analysis for {vessel_name} is shown above."
    except Exception as e:
        st.error(f"An error occurred while processing combined performance: {str(e)}")
        return "I encountered an error while analyzing combined performance. Please try again or check if the vessel name is correct."

# Main function for Streamlit Chatbot UI
def main():
    st.title("Vessel Performance Chatbot")

    # Initialize chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("What would you like to know about vessel performance?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Process user input and get response
        response = handle_user_query(prompt)

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)

if __name__ == "__main__":
    main()
