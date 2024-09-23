# streamlit_app.py

import streamlit as st
import openai
import os
from agents.agent_selector import AgentSelector
from utils.nlp_utils import extract_vessel_name, clean_vessel_name
from utils.database_utils import fetch_performance_data, fetch_speed_consumption_data

# Initialize OpenAI API
def get_api_key():
    if 'openai' in st.secrets:
        return st.secrets['openai']['api_key']
    api_key = os.getenv('OPENAI_API_KEY')
    if api_key is None:
        st.error("API key not found. Set OPENAI_API_KEY as an environment variable or add it to Streamlit secrets.")
        st.stop()
    return api_key

openai.api_key = get_api_key()

def main():
    st.title("Advanced Vessel Performance Chatbot")

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What would you like to know?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Handle the query and invoke appropriate agent(s)
        analysis = handle_user_query(prompt)

        st.session_state.messages.append({"role": "assistant", "content": analysis})
        with st.chat_message("assistant"):
            st.markdown(analysis)

def handle_user_query(query: str) -> str:
    vessel_name = clean_vessel_name(extract_vessel_name(query))
    if not vessel_name:
        return "I couldn't identify a vessel name in your query. Could you please provide a specific vessel name?"

    # Agent selection and execution
    agent_selector = AgentSelector()
    agents = agent_selector.select_agents(query)

    if not agents:
        return "I'm sorry, but I couldn't determine how to assist with your query."

    analysis_responses = []
    for agent in agents:
        response = agent.handle(query, vessel_name)
        analysis_responses.append(response)

    # Combine responses if multiple agents
    analysis = "\n\n".join(analysis_responses)

    # Generate report
    decision = agent_selector.get_decision(query)
    report_content, report_filename = generate_report(vessel_name, decision)
    if report_content:
        st.download_button(label="Download Report", data=report_content, file_name=report_filename, mime='text/csv')

    return analysis

def generate_report(vessel_name: str, decision: str):
    from io import StringIO
    if decision == "hull_performance":
        performance_data = fetch_performance_data(vessel_name)
        if performance_data.empty:
            return None, None
        csv_buffer = StringIO()
        performance_data.to_csv(csv_buffer, index=False)
        return csv_buffer.getvalue(), f"{vessel_name}_hull_performance.csv"

    elif decision == "speed_consumption":
        speed_data = fetch_speed_consumption_data(vessel_name)
        if speed_data.empty:
            return None, None
        csv_buffer = StringIO()
        speed_data.to_csv(csv_buffer, index=False)
        return csv_buffer.getvalue(), f"{vessel_name}_speed_consumption.csv"

    elif decision == "vessel_performance":
        performance_data = fetch_performance_data(vessel_name)
        speed_data = fetch_speed_consumption_data(vessel_name)
        if performance_data.empty and speed_data.empty:
            return None, None
        combined_data = pd.concat([performance_data, speed_data], axis=1)
        csv_buffer = StringIO()
        combined_data.to_csv(csv_buffer, index=False)
        return csv_buffer.getvalue(), f"{vessel_name}_vessel_performance.csv"

    return None, None

if __name__ == "__main__":
    main()
