import sys
import os
import streamlit as st
import openai

# Add the current directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from agents.hull_performance_agent import HullPerformanceAgent
from agents.speed_consumption_agent import SpeedConsumptionAgent
from utils.database_utils import get_db_engine
from utils.nlp_utils import get_llm_decision

# Initialize OpenAI API
def get_api_key():
    if 'openai' in st.secrets:
        return st.secrets['openai']['api_key']
    api_key = os.getenv('OPENAI_API_KEY')
    if api_key is None:
        raise ValueError("API key not found. Set OPENAI_API_KEY as an environment variable.")
    return api_key

openai.api_key = get_api_key()

def select_agents(query: str):
    decision = get_llm_decision(query)
    if decision['decision'] == 'hull_performance':
        return [HullPerformanceAgent()]
    elif decision['decision'] == 'speed_consumption':
        return [SpeedConsumptionAgent()]
    elif decision['decision'] == 'vessel_performance':
        return [HullPerformanceAgent(), SpeedConsumptionAgent()]
    else:
        return [HullPerformanceAgent(), SpeedConsumptionAgent()]  # Default to both

def generate_report(agents, query):
    report_sections = [agent.generate_report_section() for agent in agents]
    return "\n\n".join(report_sections)

def main():
    st.title("Vessel Performance Chatbot")

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What would you like to know about vessel performance?"):
        st.session_state.messages.append({"role": "human", "content": prompt})
        with st.chat_message("human"):
            st.markdown(prompt)

        agents = select_agents(prompt)
        engine = get_db_engine()
        
        combined_response = ""
        for agent in agents:
            response = agent.process_query(prompt, engine)
            combined_response += response + "\n\n"

        st.session_state.messages.append({"role": "assistant", "content": combined_response})
        with st.chat_message("assistant"):
            st.markdown(combined_response)

        for agent in agents:
            agent.display_charts(st)

        if "report" in prompt.lower():
            report = generate_report(agents, prompt)
            st.download_button(
                label="Download Report",
                data=report,
                file_name="vessel_performance_report.pdf",
                mime="application/pdf"
            )

if __name__ == "__main__":
    main()
