import streamlit as st
from agents.vessel_performance_agent import VesselPerformanceAgent
from agents.hull_performance_agent import HullPerformanceAgent
from utils.database_utils import get_db_engine
from utils.nlp_utils import get_llm_decision

def select_agent(query: str):
    decision = get_llm_decision(query)
    if decision['decision'] == 'hull_performance':
        return HullPerformanceAgent()
    elif decision['decision'] == 'speed_consumption':
        return VesselPerformanceAgent()
    else:
        return VesselPerformanceAgent()  # Default to vessel performance

def main():
    st.title("Agentic Vessel Performance Chatbot")

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What would you like to know about vessel performance?"):
        st.session_state.messages.append({"role": "human", "content": prompt})
        with st.chat_message("human"):
            st.markdown(prompt)

        agent = select_agent(prompt)
        engine = get_db_engine()
        
        response = agent.process_query(prompt, engine)

        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)

        agent.display_charts(st)

if __name__ == "__main__":
    main()
