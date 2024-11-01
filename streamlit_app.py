import streamlit as st
import openai
import os
import pandas as pd
import json
import re
from typing import Dict, Tuple, Optional

# Set page config first
st.set_page_config(
    page_title="VesselIQ",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Import all agents
from agents.hull_performance_agent import analyze_hull_performance
from agents.speed_consumption_agent import analyze_speed_consumption
from agents.vessel_score_agent import analyze_vessel_score
from agents.crew_score_agent import analyze_crew_score
from agents.position_tracking_agent import PositionTrackingAgent
from utils.database_utils import fetch_data_from_db
from utils.nlp_utils import clean_vessel_name

# Initialize position tracking agent
position_agent = PositionTrackingAgent()

# Constants
DECISION_PROMPT = """
You are an AI assistant specialized in vessel performance analysis. Based on the user's query:
1. Extract the vessel name (may appear after 'of', 'for', 'about').
2. Determine the required performance information type.

Decision rules:
- "vessel synopsis/summary/overview" → "vessel_synopsis"
- "vessel performance" or "hull and speed performance" → "combined_performance"
- "hull performance" → "hull_performance"
- "speed consumption" → "speed_consumption"
- "vessel score/rating/KPIs" → "vessel_score"
- "crew performance/score/rating" → "crew_score"
- "position/location/where is" → "position_tracking"

Output format:
{
    "vessel_name": "<cleaned_vessel_name>",
    "decision": "<decision_type>",
    "response_type": "concise"
}
"""

# CSS for styling
st.markdown(
    """
    <style>
        .block-container { max-width: 1200px; padding-top: 2rem; }
        .st-emotion-cache-1vbkxwb { max-width: 100% !important; }
        .stChatMessage { max-width: 100% !important; }
        .status-poor { color: #dc3545; font-weight: 500; }
        .status-average { color: #ffc107; font-weight: 500; }
        .status-good { color: #28a745; font-weight: 500; }
    </style>
    """,
    unsafe_allow_html=True
)

def init_session_state():
    """Initialize session state variables"""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'current_response' not in st.session_state:
        st.session_state.current_response = None
    if 'last_query' not in st.session_state:
        st.session_state.last_query = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = {}

def get_api_key() -> str:
    """Get OpenAI API key from secrets or environment variables."""
    if 'openai' in st.secrets:
        return st.secrets['openai']['api_key']
    api_key = os.getenv('OPENAI_API_KEY')
    if api_key is None:
        raise ValueError("OpenAI API key not found")
    return api_key

def get_llm_decision(query: str) -> Dict[str, str]:
    """Get decision from LLM about query type and vessel name."""
    try:
        openai.api_key = get_api_key()
        messages = [
            {"role": "system", "content": DECISION_PROMPT},
            {"role": "user", "content": query}
        ]
        
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=200,
            temperature=0.3
        )
        
        decision_data = json.loads(response.choices[0].message['content'].strip())
        
        if decision_data.get('vessel_name'):
            decision_data['vessel_name'] = clean_vessel_name(decision_data['vessel_name'])
        
        return decision_data
        
    except Exception as e:
        st.error(f"Error in LLM decision: {str(e)}")
        return {
            "vessel_name": None,
            "decision": "general_info",
            "response_type": "concise"
        }

def display_vessel_synopsis(vessel_name: str, data: dict):
    """Display vessel synopsis data."""
    st.subheader(f"Vessel Synopsis - {vessel_name.upper()}")
    
    # Position
    with st.expander("Current Position", expanded=True):
        if "position" in data:
            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown(data["position"]["analysis"])
            with col2:
                position_agent.show_position(vessel_name)
    
    # Hull Performance
    with st.expander("Hull Performance", expanded=True):
        if "hull" in data:
            if data["hull"].get("chart"):
                st.pyplot(data["hull"]["chart"])
            st.markdown(data["hull"]["analysis"])
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Power Loss", f"{data['hull']['power_loss']:.1f}%")
            with col2:
                st.metric("Hull Condition", data["hull"]["condition"])
    
    # Speed Consumption
    with st.expander("Speed Consumption", expanded=True):
        if "speed" in data:
            if data["speed"].get("chart"):
                st.pyplot(data["speed"]["chart"])
            st.markdown(data["speed"]["analysis"])
    
    # Vessel Score
    with st.expander("Vessel Performance", expanded=True):
        if "vessel_score" in data:
            st.markdown(data["vessel_score"]["analysis"])
            scores = data["vessel_score"]["scores"]
            cols = st.columns(len(scores))
            for i, (metric, value) in enumerate(scores.items()):
                with cols[i]:
                    st.metric(metric.replace('_', ' ').title(), f"{value:.1f}%")
    
    # Crew Score
    with st.expander("Crew Performance", expanded=True):
        if "crew" in data:
            st.markdown(data["crew"]["analysis"])
            scores = data["crew"]["scores"]
            cols = st.columns(len(scores))
            for i, (metric, value) in enumerate(scores.items()):
                with cols[i]:
                    st.metric(metric.replace('_', ' ').title(), f"{value:.1f}%")

def process_query(vessel_name: str, decision_type: str) -> Dict:
    """Process query based on decision type and return analysis results."""
    try:
        results = {}
        
        if decision_type in ["vessel_synopsis", "combined_performance"]:
            # Get hull performance data
            hull_analysis, power_loss, condition, hull_chart = analyze_hull_performance(vessel_name)
            results["hull"] = {
                "analysis": hull_analysis,
                "power_loss": power_loss,
                "condition": condition,
                "chart": hull_chart
            }
            
            # Get speed consumption data
            speed_analysis, speed_chart = analyze_speed_consumption(vessel_name)
            results["speed"] = {
                "analysis": speed_analysis,
                "chart": speed_chart
            }
            
            # Get vessel score data
            vessel_scores, vessel_analysis = analyze_vessel_score(vessel_name)
            results["vessel_score"] = {
                "scores": vessel_scores,
                "analysis": vessel_analysis
            }
            
            # Get crew data
            crew_scores, crew_analysis = analyze_crew_score(vessel_name)
            results["crew"] = {
                "scores": crew_scores,
                "analysis": crew_analysis
            }
            
            # Get position data
            position_analysis = position_agent.get_position_analysis(vessel_name)
            results["position"] = {
                "analysis": position_analysis
            }
            
        elif decision_type == "hull_performance":
            hull_analysis, power_loss, condition, hull_chart = analyze_hull_performance(vessel_name)
            results["hull"] = {
                "analysis": hull_analysis,
                "power_loss": power_loss,
                "condition": condition,
                "chart": hull_chart
            }
            
        elif decision_type == "speed_consumption":
            speed_analysis, speed_chart = analyze_speed_consumption(vessel_name)
            results["speed"] = {
                "analysis": speed_analysis,
                "chart": speed_chart
            }
            
        elif decision_type == "vessel_score":
            vessel_scores, vessel_analysis = analyze_vessel_score(vessel_name)
            results["vessel_score"] = {
                "scores": vessel_scores,
                "analysis": vessel_analysis
            }
            
        elif decision_type == "crew_score":
            crew_scores, crew_analysis = analyze_crew_score(vessel_name)
            results["crew"] = {
                "scores": crew_scores,
                "analysis": crew_analysis
            }
            
        elif decision_type == "position_tracking":
            position_analysis = position_agent.get_position_analysis(vessel_name)
            results["position"] = {
                "analysis": position_analysis
            }
        
        return results
        
    except Exception as e:
        st.error(f"Error processing analysis: {str(e)}")
        return {}

def display_specific_analysis(analysis_type: str, data: dict, vessel_name: str):
    """Display specific analysis results."""
    if analysis_type == "hull_performance" and "hull" in data:
        st.subheader(f"Hull Performance Analysis - {vessel_name}")
        if data["hull"].get("chart"):
            st.pyplot(data["hull"]["chart"])
        st.markdown(data["hull"]["analysis"])
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Power Loss", f"{data['hull']['power_loss']:.1f}%")
        with col2:
            st.metric("Hull Condition", data["hull"]["condition"])
            
    elif analysis_type == "speed_consumption" and "speed" in data:
        st.subheader(f"Speed Consumption Analysis - {vessel_name}")
        if data["speed"].get("chart"):
            st.pyplot(data["speed"]["chart"])
        st.markdown(data["speed"]["analysis"])
        
    elif analysis_type == "vessel_score" and "vessel_score" in data:
        st.subheader(f"Vessel Performance Analysis - {vessel_name}")
        st.markdown(data["vessel_score"]["analysis"])
        scores = data["vessel_score"]["scores"]
        cols = st.columns(len(scores))
        for i, (metric, value) in enumerate(scores.items()):
            with cols[i]:
                st.metric(metric.replace('_', ' ').title(), f"{value:.1f}%")
                
    elif analysis_type == "crew_score" and "crew" in data:
        st.subheader(f"Crew Performance Analysis - {vessel_name}")
        st.markdown(data["crew"]["analysis"])
        scores = data["crew"]["scores"]
        cols = st.columns(len(scores))
        for i, (metric, value) in enumerate(scores.items()):
            with cols[i]:
                st.metric(metric.replace('_', ' ').title(), f"{value:.1f}%")
                
    elif analysis_type == "position_tracking" and "position" in data:
        st.subheader(f"Current Position - {vessel_name}")
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown(data["position"]["analysis"])
        with col2:
            position_agent.show_position(vessel_name)

def main():
    init_session_state()
    
    st.title("VesselIQ - Smart Vessel Insights")
    st.markdown(
        "Ask me about vessel performance, speed consumption, crew performance, "
        "vessel position, or request a complete vessel synopsis!"
    )
    
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            
            # Display analysis results if available
            if "analysis_id" in message:
                analysis_data = st.session_state.analysis_results.get(message["analysis_id"])
                if analysis_data:
                    if analysis_data["type"] == "synopsis":
                        display_vessel_synopsis(
                            analysis_data["vessel_name"],
                            analysis_data["data"]
                        )
                    else:
                        display_specific_analysis(
                            analysis_data["type"],
                            analysis_data["data"],
                            analysis_data["vessel_name"]
                        )
    
    # Chat input
    if prompt := st.chat_input("What would you like to know about vessel performance?"):
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        # Get LLM decision
        with st.spinner("Analyzing your request..."):
            decision = get_llm_decision(prompt)
            
        if not decision.get("vessel_name"):
            response = "I couldn't identify a vessel name in your query. Could you please specify the vessel name?"
            st.session_state.chat_history.append({"role": "assistant", "content": response})
        else:
            # Process query
            with st.spinner("Processing analysis..."):
                analysis_results = process_query(decision["vessel_name"], decision["decision"])
                
                # Generate analysis ID
                analysis_id = f"{decision['vessel_name']}_{decision['decision']}_{len(st.session_state.analysis_results)}"
                
                # Store analysis results
                st.session_state.analysis_results[analysis_id] = {
                    "type": decision["decision"],
                    "vessel_name": decision["vessel_name"],
                    "data": analysis_results
                }
                
                # Add response to chat history
                response = f"Here's the {'analysis' if decision['decision'] != 'vessel_synopsis' else 'synopsis'} for {decision['vessel_name']}:"
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": response,
                    "analysis_id": analysis_id
                })
        
        # Force rerun to update display
        st.rerun()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
