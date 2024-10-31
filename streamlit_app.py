# app.py

import streamlit as st
import openai
import os
import pandas as pd
import json
import re
from typing import Dict, Tuple, Optional

# Import all agents
from agents.hull_performance_agent import analyze_hull_performance
from agents.speed_consumption_agent import analyze_speed_consumption
from agents.vessel_score_agent import analyze_vessel_score
from agents.crew_score_agent import analyze_crew_score
from agents.position_tracking_agent import PositionTrackingAgent
from utils.database_utils import fetch_data_from_db
from utils.nlp_utils import clean_vessel_name

# Initialize agents
position_agent = PositionTrackingAgent()

# LLM Decision Prompt
DECISION_PROMPT = """
You are an AI assistant specialized in vessel performance analysis. The user will ask a query related to vessel performance. Based on the user's query, do two things:
1. Extract only the vessel name from the query. The vessel name may appear after the word 'of' (e.g., 'hull performance of Trammo Marycam' => 'Trammo Marycam').
2. Determine what type of performance information is needed to answer the user's query.

Choose the decision based on these rules:
- If the user asks for "vessel synopsis", "vessel summary", or "vessel overview", return "vessel_synopsis"
- If the user asks for "vessel performance" or a combination of "hull and speed performance," return "combined_performance"
- If the user asks only about "hull performance" or "hull and propeller performance," return "hull_performance"
- If the user asks only about "speed consumption," return "speed_consumption"
- If the user asks about "vessel score", "vessel rating", or "vessel KPIs", return "vessel_score"
- If the user asks about "crew performance", "crew score", or "crew rating", return "crew_score"
- If the user asks about "position", "location", or "where is", return "position_tracking"

Output format:
{
    "vessel_name": "<vessel_name>",
    "decision": "<decision_type>",
    "response_type": "concise" or "detailed"
}
"""

def get_api_key():
    """Get OpenAI API key from secrets or environment variables."""
    if 'openai' in st.secrets:
        return st.secrets['openai']['api_key']
    api_key = os.getenv('OPENAI_API_KEY')
    if api_key is None:
        raise ValueError("API key not found. Set OPENAI_API_KEY as an environment variable.")
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
        
        decision_text = response.choices[0].message['content'].strip()
        decision_data = json.loads(decision_text)
        
        # Clean vessel name
        if decision_data.get('vessel_name'):
            decision_data['vessel_name'] = clean_vessel_name(decision_data['vessel_name'])
        else:
            # Extract vessel name using patterns
            patterns = [
                r'(?:of|for|about)\s+"?([^"]+?)"?\s*(?:\?|$)',
                r'"([^"]+)"',
                r'vessel\s+([^\s?]+(?:\s+[^\s?]+)*)',
                r'mv\s+([^\s?]+(?:\s+[^\s?]+)*)'
            ]
            
            for pattern in patterns:
                match = re.search(pattern, query, re.IGNORECASE)
                if match:
                    decision_data['vessel_name'] = clean_vessel_name(match.group(1))
                    break
        
        return decision_data
        
    except Exception as e:
        st.error(f"Error in LLM decision: {str(e)}")
        return {
            "vessel_name": None,
            "decision": "general_info",
            "response_type": "concise"
        }

def show_vessel_synopsis(vessel_name: str) -> dict:
    """Generate comprehensive vessel synopsis data."""
    response_data = {}
    try:
        # Get all analysis data
        hull_analysis, power_loss, hull_condition, hull_chart = analyze_hull_performance(vessel_name)
        speed_analysis, speed_charts = analyze_speed_consumption(vessel_name)
        vessel_scores, vessel_analysis = analyze_vessel_score(vessel_name)
        crew_scores, crew_analysis = analyze_crew_score(vessel_name)
        
        # Compile all data
        response_data = {
            "charts": [],
            "analyses": {},
            "scores": {},
            "title": f"Vessel Synopsis - {vessel_name.upper()}"
        }
        
        # Position data
        position_info = position_agent.get_position_analysis(vessel_name)
        response_data["analyses"]["position"] = position_info
        
        # Hull performance data
        if hull_chart:
            response_data["charts"].append(("hull", hull_chart))
            response_data["analyses"]["hull"] = hull_analysis
            response_data["scores"]["hull_power_loss"] = power_loss
            response_data["scores"]["hull_condition"] = hull_condition
        
        # Speed consumption data
        if speed_charts:
            response_data["charts"].append(("speed", speed_charts))
            response_data["analyses"]["speed"] = speed_analysis
        
        # Vessel scores
        if vessel_scores:
            response_data["scores"]["vessel"] = vessel_scores
            response_data["analyses"]["vessel"] = vessel_analysis
        
        # Crew scores
        if crew_scores:
            response_data["scores"]["crew"] = crew_scores
            response_data["analyses"]["crew"] = crew_analysis
        
        return response_data
        
    except Exception as e:
        st.error(f"Error generating vessel synopsis: {str(e)}")
        return None

def display_synopsis(response_data: dict):
    """Display stored synopsis data."""
    if not response_data:
        return
    
    st.header(response_data["title"])
    
    # Position information
    with st.expander("Last Reported Position", expanded=True):
        if "position" in response_data["analyses"]:
            st.markdown(response_data["analyses"]["position"])
            position_agent.show_position(response_data["title"].split('-')[1].strip())
        else:
            st.warning("No position data available")
    
    # Hull performance
    with st.expander("Hull Performance", expanded=True):
        hull_chart = next((chart for name, chart in response_data["charts"] if name == "hull"), None)
        if hull_chart:
            st.pyplot(hull_chart)
            st.markdown(response_data["analyses"]["hull"])
        else:
            st.warning("No hull performance data available")
    
    # Speed consumption
    with st.expander("Speed Consumption Profile", expanded=True):
        speed_chart = next((chart for name, chart in response_data["charts"] if name == "speed"), None)
        if speed_chart:
            st.pyplot(speed_chart)
            st.markdown(response_data["analyses"]["speed"])
        else:
            st.warning("No speed consumption data available")
    
    # Vessel scores
    with st.expander("Vessel Performance Scores", expanded=True):
        if "vessel" in response_data["scores"]:
            st.markdown(response_data["analyses"]["vessel"])
            cols = st.columns(3)
            for i, (metric, value) in enumerate(response_data["scores"]["vessel"].items()):
                with cols[i % 3]:
                    st.metric(metric.replace('_', ' ').title(), f"{value:.1f}%")
        else:
            st.warning("No vessel score data available")
    
    # Crew scores
    with st.expander("Crew Performance Scores", expanded=True):
        if "crew" in response_data["scores"]:
            st.markdown(response_data["analyses"]["crew"])
            cols = st.columns(3)
            for i, (metric, value) in enumerate(response_data["scores"]["crew"].items()):
                with cols[i % 3]:
                    st.metric(metric.replace('_', ' ').title(), f"{value:.1f}%")
        else:
            st.warning("No crew score data available")

def process_specific_analysis(decision_type: str, vessel_name: str) -> dict:
    """Process specific analysis types and return response data."""
    try:
        if decision_type == "hull_performance":
            analysis, power_loss, condition, chart = analyze_hull_performance(vessel_name)
            return {
                "content": f"Hull Performance Analysis for {vessel_name}",
                "response_data": {
                    "analysis": analysis,
                    "charts": [("hull", chart)] if chart else [],
                    "scores": {"power_loss": power_loss, "condition": condition}
                },
                "type": "hull_performance"
            }
            
        elif decision_type == "speed_consumption":
            analysis, charts = analyze_speed_consumption(vessel_name)
            return {
                "content": f"Speed Consumption Analysis for {vessel_name}",
                "response_data": {
                    "analysis": analysis,
                    "charts": [("speed", charts)] if charts else []
                },
                "type": "speed_consumption"
            }
            
        elif decision_type == "vessel_score":
            scores, analysis = analyze_vessel_score(vessel_name)
            return {
                "content": f"Vessel Score Analysis for {vessel_name}",
                "response_data": {
                    "analysis": analysis,
                    "scores": scores
                },
                "type": "vessel_score"
            }
            
        elif decision_type == "crew_score":
            scores, analysis = analyze_crew_score(vessel_name)
            return {
                "content": f"Crew Performance Analysis for {vessel_name}",
                "response_data": {
                    "analysis": analysis,
                    "scores": scores
                },
                "type": "crew_score"
            }
            
        elif decision_type == "position_tracking":
            analysis = position_agent.get_position_analysis(vessel_name)
            return {
                "content": f"Current Position of {vessel_name}",
                "response_data": {
                    "analysis": analysis,
                    "position_data": True
                },
                "type": "position"
            }
            
    except Exception as e:
        return {
            "content": f"Error processing {decision_type}: {str(e)}",
            "type": "error"
        }

def handle_user_query(query: str) -> dict:
    """Process user query and return appropriate response."""
    decision_data = get_llm_decision(query)
    vessel_name = decision_data.get("vessel_name")
    decision_type = decision_data.get("decision", "general_info")
    
    if not vessel_name:
        return {
            "content": "I couldn't identify a vessel name in your query. Could you please specify the vessel name?",
            "type": "warning"
        }
    
    try:
        if decision_type == "vessel_synopsis":
            response_data = show_vessel_synopsis(vessel_name)
            if response_data:
                return {
                    "content": f"Here's the complete synopsis for {vessel_name}:",
                    "response_data": response_data,
                    "type": "synopsis"
                }
                
        elif decision_type in ["hull_performance", "speed_consumption", 
                             "vessel_score", "crew_score", "position_tracking"]:
            return process_specific_analysis(decision_type, vessel_name)
            
        else:
            return {
                "content": """I understand you're asking about vessel information. Please specify what aspect you'd like to know about:
                - Hull performance
                - Speed consumption
                - Vessel score
                - Crew performance
                - Current position
                - Complete vessel synopsis""",
                "type": "info"
            }
            
    except Exception as e:
        return {
            "content": f"Error processing query: {str(e)}",
            "type": "error"
        }

def display_response(response: dict):
    """Display the appropriate response based on type."""
    if response["type"] == "synopsis":
        display_synopsis(response["response_data"])
    
    elif response["type"] in ["hull_performance", "speed_consumption"]:
        st.markdown(response["content"])
        for chart_name, chart in response["response_data"].get("charts", []):
            st.pyplot(chart)
        st.markdown(response["response_data"]["analysis"])
    
    elif response["type"] in ["vessel_score", "crew_score"]:
        st.markdown(response["content"])
        st.markdown(response["response_data"]["analysis"])
        cols = st.columns(3)
        for i, (metric, value) in enumerate(response["response_data"]["scores"].items()):
            with cols[i % 3]:
                st.metric(metric.replace('_', ' ').title(), f"{value:.1f}%")
    
    elif response["type"] == "position":
        st.markdown(response["content"])
        st.markdown(response["response_data"]["analysis"])
        vessel_name = response["content"].split("of ")[-1]
        position_agent.show_position(vessel_name)

def main():
    # Set page config
    st.set_page_config(layout="wide", page_title="VesselIQ")
    
    # Add custom CSS
    st.markdown(
        """
        <style>
            .block-container { max-width: 1200px; padding-top: 2rem; }
            .stTitle { font-size: 2rem; font-weight: bold; margin-bottom: 1rem; }
            .stMarkdown { font-size: 1.1rem; }
            .stChatFloatingInputContainer {
                max-width: 80% !important;
                margin-left: auto !important;
                margin-right: auto !important;
            }
            .stChatMessage {
                max-width: 100% !important;
                padding: 1rem !important;
            }
            .status-poor { color: #dc3545; font-weight: 500; }
            .status-average { color: #ffc107; font-weight: 500; }
            .status-good { color: #28a745; font-weight: 500; }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    # Application header
    st.title("VesselIQ - Smart Vessel Insights")
    st.markdown(
        "Ask me about vessel performance, speed consumption, crew performance, "
        "vessel position, or request a complete vessel synopsis!"
    )
    
    # Initialize session state
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "response_data" in message:
                display_response(message)
    
    
    # Handle user input
    if prompt := st.chat_input("What would you like to know about vessel performance?"):
        # Add user message
        st.session_state.messages.append({"role": "human", "content": prompt})
        with st.chat_message("human"):
            st.markdown(prompt)
        
        # Process query and add response
        response = handle_user_query(prompt)
        
        # Display assistant response
        with st.chat_message("assistant"):
            st.markdown(response["content"])
            if "response_data" in response:
                display_response(response)
        
        # Store response in chat history
        st.session_state.messages.append({
            "role": "assistant",
            "content": response["content"],
            "type": response.get("type"),
            "response_data": response.get("response_data")
        })
        
        # Rerun the app to update the display
        st.rerun()

def initialize_page():
    """Initialize the page with required styles and configurations."""
    # Set page config
    st.set_page_config(
        page_title="VesselIQ",
        page_icon="🚢",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Add custom CSS
    st.markdown(
        """
        <style>
            .block-container { max-width: 1200px; padding-top: 2rem; }
            .stTitle { font-size: 2rem; font-weight: bold; margin-bottom: 1rem; }
            .stMarkdown { font-size: 1.1rem; }
            .stChatFloatingInputContainer {
                max-width: 80% !important;
                margin-left: auto !important;
                margin-right: auto !important;
            }
            .stChatMessage {
                max-width: 100% !important;
                padding: 1rem !important;
            }
            .status-poor { color: #dc3545; font-weight: 500; }
            .status-average { color: #ffc107; font-weight: 500; }
            .status-good { color: #28a745; font-weight: 500; }
            
            /* Additional styling for better visual hierarchy */
            h1 { margin-bottom: 2rem; }
            .stExpander { margin-bottom: 1rem; }
            .stMetric { background-color: #f8f9fa; padding: 1rem; border-radius: 0.5rem; }
            
            /* Improve chart visibility */
            .stPlot { background-color: white; padding: 1rem; border-radius: 0.5rem; }
            
            /* Better spacing for expandable sections */
            .streamlit-expanderHeader {
                background-color: #f8f9fa;
                border-radius: 0.5rem;
                margin-bottom: 0.5rem;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    try:
        initialize_page()
        main()
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        if "messages" in st.session_state:
            # Clear messages if there's an error to prevent error loop
            st.session_state.messages = []
