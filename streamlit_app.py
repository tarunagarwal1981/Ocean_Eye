import streamlit as st
import openai
import os
import pandas as pd
import json
import re
from typing import Dict, Tuple, Optional
import time

# Set page config
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

# CSS Styling
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
        h1 { margin-bottom: 2rem; }
        .stExpander { margin-bottom: 1rem; }
        .stMetric { background-color: #f8f9fa; padding: 1rem; border-radius: 0.5rem; }
        .stPlot { background-color: white; padding: 1rem; border-radius: 0.5rem; }
        .streamlit-expanderHeader {
            background-color: #f8f9fa;
            border-radius: 0.5rem;
            margin-bottom: 0.5rem;
        }
    </style>
    """,
    unsafe_allow_html=True
)

class SessionState:
    """Handle session state initialization and management."""
    @staticmethod
    def initialize():
        if 'initialized' not in st.session_state:
            st.session_state.initialized = True
            st.session_state.messages = []
            st.session_state.current_vessel = None
            st.session_state.current_analysis = None
            st.session_state.data_cache = {}
            st.session_state.processing = False
            st.session_state.error = None
    
    @staticmethod
    def cache_data(key: str, data: dict):
        """Store data in cache with timestamp."""
        st.session_state.data_cache[key] = {
            'data': data,
            'timestamp': time.time()
        }
    
    @staticmethod
    def get_cached_data(key: str, max_age: int = 300) -> Optional[dict]:
        """Retrieve cached data if not expired."""
        if key in st.session_state.data_cache:
            cache_entry = st.session_state.data_cache[key]
            if time.time() - cache_entry['timestamp'] < max_age:
                return cache_entry['data']
        return None
    
    @staticmethod
    def clear_old_cache(max_entries: int = 50):
        """Clear old cache entries if cache gets too large."""
        if len(st.session_state.data_cache) > max_entries:
            sorted_cache = sorted(
                st.session_state.data_cache.items(),
                key=lambda x: x[1]['timestamp']
            )
            for key, _ in sorted_cache[:-max_entries]:
                del st.session_state.data_cache[key]

def get_api_key() -> str:
    """Get OpenAI API key from secrets or environment variables."""
    if 'openai' in st.secrets:
        return st.secrets['openai']['api_key']
    api_key = os.getenv('OPENAI_API_KEY')
    if api_key is None:
        raise ValueError("OpenAI API key not found in environment variables or secrets.")
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
        
        # Clean and verify vessel name
        if decision_data.get('vessel_name'):
            decision_data['vessel_name'] = clean_vessel_name(decision_data['vessel_name'])
        else:
            # Fallback pattern matching
            patterns = [
                r'(?:of|for|about)\s+"?([^"]+?)"?\s*(?:\?|$)',
                r'"([^"]+)"',
                r'vessel\s+([^\s?]+(?:\s+[^\s?]+)*)',
                r'mv\s+([^\s?]+(?:\s+[^\s?]+)*)'
            ]
            
            for pattern in patterns:
                if match := re.search(pattern, query, re.IGNORECASE):
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
    """Generate comprehensive vessel synopsis data with caching."""
    cache_key = f"synopsis_{vessel_name}"
    cached_data = SessionState.get_cached_data(cache_key)
    if cached_data:
        return cached_data
    
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
        
        # Store in cache
        SessionState.cache_data(cache_key, response_data)
        return response_data
        
    except Exception as e:
        st.error(f"Error generating vessel synopsis: {str(e)}")
        return None

def process_specific_analysis(decision_type: str, vessel_name: str) -> dict:
    """Process specific analysis types and return response data with caching."""
    cache_key = f"{decision_type}_{vessel_name}"
    cached_data = SessionState.get_cached_data(cache_key)
    if cached_data:
        return cached_data
    
    try:
        if decision_type == "hull_performance":
            analysis, power_loss, condition, chart = analyze_hull_performance(vessel_name)
            response = {
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
            response = {
                "content": f"Speed Consumption Analysis for {vessel_name}",
                "response_data": {
                    "analysis": analysis,
                    "charts": [("speed", charts)] if charts else []
                },
                "type": "speed_consumption"
            }
            
        elif decision_type == "vessel_score":
            scores, analysis = analyze_vessel_score(vessel_name)
            response = {
                "content": f"Vessel Score Analysis for {vessel_name}",
                "response_data": {
                    "analysis": analysis,
                    "scores": scores
                },
                "type": "vessel_score"
            }
            
        elif decision_type == "crew_score":
            scores, analysis = analyze_crew_score(vessel_name)
            response = {
                "content": f"Crew Performance Analysis for {vessel_name}",
                "response_data": {
                    "analysis": analysis,
                    "scores": scores
                },
                "type": "crew_score"
            }
            
        elif decision_type == "position_tracking":
            analysis = position_agent.get_position_analysis(vessel_name)
            response = {
                "content": f"Current Position of {vessel_name}",
                "response_data": {
                    "analysis": analysis,
                    "position_data": True
                },
                "type": "position"
            }
        
        # Store in cache
        SessionState.cache_data(cache_key, response)
        return response
            
    except Exception as e:
        return {
            "content": f"Error processing {decision_type}: {str(e)}",
            "type": "error"
        }

def display_response(response: dict):
    """Display the appropriate response based on type with persistent state."""
    if response["type"] == "synopsis":
        display_synopsis(response["response_data"])
        
    elif response["type"] in ["hull_performance", "speed_consumption"]:
        container = st.container()
        with container:
            st.markdown(response["content"])
            for chart_name, chart in response["response_data"].get("charts", []):
                st.pyplot(chart)
            st.markdown(response["response_data"]["analysis"])
            
            if "scores" in response["response_data"]:
                cols = st.columns(len(response["response_data"]["scores"]))
                for i, (metric, value) in enumerate(response["response_data"]["scores"].items()):
                    with cols[i]:
                        if isinstance(value, float):
                            st.metric(metric.replace('_', ' ').title(), f"{value:.1f}%")
                        else:
                            st.metric(metric.replace('_', ' ').title(), value)
    
    elif response["type"] in ["vessel_score", "crew_score"]:
        container = st.container()
        with container:
            st.markdown(response["content"])
            st.markdown(response["response_data"]["analysis"])
            cols = st.columns(3)
            for i, (metric, value) in enumerate(response["response_data"]["scores"].items()):
                with cols[i % 3]:
                    st.metric(metric.replace('_', ' ').title(), f"{value:.1f}%")
    
    elif response["type"] == "position":
        container = st.container()
        with container:
            st.markdown(response["content"])
            st.markdown(response["response_data"]["analysis"])
            vessel_name = response["content"].split("of ")[-1]
            position_agent.show_position(vessel_name)

def display_synopsis(response_data: dict):
    """Display stored synopsis data with persistent state."""
    if not response_data:
        return
    
    st.header(response_data["title"])
    
    # Position information
    with st.expander("Last Reported Position", expanded=True):
        if "position" in response_data["analyses"]:
            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown(response_data["analyses"]["position"])
            with col2:
                position_agent.show_position(response_data["title"].split('-')[1].strip())
        else:
            st.warning("No position data available")
    
    # Hull performance
    with st.expander("Hull Performance", expanded=True):
        hull_chart = next((chart for name, chart in response_data["charts"] if name == "hull"), None)
        if hull_chart:
            st.pyplot(hull_chart)
            st.markdown(response_data["analyses"]["hull"])
            
            # Display metrics
            cols = st.columns(2)
            with cols[0]:
                st.metric(
                    "Power Loss",
                    f"{response_data['scores']['hull_power_loss']:.1f}%",
                    delta=None,
                    delta_color="inverse"
                )
            with cols[1]:
                st.metric(
                    "Hull Condition",
                    response_data['scores']['hull_condition'],
                    delta=None
                )
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
                    st.metric(
                        metric.replace('_', ' ').title(),
                        f"{value:.1f}%",
                        delta=None
                    )
        else:
            st.warning("No vessel score data available")
    
    # Crew scores
    with st.expander("Crew Performance Scores", expanded=True):
        if "crew" in response_data["scores"]:
            st.markdown(response_data["analyses"]["crew"])
            cols = st.columns(3)
            for i, (metric, value) in enumerate(response_data["scores"]["crew"].items()):
                with cols[i % 3]:
                    st.metric(
                        metric.replace('_', ' ').title(),
                        f"{value:.1f}%",
                        delta=None
                    )
        else:
            st.warning("No crew score data available")

def handle_user_query(query: str) -> dict:
    """Process user query and return appropriate response with caching."""
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

def main():
    """Main application function with persistent state."""
    # Initialize session state
    SessionState.initialize()
    
    # Application header
    st.title("VesselIQ - Smart Vessel Insights")
    st.markdown(
        "Ask me about vessel performance, speed consumption, crew performance, "
        "vessel position, or request a complete vessel synopsis!"
    )
    
    # Create a container for chat history
    chat_container = st.container()
    
    # Display chat history and stored visualizations
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if "response_data" in message and message.get("type"):
                    display_response({
                        "content": message["content"],
                        "response_data": message["response_data"],
                        "type": message["type"]
                    })
    
    # Handle user input
    if prompt := st.chat_input("What would you like to know about vessel performance?", key="chat_input"):
        try:
            # Add user message
            with st.chat_message("human"):
                st.markdown(prompt)
            st.session_state.messages.append({"role": "human", "content": prompt})
            
            # Process query
            with st.spinner("Processing your request..."):
                response = handle_user_query(prompt)
            
            # Display assistant response
            with st.chat_message("assistant"):
                st.markdown(response["content"])
                if "response_data" in response:
                    display_response(response)
            
            # Store response in chat history
            message_data = {
                "role": "assistant",
                "content": response["content"],
                "type": response.get("type"),
                "response_data": response.get("response_data")
            }
            st.session_state.messages.append(message_data)
            
            # Clean old cache if necessary
            SessionState.clear_old_cache()
            
            # Force rerun to update display
            st.experimental_rerun()
            
        except Exception as e:
            error_message = f"An error occurred while processing your request: {str(e)}"
            st.error(error_message)
            st.session_state.messages.append({
                "role": "assistant",
                "content": error_message,
                "type": "error"
            })

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
