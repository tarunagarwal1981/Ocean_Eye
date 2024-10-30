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
from agents.vessel_performance_agent import analyze_vessel_score
from agents.crew_performance_agent import analyze_crew_score
from agents.position_tracking_agent import PositionTrackingAgent
from utils.database_utils import fetch_data_from_db
from utils.nlp_utils import clean_vessel_name

# Initialize position tracking agent
position_agent = PositionTrackingAgent()

# LLM Decision Prompt
DECISION_PROMPT = """
You are an AI assistant specialized in vessel performance analysis. The user will ask a query related to vessel performance. Based on the user's query, do two things:
1. Extract only the vessel name from the query. The vessel name may appear after the word 'of' (e.g., 'hull performance of Trammo Marycam' => 'Trammo Marycam').
2. Determine what type of performance information is needed to answer the user's query. The options are:
   - Hull performance
   - Speed consumption
   - Vessel score
   - Crew score
   - Position tracking
   - Combined performance (hull and speed)
   - Vessel synopsis (complete vessel overview)
   - General vessel information

Choose the decision based on these rules:
- If the user asks for "vessel synopsis", "vessel summary", or "vessel overview", return "vessel_synopsis"
- If the user asks for "vessel performance" or a combination of "hull and speed performance," return "combined_performance"
- If the user asks only about "hull performance" or "hull and propeller performance," return "hull_performance"
- If the user asks only about "speed consumption," return "speed_consumption"
- If the user asks about "vessel score", "vessel rating", or "vessel KPIs", return "vessel_score"
- If the user asks about "crew performance", "crew score", or "crew rating", return "crew_score"
- If the user asks about "position", "location", or "where is", return "position_tracking"

Output your response as a JSON object with the following structure:
{
    "vessel_name": "<vessel_name>",
    "decision": "hull_performance" or "speed_consumption" or "combined_performance" or "vessel_synopsis" or "vessel_score" or "crew_score" or "position_tracking" or "general_info",
    "response_type": "concise" or "detailed",
    "explanation": "Brief explanation of why you made this decision"
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

# Initialize OpenAI API
openai.api_key = get_api_key()

def get_llm_decision(query: str) -> Dict[str, str]:
    """Get decision from LLM about query type and vessel name."""
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
        decision_data = json.loads(decision_text)
        
        # Fallback if vessel name extraction fails
        if not decision_data.get('vessel_name'):
            match = re.search(r'of\s+(.+)', query, re.IGNORECASE)
            if match:
                decision_data['vessel_name'] = match.group(1).strip()
            else:
                decision_data['vessel_name'] = query
        
        return decision_data
        
    except Exception as e:
        st.error(f"Error in LLM decision: {str(e)}")
        return {
            "vessel_name": query,
            "decision": "general_info",
            "response_type": "concise",
            "explanation": "Error occurred, defaulting to general info"
        }

def show_vessel_synopsis(vessel_name: str):
    """Display comprehensive vessel synopsis using data from all agents."""
    try:
        # Get hull performance data
        hull_analysis, power_loss, hull_condition, hull_chart = analyze_hull_performance(vessel_name)
        
        # Get speed consumption data
        speed_analysis, speed_charts = analyze_speed_consumption(vessel_name)
        
        # Get vessel score data
        vessel_scores, vessel_analysis = analyze_vessel_score(vessel_name)
        
        # Get crew score data
        crew_scores, crew_analysis = analyze_crew_score(vessel_name)
        
        # Create header with vessel name
        st.header(f"Vessel Synopsis - {vessel_name.upper()}")
        
        # Add CSS for status colors
        st.markdown("""
            <style>
                .status-poor { color: #dc3545; font-weight: 500; }
                .status-average { color: #ffc107; font-weight: 500; }
                .status-good { color: #28a745; font-weight: 500; }
            </style>
        """, unsafe_allow_html=True)
        
        # Display position information
        with st.expander("Last Reported Position", expanded=False):
            position_agent.show_position(vessel_name)
        
        # Display hull performance
        with st.expander("Hull Performance", expanded=False):
            if hull_chart:
                st.pyplot(hull_chart)
                st.markdown(hull_analysis)
            else:
                st.warning("No hull performance data available")
        
        # Display speed consumption
        with st.expander("Speed Consumption Profile", expanded=False):
            if speed_charts:
                st.pyplot(speed_charts)
                st.markdown(speed_analysis)
            else:
                st.warning("No speed consumption data available")
        
        # Display vessel score
        with st.expander("Vessel Performance Scores", expanded=False):
            if vessel_scores:
                st.markdown(vessel_analysis)
                cols = st.columns(3)
                metrics = [
                    ("Vessel Score", vessel_scores.get('vessel_score', 0)),
                    ("Cost Score", vessel_scores.get('cost_score', 0)),
                    ("Environment Score", vessel_scores.get('environment_score', 0)),
                    ("Operation Score", vessel_scores.get('operation_score', 0)),
                    ("Reliability Score", vessel_scores.get('reliability_score', 0)),
                    ("Digitalization Score", vessel_scores.get('digitalization_score', 0))
                ]
                
                for i, (label, value) in enumerate(metrics):
                    with cols[i % 3]:
                        st.metric(label, f"{value:.1f}%")
            else:
                st.warning("No vessel score data available")
        
        # Display crew score
        with st.expander("Crew Performance Scores", expanded=False):
            if crew_scores:
                st.markdown(crew_analysis)
                cols = st.columns(3)
                metrics = [
                    ("Crew Skill Index", crew_scores.get('crew_skill_index', 0)),
                    ("Capability Index", crew_scores.get('capability_index', 0)),
                    ("Competency Index", crew_scores.get('competency_index', 0)),
                    ("Collaboration Index", crew_scores.get('collaboration_index', 0)),
                    ("Character Index", crew_scores.get('character_index', 0))
                ]
                
                for i, (label, value) in enumerate(metrics):
                    with cols[i % 3]:
                        st.metric(label, f"{value:.1f}%")
            else:
                st.warning("No crew score data available")
                
    except Exception as e:
        st.error(f"Error generating vessel synopsis: {str(e)}")

def handle_user_query(query: str):
    """Process user query and return appropriate response."""
    decision_data = get_llm_decision(query)
    vessel_name = decision_data.get("vessel_name", "")
    decision_type = decision_data.get("decision", "general_info")
    response_type = decision_data.get("response_type", "concise")
    
    if not vessel_name:
        return "I couldn't identify a vessel name in your query."
    
    # Store context in session state
    st.session_state.vessel_name = vessel_name
    st.session_state.decision_type = decision_type
    st.session_state.response_type = response_type
    
    # Handle different types of requests
    if decision_type == "vessel_synopsis":
        show_vessel_synopsis(vessel_name)
        return f"Here's the vessel synopsis for {vessel_name}. Let me know if you need any specific information explained."
    
    elif decision_type == "hull_performance":
        hull_analysis, power_loss, hull_condition, hull_chart = analyze_hull_performance(vessel_name)
        if response_type == "concise":
            return f"The hull of {vessel_name} is in {hull_condition} condition with {power_loss:.1f}% power loss. Would you like to see detailed analysis and charts?"
        else:
            st.pyplot(hull_chart)
            return hull_analysis
    
    elif decision_type == "speed_consumption":
        speed_analysis, speed_charts = analyze_speed_consumption(vessel_name)
        if response_type == "concise":
            return f"I've analyzed the speed consumption profile for {vessel_name}. Would you like to see the detailed analysis and charts?"
        else:
            st.pyplot(speed_charts)
            return speed_analysis
    
    elif decision_type == "vessel_score":
        scores, analysis = analyze_vessel_score(vessel_name)
        if scores:
            if response_type == "concise":
                return f"The overall vessel score for {vessel_name} is {scores['vessel_score']:.1f}%. Would you like to see the detailed analysis?"
            else:
                st.markdown(analysis)
                return "I've displayed the detailed vessel score analysis above. Let me know if you need any clarification."
        else:
            return "Unable to retrieve vessel score data."
    
    elif decision_type == "crew_score":
        scores, analysis = analyze_crew_score(vessel_name)
        if scores:
            if response_type == "concise":
                return f"The crew skill index for {vessel_name} is {scores['crew_skill_index']:.1f}%. Would you like to see the detailed analysis?"
            else:
                st.markdown(analysis)
                return "I've displayed the detailed crew score analysis above. Let me know if you need any clarification."
        else:
            return "Unable to retrieve crew score data."
    
    elif decision_type == "position_tracking":
        position_agent.show_position(vessel_name)
        analysis = position_agent.get_position_analysis(vessel_name)
        return analysis
    
    elif decision_type == "combined_performance":
        hull_analysis, _, hull_condition, hull_chart = analyze_hull_performance(vessel_name)
        speed_analysis, speed_charts = analyze_speed_consumption(vessel_name)
        
        if response_type == "concise":
            return f"I have analyzed both hull and speed performance for {vessel_name}. Would you like to see the detailed analysis and charts?"
        else:
            st.pyplot(hull_chart)
            st.pyplot(speed_charts)
            return f"{hull_analysis}\n\n{speed_analysis}"
    
    else:
        return "I understand you're asking about vessel information, but could you please specify what aspect you're interested in? (hull performance, speed consumption, vessel score, crew score, position tracking, or complete vessel synopsis)"

def handle_follow_up(query: str):
    """Handle follow-up requests for more information or charts."""
    if 'vessel_name' not in st.session_state or 'decision_type' not in st.session_state:
        return "Could you please provide your initial question again?"
    
    vessel_name = st.session_state.vessel_name
    decision_type = st.session_state.decision_type
    
    if decision_type == "vessel_synopsis":
        show_vessel_synopsis(vessel_name)
    elif decision_type == "hull_performance":
        _, _, _, hull_chart = analyze_hull_performance(vessel_name)
        st.pyplot(hull_chart)
    elif decision_type == "speed_consumption":
        _, speed_charts = analyze_speed_consumption(vessel_name)
        st.pyplot(speed_charts)
    elif decision_type == "vessel_score":
        scores, analysis = analyze_vessel_score(vessel_name)
        st.markdown(analysis)
    elif decision_type == "crew_score":
        scores, analysis = analyze_crew_score(vessel_name)
        st.markdown(analysis)
    elif decision_type == "position_tracking":
        position_agent.show_position(vessel_name)
    elif decision_type == "combined_performance":
        _, _, _, hull_chart = analyze_hull_performance(vessel_name)
        _, speed_charts = analyze_speed_consumption(vessel_name)
        st.pyplot(hull_chart)
        st.pyplot(speed_charts)

    return "I've updated the information above. Is there anything specific you'd like me to explain?"

def main():
    # Set page config
    st.set_page_config(layout="wide", page_title="VesselIQ")
    
    # Add custom CSS
    st.markdown("""
        <style>
            .block-container { max-width: 1200px; padding-top: 2rem; }
            .stTitle { font-size: 2rem; font-weight: bold; margin-bottom: 1rem; }
            .stMarkdown { font-size: 1.1rem; }
        </style>
    """, unsafe_allow_html=True)
    
    st.title("VesselIQ - Smart Vessel Insights")
    st.markdown("Ask me about vessel performance, speed consumption, crew performance, vessel position, or request a complete vessel synopsis!")
    
    # Initialize session state variables
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Handle user input
    if prompt := st.chat_input("What would you like to know about vessel performance?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "human", "content": prompt})
        with st.chat_message("human"):
            st.markdown(prompt)
        
        # Check if it's a follow-up request
        if any(word in prompt.lower() for word in ["more", "details", "charts", "yes", "show me"]):
            response = handle_follow_up(prompt)
        else:
            response = handle_user_query(prompt)
        
        # Add assistant response to chat history
        if response:
            st.session_state.messages.append({"role": "assistant", "content": response})
            with st.chat_message("assistant"):
                st.markdown(response)

if __name__ == "__main__":
    main()
