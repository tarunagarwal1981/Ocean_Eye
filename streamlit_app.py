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

Naming pattern rules:
1. Remove any unnecessary prefixes or suffixes
2. If the name is between quotes, extract only the content within quotes
3. If the name follows keywords like "for", "of", "about", extract the subsequent text
4. Convert vessel names to proper case format

Example extractions:
- "Show me hull performance of mv trammo marycam" => "Trammo Marycam"
- "Where is 'Nordic Aurora' now?" => "Nordic Aurora"
- "Performance report for vessel oceanica explorer" => "Oceanica Explorer"

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
        
        # Verify and clean vessel name
        if decision_data.get('vessel_name'):
            decision_data['vessel_name'] = clean_vessel_name(decision_data['vessel_name'])
        else:
            # Advanced pattern matching for vessel name extraction
            patterns = [
                r'(?:of|for|about)\s+["\']?([^"\']+?)["\']?\s*(?:\?|$)',  # Matches after 'of', 'for', 'about'
                r'["\']([^"\']+)["\']',  # Matches quoted names
                r'vessel\s+([^\s?]+(?:\s+[^\s?]+)*)',  # Matches after 'vessel'
                r'mv\s+([^\s?]+(?:\s+[^\s?]+)*)'  # Matches after 'mv'
            ]
            
            for pattern in patterns:
                match = re.search(pattern, query, re.IGNORECASE)
                if match:
                    decision_data['vessel_name'] = clean_vessel_name(match.group(1))
                    break
            
            if not decision_data.get('vessel_name'):
                # Last resort: try to find any capitalized words sequence
                capitalized_words = re.findall(r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', query)
                if capitalized_words:
                    decision_data['vessel_name'] = clean_vessel_name(capitalized_words[0])
        
        return decision_data
        
    except Exception as e:
        st.error(f"Error in LLM decision: {str(e)}")
        return {
            "vessel_name": None,
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
                for i, (metric, value) in enumerate(vessel_scores.items()):
                    with cols[i % 3]:
                        st.metric(metric.replace('_', ' ').title(), f"{value:.1f}%")
            else:
                st.warning("No vessel score data available")
        
        # Display crew score
        with st.expander("Crew Performance Scores", expanded=False):
            if crew_scores:
                st.markdown(crew_analysis)
                cols = st.columns(3)
                for i, (metric, value) in enumerate(crew_scores.items()):
                    with cols[i % 3]:
                        st.metric(metric.replace('_', ' ').title(), f"{value:.1f}%")
            else:
                st.warning("No crew score data available")
                
    except Exception as e:
        st.error(f"Error generating vessel synopsis: {str(e)}")

def handle_user_query(query: str):
    """Process user query and return appropriate response."""
    decision_data = get_llm_decision(query)
    vessel_name = decision_data.get("vessel_name")
    decision_type = decision_data.get("decision", "general_info")
    response_type = decision_data.get("response_type", "concise")
    
    if not vessel_name:
        return "I couldn't identify a vessel name in your query. Could you please specify the vessel name?"
    
    # Store context in session state
    st.session_state.vessel_name = vessel_name
    st.session_state.decision_type = decision_type
    st.session_state.response_type = response_type
    
    try:
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
            return """I understand you're asking about vessel information. Please specify what aspect you'd like to know about:
            - Hull performance
            - Speed consumption
            - Vessel score
            - Crew performance
            - Current position
            - Complete vessel synopsis"""
            
    except Exception as e:
        st.error(f"Error processing query: {str(e)}")
        return "I encountered an error while processing your request. Please try again or rephrase your query."

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
    
    # Add custom CSS for better UI
    st.markdown(
        """
        <style>
            /* Main container styling */
            .block-container {
                max-width: 1200px;
                padding-top: 2rem;
            }
            
            /* Title styling */
            .stTitle {
                font-size: 2rem;
                font-weight: bold;
                margin-bottom: 1rem;
            }
            
            /* Text content styling */
            .stMarkdown {
                font-size: 1.1rem;
            }
            
            /* Chat container styling */
            .stChatFloatingInputContainer {
                max-width: 80% !important;
                margin-left: auto !important;
                margin-right: auto !important;
            }
            
            /* Chat message styling */
            .stChatMessage {
                max-width: 100% !important;
                padding: 1rem !important;
            }
            
            /* Status colors */
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
    
    # Initialize session state for chat history if not exists
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Handle user input
    if prompt := st.chat_input("What would you like to know about vessel performance?"):
        try:
            # Display user message
            st.session_state.messages.append({"role": "human", "content": prompt})
            with st.chat_message("human"):
                st.markdown(prompt)
            
            # Process user input
            if any(word in prompt.lower() for word in ["more", "details", "charts", "yes", "show me"]):
                # Handle follow-up questions
                response = handle_follow_up(prompt)
            else:
                # Process new queries
                response = handle_user_query(prompt)
            
            # Display assistant response
            if response:
                st.session_state.messages.append({"role": "assistant", "content": response})
                with st.chat_message("assistant"):
                    st.markdown(response)
        
        except Exception as e:
            # Handle any errors gracefully
            error_message = (
                "I encountered an error processing your request. "
                "Please try again with a different query."
            )
            
            # Display error in chat
            st.session_state.messages.append({"role": "assistant", "content": error_message})
            with st.chat_message("assistant"):
                st.markdown(error_message)
            
            # Log detailed error
            st.error(f"Error details: {str(e)}")

if __name__ == "__main__":
    main()
