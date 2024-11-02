import streamlit as st
import openai
import os
import pandas as pd
import json
import re
from typing import Dict, Tuple, Optional
import folium
from streamlit_folium import st_folium

from agents.vessel_score_agent import analyze_vessel_score
from agents.crew_score_agent import analyze_crew_score
from agents.hull_performance_agent import analyze_hull_performance
from agents.speed_consumption_agent import analyze_speed_consumption
from utils.database_utils import fetch_data_from_db
from agents.engine_troubleshooting_agent import analyze_engine_troubleshooting

# LLM Prompts
DECISION_PROMPT = """
You are an AI assistant specialized in vessel performance analysis. The user will ask a query related to vessel performance. Based on the user's query, do two things:
1. For vessel performance queries:
   Extract the vessel name from the query. The vessel name may appear after the word 'of' (e.g., 'hull performance of Trammo Marycam' => 'Trammo Marycam').

2. For engine-related queries:
   - Vessel name is optional
   - If query is about engine components, maintenance, overhaul, assembling, dismantling, or troubleshooting, classify as "engine_troubleshooting" even without vessel name

3. Determine what type of information is needed to answer the user's query. The options are:
   - Hull performance
   - Speed consumption
   - Combined performance (both hull and speed)
   - Vessel synopsis (complete vessel overview)
   - General vessel information
   - Vessel score
   - Crew performance
   - Engine troubleshooting


Choose the decision based on these rules:
- If the user asks for "vessel synopsis", "vessel summary", or "vessel overview", return "vessel_synopsis"
- If the user asks for "vessel performance" or a combination of "hull and speed performance," return "combined_performance"
- If the user asks only about "hull performance" or "hull and propeller performance," return "hull_performance"
- If the user asks only about "speed consumption," return "speed_consumption"
- If the user asks about "vessel score" or "performance score," return "vessel_score"
- If the user asks about "crew performance" or "crew score," return "crew_score"
- If the query contains words like "engine", "overhaul", "maintenance", "repair", "troubleshoot", return "engine_troubleshooting"
- For engine queries, vessel name is optional and can be null

Output your response as a JSON object with the following structure:
{
    "vessel_name": "<vessel_name or null for engine queries>",
    "decision": "hull_performance" or "speed_consumption" or "combined_performance" or "vessel_synopsis" or "general_info" or "vessel_score" or "crew_score" or "engine_troubleshooting",
    "response_type": "concise" or "detailed",
    "explanation": "Brief explanation of why you made this decision"
}
"""

def get_api_key():
    """Get OpenAI API key from secrets or environment."""
    if 'openai' in st.secrets:
        return st.secrets['openai']['api_key']
    api_key = os.getenv('OPENAI_API_KEY')
    if api_key is None:
        raise ValueError("API key not found. Set OPENAI_API_KEY as an environment variable.")
    return api_key

def get_kpi_summary(vessel_name: str, hull_condition: str, cii_rating: str, 
                    vessel_score: float, cost_score: float, digitalization_score: float,
                    environment_score: float, operation_score: float, reliability_score: float,
                    crew_skill_index: float, capability_index: float, competency_index: float,
                    collaboration_index: float, character_index: float) -> str:
    """
    Get comprehensive KPI analysis with proper vessel name inclusion.
    """
    SUMMARY_PROMPT = f"""
    You are a vessel performance analyst providing insights about vessel metrics. Create a comprehensive but concise summary
    addressing all major performance areas: hull condition, vessel performance scores, and crew performance.

    Important: The vessel name is "{vessel_name}" - use this exact name in your summary.

    Rules for summary:
    1. ALWAYS start with exactly: "Based on the data of {vessel_name.upper()},"
    2. Then discuss hull condition and its implications
    3. Then discuss the vessel score and its components
    4. Finally address crew performance
    5. Keep total length to 4-5 sentences maximum
    6. Prioritize the most critical issues needing immediate attention
    7. For each major issue, provide specific, time-bound recommendation

    Formatting rules:
    1. Use the exact vessel name provided above - do not use placeholders
    2. Format hull condition as:
       - <span class="status-poor">poor</span> for poor condition
       - <span class="status-average">average</span> for average condition
       - <span class="status-good">good</span> for good condition
    3. Format numeric values as:
       - <span class="status-poor">[value]</span>% for values below 60
       - <span class="status-average">[value]</span>% for values 60-75
       - <span class="status-good">[value]</span>% for values above 75
    4. Always include % symbol after the closing span tag for metrics

    Current Data for {vessel_name}:
    Hull Performance:
    - Hull Condition: {hull_condition}
    - CII Rating: {cii_rating}

    Vessel Scores (Target >75%):
    - Overall Score: {vessel_score:.1f}%
    - Cost: {cost_score:.1f}%
    - Operation: {operation_score:.1f}%
    - Environment: {environment_score:.1f}%
    - Reliability: {reliability_score:.1f}%
    - Digitalization: {digitalization_score:.1f}%

    Crew Performance (Target >80%):
    - Overall Crew Skill: {crew_skill_index:.1f}%
    - Capability: {capability_index:.1f}%
    - Competency: {competency_index:.1f}%
    - Collaboration: {collaboration_index:.1f}%
    - Character: {character_index:.1f}%
    """
    
    try:
        messages = [
            {"role": "system", "content": SUMMARY_PROMPT},
            {"role": "user", "content": f"Generate a comprehensive performance summary for vessel {vessel_name} prioritizing hull condition and highlighting critical areas across all KPIs."}
        ]
        
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=400,
            temperature=0.7
        )
        
        summary = response.choices[0].message['content'].strip()
        if not summary.startswith(f"Based on the data of {vessel_name}"):
            summary = f"Based on the data of {vessel_name}, " + summary
            
        return summary
        
    except Exception as e:
        return f"Error generating performance summary: {str(e)}"

def get_last_position(vessel_name: str) -> Tuple[Optional[float], Optional[float]]:
    """Fetch the last reported position for a vessel."""
    query = f"""
    select
      "LATITUDE",
      "LONGITUDE"
    from
      sf_consumption_logs
    where
      UPPER("VESSEL_NAME") = '{vessel_name.upper()}'
      and "LATITUDE" is not null
      and "LONGITUDE" is not null
    order by
      "REPORT_DATE" desc
    limit
      1;
    """
    
    try:
        position_data = fetch_data_from_db(query)
        if not position_data.empty:
            return (
                float(position_data.iloc[0]['LATITUDE']),
                float(position_data.iloc[0]['LONGITUDE'])
            )
        return None, None
    except Exception as e:
        st.error(f"Error fetching position data: {str(e)}")
        return None, None

def show_vessel_position(vessel_name: str):
    """Display the vessel's last reported position with map and coordinates."""
    latitude, longitude = get_last_position(vessel_name)
    
    if latitude is not None and longitude is not None:
        # Add CSS for map visibility
        st.markdown("""
            <style>
                .folium-map {
                    width: 100% !important;
                    min-height: 300px !important;
                    z-index: 1 !important;
                }
            </style>
        """, unsafe_allow_html=True)
        
        # Show coordinates
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Latitude", f"{latitude:.4f}°")
        with col2:
            st.metric("Longitude", f"{longitude:.4f}°")
        
        # Create and display map
        vessel_map = create_vessel_map(latitude, longitude)
        st.markdown('<div style="min-height:300px;">', unsafe_allow_html=True)
        st_folium(
            vessel_map,
            height=300,
            width="100%",
            returned_objects=[],
            key=f"vessel_map_{vessel_name}_{latitude}_{longitude}"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.warning("No position data available for this vessel")

def create_vessel_map(latitude: float, longitude: float) -> folium.Map:
    """Create a Folium map centered on the vessel's position."""
    m = folium.Map(
        location=[latitude, longitude],
        zoom_start=4,
        tiles='cartodb positron',
        scrollWheelZoom=True,
        dragging=True
    )
    
    folium.Marker(
        [latitude, longitude],
        popup='Vessel Position',
        icon=folium.Icon(color='red', icon='info-sign')
    ).add_to(m)
    
    return m

def display_vessel_score(vessel_name: str):
    """
    Display vessel score and its components using Streamlit components.
    Args:
        vessel_name (str): Name of the vessel
    """
    try:
        # Get scores and analysis from the agent
        scores, analysis = analyze_vessel_score(vessel_name)
        
        if not scores:
            st.warning("No vessel score data available")
            return
            
        # Create score display section
        with st.expander("Vessel Performance Score", expanded=False):
            # Display overall score prominently
            st.metric("Overall Vessel Score", f"{scores['vessel_score']:.1f}%")
            
            # Display component scores in columns
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Cost Score", f"{scores['cost_score']:.1f}%")
                st.metric("Environment Score", f"{scores['environment_score']:.1f}%")
            
            with col2:
                st.metric("Operation Score", f"{scores['operation_score']:.1f}%")
                st.metric("Reliability Score", f"{scores['reliability_score']:.1f}%")
            
            with col3:
                st.metric("Digitalization Score", f"{scores['digitalization_score']:.1f}%")
            
            # Display analysis
            st.markdown("### Analysis")
            st.write(analysis)
            
    except Exception as e:
        st.error(f"Error displaying vessel score: {str(e)}")

def display_crew_score(vessel_name: str):
    """
    Display crew performance score and analysis using Streamlit components.
    Args:
        vessel_name (str): Name of the vessel
    """
    try:
        # Get scores and analysis from the agent
        scores, analysis = analyze_crew_score(vessel_name)
        
        if not scores:
            st.warning("No crew performance data available")
            return
            
        # Create crew score display section
        with st.expander("Crew Performance Score", expanded=False):
            # Display overall crew score
            if 'overall_score' in scores:
                st.metric("Overall Crew Score", f"{scores['overall_score']:.1f}%")
            
            # Display component scores if available
            if len(scores) > 1:  # If we have component scores
                cols = st.columns(3)
                
                # Distribute scores across columns
                score_items = list(scores.items())
                for i, col in enumerate(cols):
                    with col:
                        for key, value in score_items[i::3]:  # Display every third item
                            if key != 'overall_score':  # Skip overall score as it's already displayed
                                st.metric(
                                    key.replace('_', ' ').title(),
                                    f"{value:.1f}%"
                                )
            
            # Display analysis
            st.markdown("### Analysis")
            st.write(analysis)
            
    except Exception as e:
        st.error(f"Error displaying crew score: {str(e)}")

def show_vessel_synopsis(vessel_name: str):
   """Display comprehensive vessel synopsis."""
   try:
       # Get all performance data upfront
       hull_analysis, power_loss, hull_condition, hull_chart = analyze_hull_performance(vessel_name)
       speed_analysis, speed_charts = analyze_speed_consumption(vessel_name)
       
       # Get vessel scores
       try:
           vessel_scores, vessel_analysis = analyze_vessel_score(vessel_name)
           vessel_score = vessel_scores.get('vessel_score', 0.0)
           cost_score = vessel_scores.get('cost_score', 0.0)
           digitalization_score = vessel_scores.get('digitalization_score', 0.0)
           environment_score = vessel_scores.get('environment_score', 0.0)
           operation_score = vessel_scores.get('operation_score', 0.0)
           reliability_score = vessel_scores.get('reliability_score', 0.0)
       except Exception as e:
           st.error(f"Error fetching vessel scores: {str(e)}")
           vessel_score = cost_score = digitalization_score = environment_score = operation_score = reliability_score = 0.0
           vessel_analysis = None
       
       # Get crew scores
       try:
           crew_scores, crew_analysis = analyze_crew_score(vessel_name)
           crew_skill_index = crew_scores.get('overall_score', 0.0)
           capability_index = crew_scores.get('capability_index', 0.0)
           competency_index = crew_scores.get('competency_index', 0.0)
           collaboration_index = crew_scores.get('collaboration_index', 0.0)
           character_index = crew_scores.get('character_index', 0.0)
       except Exception as e:
           st.error(f"Error fetching crew scores: {str(e)}")
           crew_skill_index = capability_index = competency_index = collaboration_index = character_index = 0.0
           crew_analysis = None
       
       # Fetch CII Rating
       cii_query = f"""
       select cr."cii_rating"
       from "CII ratings" cr
       join "vessel_particulars" vp on cr."vessel_imo" = vp."vessel_imo"::bigint
       where vp."vessel_name" = '{vessel_name.upper()}';
       """
       cii_data = fetch_data_from_db(cii_query)
       cii_rating = cii_data.iloc[0]['cii_rating'] if not cii_data.empty else "N/A"
       
       # Create header
       st.header(f"Vessel Synopsis - {vessel_name.upper()}")
       
       # Add styling
       st.markdown("""
           <style>
               .status-poor { color: #dc3545; font-weight: 500; }
               .status-average { color: #ffc107; font-weight: 500; }
               .status-good { color: #28a745; font-weight: 500; }
               .kpi-summary {
                   background-color: #f8f9fa;
                   border-radius: 8px;
                   padding: 20px;
                   margin-bottom: 20px;
                   border: 1px solid #e9ecef;
                   line-height: 1.6;
               }
               table {
                   width: 100%;
                   border-collapse: collapse;
               }
               th, td {
                   padding: 8px;
                   border: 1px solid #ddd;
                   text-align: left;
               }
               th {
                   background-color: #f8f9fa;
               }
           </style>
       """, unsafe_allow_html=True)
       
       # Get and display KPI summary
       kpi_summary = get_kpi_summary(
           vessel_name,
           hull_condition,
           cii_rating,
           vessel_score,
           cost_score,
           digitalization_score,
           environment_score,
           operation_score,
           reliability_score,
           crew_skill_index,
           capability_index,
           competency_index,
           collaboration_index,
           character_index
       )
       
       # Display the summary at the top
       st.markdown(f'<div class="kpi-summary">{kpi_summary}</div>', unsafe_allow_html=True)
       
       # Display vessel information
       with st.expander("Vessel Information", expanded=False):
           st.markdown(
               f"""
               | Parameter | Value |
               |-----------|--------|
               | Vessel Name | {vessel_name} |
               | Hull Condition | {hull_condition if hull_condition else "N/A"} |
               | CII Rating | {cii_rating} |
               """
           )
       
       # Display position
       with st.expander("Last Reported Position", expanded=False):
           show_vessel_position(vessel_name)
       
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
       
       # Display vessel score with colored metrics
       with st.expander("Vessel Score Details", expanded=False):
           if vessel_score > 0:
               col1, col2 = st.columns([1, 2])
               with col1:
                   st.metric("Overall Vessel Score", f"{vessel_score:.1f}%")
               
               with col2:
                   st.markdown(
                       f"""
                       <table>
                           <tr>
                               <th>Component</th>
                               <th>Score</th>
                           </tr>
                           <tr>
                               <td>Cost</td>
                               <td><span class='status-{"good" if cost_score >= 75 else "average" if cost_score >= 60 else "poor"}'>{cost_score:.1f}%</span></td>
                           </tr>
                           <tr>
                               <td>Digitalization</td>
                               <td><span class='status-{"good" if digitalization_score >= 75 else "average" if digitalization_score >= 60 else "poor"}'>{digitalization_score:.1f}%</span></td>
                           </tr>
                           <tr>
                               <td>Environment</td>
                               <td><span class='status-{"good" if environment_score >= 75 else "average" if environment_score >= 60 else "poor"}'>{environment_score:.1f}%</span></td>
                           </tr>
                           <tr>
                               <td>Operation</td>
                               <td><span class='status-{"good" if operation_score >= 75 else "average" if operation_score >= 60 else "poor"}'>{operation_score:.1f}%</span></td>
                           </tr>
                           <tr>
                               <td>Reliability</td>
                               <td><span class='status-{"good" if reliability_score >= 75 else "average" if reliability_score >= 60 else "poor"}'>{reliability_score:.1f}%</span></td>
                           </tr>
                       </table>
                       """,
                       unsafe_allow_html=True
                   )
               
               if vessel_analysis:
                   st.markdown("### Analysis")
                   st.write(vessel_analysis)
           else:
               st.warning("No vessel score data available")
       
       # Display crew score with colored metrics
       with st.expander("Crew Score Details", expanded=False):
           if (capability_index > 0 or competency_index > 0 or 
               collaboration_index > 0 or character_index > 0):
               col1, col2 = st.columns([1, 2])
               with col1:
                   st.metric("Crew Skill Index", 
                            f"{crew_skill_index:.1f}%" if crew_skill_index > 0 else "N/A")
               
               with col2:
                   st.markdown(
                       f"""
                       <table>
                           <tr>
                               <th>Component</th>
                               <th>Score</th>
                           </tr>
                           <tr>
                               <td>Capability</td>
                               <td><span class='status-{"good" if capability_index >= 75 else "average" if capability_index >= 60 else "poor"}'>{capability_index:.1f}%</span></td>
                           </tr>
                           <tr>
                               <td>Competency</td>
                               <td><span class='status-{"good" if competency_index >= 75 else "average" if competency_index >= 60 else "poor"}'>{competency_index:.1f}%</span></td>
                           </tr>
                           <tr>
                               <td>Collaboration</td>
                               <td><span class='status-{"good" if collaboration_index >= 75 else "average" if collaboration_index >= 60 else "poor"}'>{collaboration_index:.1f}%</span></td>
                           </tr>
                           <tr>
                               <td>Character</td>
                               <td><span class='status-{"good" if character_index >= 75 else "average" if character_index >= 60 else "poor"}'>{character_index:.1f}%</span></td>
                           </tr>
                       </table>
                       """,
                       unsafe_allow_html=True
                   )
               
               if crew_analysis:
                   st.markdown("### Analysis")
                   st.write(crew_analysis)
           else:
               st.warning("No crew score data available")
               
   except Exception as e:
       st.error(f"Error generating vessel synopsis: {str(e)}")
       st.error("Please check the vessel name and try again.")
        
def get_llm_decision(query: str) -> Dict[str, str]:
    """
    Use OpenAI API to analyze the query and determine the type of information needed.
    
    Args:
        query (str): User's query text
        
    Returns:
        Dict[str, str]: Decision data including vessel name and analysis type
    """
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
        
        try:
            decision_data = json.loads(decision_text)
        except json.JSONDecodeError:
            st.error("Error parsing LLM response. Using fallback logic.")
            decision_data = {}
        
        # Fallback if vessel name extraction fails
        if not decision_data.get('vessel_name'):
            # Try to extract vessel name after 'of'
            match = re.search(r'of\s+([^,.]+)', query, re.IGNORECASE)
            if match:
                decision_data['vessel_name'] = match.group(1).strip()
            else:
                # Try to extract any capitalized words as vessel name
                capitalized_words = re.findall(r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*', query)
                if capitalized_words:
                    decision_data['vessel_name'] = capitalized_words[0]
                else:
                    # Last resort: use the whole query
                    decision_data['vessel_name'] = query
        
        # Ensure all required fields are present
        decision_data.setdefault('decision', 'general_info')
        decision_data.setdefault('response_type', 'concise')
        decision_data.setdefault('explanation', 'Fallback decision')
        
        return decision_data
        
    except Exception as e:
        st.error(f"Error in LLM decision: {str(e)}")
        # Fallback to basic extraction
        vessel_name = re.search(r'of\s+([^,.]+)', query, re.IGNORECASE)
        return {
            "vessel_name": vessel_name.group(1) if vessel_name else query,
            "decision": "general_info",
            "response_type": "concise",
            "explanation": "Error occurred, defaulting to general info"
        }

import streamlit as st
import logging
from typing import Dict, List
import base64
from PIL import Image
import io
import matplotlib.pyplot as plt

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def handle_user_query(query: str) -> str:
    """Process user query and return appropriate response."""
    try:
        st.session_state.setdefault('messages', [])
        decision_data = get_llm_decision(query)
        vessel_name = decision_data.get("vessel_name", "")
        decision_type = decision_data.get("decision", "general_info")
        response_type = decision_data.get("response_type", "concise")
        
        if not vessel_name and decision_type != "engine_troubleshooting":
            st.warning("I couldn't identify a vessel name in your query. Please specify the vessel name.")
            return "Please specify a vessel name."
        
        # Store context in session state
        st.session_state.vessel_name = vessel_name
        st.session_state.decision_type = decision_type
        st.session_state.response_type = response_type
        
        try:
            # Handle different types of requests
            if decision_type == "engine_troubleshooting":
                answer, images = analyze_engine_troubleshooting(vessel_name, query)
                
                # Display answer first
                st.markdown("**Answer:** " + answer)
                
                # Display images if available
                if images:
                    st.markdown("**Relevant Technical Diagrams:**")
                    cols = st.columns(2)
                    
                    for idx, img_data in enumerate(images, 1):
                        try:
                            with cols[idx % 2]:
                                st.markdown(f"#### Image {idx}")
                                # Display source info
                                st.markdown(f"*Source: {img_data.get('file_name', 'Unknown')}, "
                                          f"Page: {img_data.get('page', 'Unknown')}*")
                                
                                # Display image
                                if 'image_data' in img_data:
                                    st.image(
                                        img_data['image_data'],
                                        caption=img_data.get('image_name', f'Image {idx}'),
                                        use_column_width=True
                                    )
                                    
                                    # Create unique key for zoom button
                                    zoom_key = f"zoom_button_{vessel_name}_{idx}"
                                    if st.button("Zoom", key=zoom_key):
                                        st.session_state[zoom_key] = not st.session_state.get(zoom_key, False)
                                    
                                    if st.session_state.get(zoom_key, False):
                                        st.image(
                                            img_data['image_data'],
                                            caption="Zoomed View",
                                            use_column_width=True
                                        )
                                
                                # Show context in expander
                                if img_data.get('surrounding_text'):
                                    with st.expander("Show Context", expanded=False):
                                        st.markdown(img_data['surrounding_text'])
                                        
                        except Exception as e:
                            st.error(f"Could not display image {idx}")
                            logger.error(f"Image display error: {str(e)}")
                            continue
                
                return answer
                
            elif decision_type == "vessel_synopsis":
                show_vessel_synopsis(vessel_name)
                return f"Here's the complete synopsis for {vessel_name}. Let me know if you need any specific information explained."
            
            elif decision_type == "hull_performance":
                hull_analysis, power_loss, hull_condition, hull_chart = analyze_hull_performance(vessel_name)
                if response_type == "concise":
                    return f"The hull of {vessel_name} is in {hull_condition} condition with {power_loss:.1f}% power loss. Would you like to see detailed analysis and charts?"
                else:
                    if hull_chart:
                        st.pyplot(hull_chart)
                    return hull_analysis
            
            elif decision_type == "speed_consumption":
                speed_analysis, speed_charts = analyze_speed_consumption(vessel_name)
                if response_type == "concise":
                    return f"I've analyzed the speed consumption profile for {vessel_name}. Would you like to see the detailed analysis and charts?"
                else:
                    if speed_charts:
                        st.pyplot(speed_charts)
                    return speed_analysis
            
            elif decision_type == "combined_performance":
                hull_analysis, _, hull_condition, hull_chart = analyze_hull_performance(vessel_name)
                speed_analysis, speed_charts = analyze_speed_consumption(vessel_name)
                
                if response_type == "concise":
                    return f"I have analyzed both hull and speed performance for {vessel_name}. Would you like to see the detailed analysis and charts?"
                else:
                    if hull_chart:
                        st.pyplot(hull_chart)
                    if speed_charts:
                        st.pyplot(speed_charts)
                    return f"{hull_analysis}\n\n{speed_analysis}"
            
            elif decision_type == "vessel_score":
                scores, analysis = analyze_vessel_score(vessel_name)
                display_vessel_score(vessel_name)
                return analysis
            
            elif decision_type == "crew_score":
                scores, analysis = analyze_crew_score(vessel_name)
                display_crew_score(vessel_name)
                return analysis
            
            else:
                options = [
                    "- Hull performance",
                    "- Speed consumption",
                    "- Vessel score",
                    "- Crew performance",
                    "- Engine troubleshooting",
                    "- Complete vessel synopsis"
                ]
                return (f"I understand you're asking about {vessel_name}. "
                       f"What specific aspect would you like to know about?\n"
                       + "\n".join(options))
                
        except Exception as e:
            st.error("Error processing request")
            logger.error(f"Error processing request type {decision_type}: {str(e)}")
            return "Sorry, there was an error processing your request. Please try again."
            
    except Exception as e:
        st.error("Error processing query")
        logger.error(f"Error in handle_user_query: {str(e)}")
        return "Sorry, I encountered an error processing your query. Please try again."
        

def handle_follow_up(query: str):
    """Handle follow-up requests for more information or charts."""
    if 'vessel_name' not in st.session_state or 'decision_type' not in st.session_state:
        return "Could you please provide your initial question again?"
    
    vessel_name = st.session_state.vessel_name
    decision_type = st.session_state.decision_type
    
    if decision_type == "engine_troubleshooting":
        answer, diagrams = analyze_engine_troubleshooting(vessel_name, query)
        if diagrams:
            st.write("**Additional Technical Diagrams:**")
            for idx, diagram in enumerate(diagrams):
                st.write(f"\n### Diagram {idx + 1}: {diagram['image_name']}")
                display_image_in_streamlit(
                    diagram['image_data'],
                    f"Technical Diagram {idx + 1}"
                )
        return answer
        
    elif decision_type == "hull_performance":
        hull_analysis, _, _, hull_chart = analyze_hull_performance(vessel_name)
        st.pyplot(hull_chart)
        return hull_analysis
    elif decision_type == "speed_consumption":
        speed_analysis, speed_charts = analyze_speed_consumption(vessel_name)
        st.pyplot(speed_charts)
        return speed_analysis
    elif decision_type == "combined_performance":
        hull_analysis, _, _, hull_chart = analyze_hull_performance(vessel_name)
        speed_analysis, speed_charts = analyze_speed_consumption(vessel_name)
        st.pyplot(hull_chart)
        st.pyplot(speed_charts)
        return f"{hull_analysis}\n\n{speed_analysis}"
    elif decision_type == "vessel_synopsis":
        show_vessel_synopsis(vessel_name)
        return "I've updated the synopsis above. Let me know if you need any clarification."

def main():
    # Page configuration
    st.set_page_config(layout="wide", page_title="VesselIQ")
    
    # Initialize OpenAI API
    openai.api_key = get_api_key()
    
    # Add CSS styles
    st.markdown("""
        <style>
            /* Your existing CSS styles here */
        </style>
    """, unsafe_allow_html=True)
    
    st.title("VesselIQ - Smart Vessel Insights")
    st.markdown("Ask me about vessel performance, engine troubleshooting, speed consumption, or request a complete vessel synopsis!")
    
    # Initialize session state
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Handle user input
    if prompt := st.chat_input("What would you like to know about vessel performance?"):
        st.session_state.messages.append({"role": "human", "content": prompt})
        with st.chat_message("human"):
            st.markdown(prompt)
        
        # Check if it's a follow-up request
        if re.search(r"(more|details|charts|yes)", prompt.lower()):
            response = handle_follow_up(prompt)
        else:
            response = handle_user_query(prompt)
        
        if response:
            st.session_state.messages.append({"role": "assistant", "content": response})
            with st.chat_message("assistant"):
                st.markdown(response)

if __name__ == "__main__":
    main()
