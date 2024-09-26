import streamlit as st
import openai
import os
import pandas as pd
import json
import re
from typing import Dict
from agents.hull_performance_agent import analyze_hull_performance
from agents.speed_consumption_agent import analyze_speed_consumption
from utils.nlp_utils import clean_vessel_name

# LLM Prompts
DECISION_PROMPT = """
You are an AI assistant specialized in vessel performance analysis. The user will ask a query related to vessel performance. Based on the user's query, do two things:
1. Extract only the vessel name from the query. The vessel name may appear after the word 'of' (e.g., 'hull performance of Trammo Marycam' => 'Trammo Marycam').
2. Determine what type of performance information is needed to answer the user's query. The options are:
   - Hull performance
   - Speed consumption
   - Combined performance (both hull and speed)
   - General vessel information

Output your response as a JSON object with the following structure:
{
    "vessel_name": "<vessel_name>",
    "decision": "hull_performance" or "speed_consumption" or "combined_performance" or "general_info",
    "explanation": "Brief explanation of why you made this decision"
}
"""

# LLM Prompts and Few-Shot Examples
FEW_SHOT_EXAMPLES = """
Example 1:
Q: What's the hull performance of the vessel Oceanica Explorer?
A: I've analyzed the hull performance data for Oceanica Explorer. Here's what I found:

1. Current Excess Power: 8.7%. This means the vessel currently requires 8.7% more power to maintain its speed compared to a clean hull condition.
2. Fouling Rate: Approximately 0.5% increase in power loss per month over the last 6 months, indicating a moderate level of fouling accumulation.
3. Hull Condition: Based on the current excess power, the hull condition is considered Average.
4. Forecasted Hull Cleaning: Scheduled for 2023-11-15. It's important to plan operations around this date to optimize performance.
5. Performance Impact: The 8.7% excess power requirement translates to approximately 6-7% increased fuel consumption, assuming typical operating conditions.

Recommendations:
1. Monitor the hull condition closely as you approach the forecasted cleaning date.
2. **Since the hull condition is Average**, I recommend performing an underwater hull inspection and propeller polishing to validate the fouling rate and adjust the cleaning date if necessary.
3. Implement operational measures like speed optimization to mitigate the impact of increased power requirements.
4. After the next hull cleaning, ensure proper data collection to maintain accurate performance tracking.

**Additionally, here's the speed consumption chart and hull performance chart for your reference.**

*Please note: This analysis is as good as the data reported by the vessel. We kindly request you to remind the vessel to report data accurately and periodically. Accurate analysis not only helps reduce the vessel's carbon footprint but also realizes fuel cost savings.*

Example 2:
Q: Can you provide the speed consumption profile for the vessel Starlight Voyager?
A: I've analyzed the speed consumption data for Starlight Voyager. Here's what I found:

1. Speed Range: The vessel operates between 10 to 18 knots based on the available data.
2. Consumption Trend: There's a clear non-linear increase in fuel consumption as speed increases.
3. Optimal Speed: The most fuel-efficient speed appears to be around 12-13 knots, where the increase in consumption per knot is lowest.
4. High-Speed Impact: Operating at speeds above 16 knots results in a sharp increase in fuel consumption, potentially over 50% more than at optimal speed.
5. Loading Conditions: The data shows distinct consumption profiles for laden and ballast conditions, with ballast condition generally showing lower consumption at the same speeds.

Recommendations:
1. Prioritize operating at speeds between 12-13 knots when possible to maximize fuel efficiency.
2. For time-sensitive operations, consider the trade-off between increased speed and fuel consumption, especially above 16 knots.
3. Optimize route planning to take advantage of the more efficient ballast condition where applicable.
4. Monitor and record speed and consumption data regularly to identify any deviations from this profile, which could indicate performance issues.
5. Consider conducting a detailed analysis of the economic impact of speed on your specific trade routes to find the optimal balance between speed and efficiency.

**Additionally, here's the speed consumption chart for your reference.**

*Please note: This analysis is as good as the data reported by the vessel. We kindly request you to remind the vessel to report data accurately and periodically. Accurate analysis not only helps reduce the vessel's carbon footprint but also realizes fuel cost savings.*

Guidelines for Analysis:
- If the user asks about vessel performance or hull performance, always include **both the speed consumption chart and hull performance chart** for reference.
- If the hull condition is gauged as **Good**, do not recommend any hull cleaning or underwater inspection.
- If the hull condition is **Average**, recommend performing an underwater hull inspection and propeller polishing.
- If the hull condition is **Poor**, recommend performing hull cleaning and propeller polishing.
- After each response, include a polite reminder: *This analysis is as good as the data reported by the vessel. We kindly request you to remind the vessel to report data accurately and periodically. Accurate analysis not only helps reduce the vessel's carbon footprint but also realizes fuel cost savings.*

Now, please answer the following question in a similar style, using the data I provide:
{user_question}

{data_summary}

Provide a detailed analysis and recommendations based on this data.
"""

# LLM detailed analysis function
def get_llm_analysis(query: str, hull_analysis: str, speed_analysis: str, hull_condition: str) -> str:
    # Prepare the vessel data summary based on hull and speed analysis
    data_summary = f"Hull Analysis: {hull_analysis}\nSpeed Analysis: {speed_analysis}\nHull Condition: {hull_condition}"

    # LLM prompt including few-shot examples
    prompt = f"""
    {FEW_SHOT_EXAMPLES}

    User Question: {query}

    Vessel Data Summary:
    {data_summary}

    Provide a detailed analysis and recommendations based on this data.
    """

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a vessel performance analysis assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
            temperature=0.5
        )

        return response['choices'][0]['message']['content']

    except Exception as e:
        st.error(f"Error in LLM analysis: {str(e)}")
        return "An error occurred during the analysis."




# Function to get the OpenAI API key
def get_api_key():
    if 'openai' in st.secrets:
        return st.secrets['openai']['api_key']
    api_key = os.getenv('OPENAI_API_KEY')
    if api_key is None:
        raise ValueError("API key not found. Set OPENAI_API_KEY as an environment variable.")
    return api_key

# Initialize OpenAI API
openai.api_key = get_api_key()

# Fallback regex to extract vessel name in case LLM fails
def fallback_extract_vessel_name(query: str) -> str:
    match = re.search(r'of\s+(.+)', query, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return query  # If regex fails, return the original query

# Function to call ChatGPT for decision making using ChatCompletion
def get_llm_decision(query: str) -> Dict[str, str]:
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
        st.write(f"LLM Response: {decision_text}")  # Debugging output
        
        decision_data = json.loads(decision_text)
        
        # Fallback if the vessel name is not correctly extracted
        if "vessel_name" not in decision_data or decision_data['vessel_name'] is None or 'hull performance' in decision_data['vessel_name'].lower():
            decision_data['vessel_name'] = fallback_extract_vessel_name(query)
        
        return decision_data
    except openai.error.InvalidRequestError as e:
        st.error(f"InvalidRequestError: {str(e)}")
        return {
            "vessel_name": fallback_extract_vessel_name(query),
            "decision": "general_info",
            "explanation": "Invalid request. Defaulting to general info."
        }
    except Exception as e:
        st.error(f"Error in LLM decision: {str(e)}")
        return {
            "vessel_name": fallback_extract_vessel_name(query),
            "decision": "general_info",
            "explanation": "An error occurred. Defaulting to general info."
        }

# Function to handle user query and return analysis
def handle_user_query(query: str):
    # Initialize analysis with a default message
    analysis = "No analysis available. Please check your query."

    # Get the decision and vessel name from the LLM (ChatGPT)
    llm_decision = get_llm_decision(query)
    
    vessel_name = llm_decision.get("vessel_name")
    if not vessel_name:
        return "I couldn't identify a vessel name in your query."

    st.write(f"Extracted Vessel Name: {vessel_name}")
    
    # Based on the decision, call the appropriate agent
    if llm_decision['decision'] == 'hull_performance':
        # Unpack the values from hull performance analysis
        analysis, power_loss_pct, hull_condition, hull_chart = analyze_hull_performance(vessel_name)
        st.write(f"Hull performance analysis executed for {vessel_name}.")
        st.write(f"Analysis: {analysis}")
        
        if power_loss_pct is None or hull_condition is None:
            st.warning(f"Hull performance chart is not available for this vessel.")
        else:
            st.success(f"Average Power Loss: {power_loss_pct:.2f}%, Hull Condition: {hull_condition}")
    
    elif llm_decision['decision'] == 'speed_consumption':
        analysis = analyze_speed_consumption(vessel_name)
        st.write("Speed consumption analysis executed.")
    
    elif llm_decision['decision'] == 'combined_performance':
        # Call both hull and speed consumption agents and combine the analysis
        hull_analysis, _, hull_condition, hull_chart = analyze_hull_performance(vessel_name)
        speed_analysis = analyze_speed_consumption(vessel_name)
        
        # Use the LLM to provide a detailed analysis
        analysis = get_llm_analysis(query, hull_analysis, speed_analysis, hull_condition)
        st.write(analysis)
    
    else:
        analysis = "The query seems to require general vessel information or is unclear. Please refine the query."

    return analysis



# Function to display the charts based on the LLM's decision
def display_charts(decision: str, vessel_name: str):
    if decision in ["speed_consumption", "combined_performance"]:
        try:
            speed_analysis, speed_chart = analyze_speed_consumption(vessel_name)
            if speed_chart is not None and hasattr(speed_chart, 'savefig'):
                st.pyplot(speed_chart)
            else:
                st.warning("Speed consumption chart is not available for this vessel.")
        except Exception as e:
            st.error(f"An error occurred while generating the speed consumption chart: {str(e)}")
    
    if decision in ["hull_performance", "combined_performance"]:
        try:
            # Unpack the returned values from the hull performance agent
            hull_analysis, _, _, hull_chart = analyze_hull_performance(vessel_name)
            if hull_chart is not None and hasattr(hull_chart, 'savefig'):
                st.pyplot(hull_chart)
            else:
                st.warning("Hull performance chart is not available for this vessel.")
        except Exception as e:
            st.error(f"An error occurred while generating the hull performance chart: {str(e)}")


# Main function for the Streamlit app
def main():
    st.title("Advanced Vessel Performance Chatbot (Powered by ChatGPT)")

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What would you like to know about vessel performance?"):
        st.session_state.messages.append({"role": "human", "content": prompt})
        with st.chat_message("human"):
            st.markdown(prompt)

        # Get the response from the chatbot
        analysis = handle_user_query(prompt)

        st.session_state.messages.append({"role": "assistant", "content": analysis})
        with st.chat_message("assistant"):
            st.markdown(analysis)

# Run the app
if __name__ == "__main__":
    main()
