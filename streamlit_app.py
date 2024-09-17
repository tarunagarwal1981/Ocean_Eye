import streamlit as st
import openai
import os
import json
import pandas as pd
from typing import Dict, Any, Tuple
from datetime import datetime, timedelta
from utils.database_utils import get_db_connection
from utils.nlp_utils import extract_vessel_name, clean_vessel_name

# Initialize OpenAI API
def get_api_key():
    if 'openai' in st.secrets:
        return st.secrets['openai']['api_key']
    api_key = os.getenv('OPENAI_API_KEY')
    if api_key is None:
        raise ValueError("API key not found. Set OPENAI_API_KEY as an environment variable.")
    return api_key

openai.api_key = get_api_key()

# Few-Shot Learning Examples
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
2. Consider performing underwater hull inspections to validate the fouling rate and adjust the cleaning date if necessary.
3. Implement operational measures like speed optimization to mitigate the impact of increased power requirements.
4. After the next hull cleaning, ensure proper data collection to maintain accurate performance tracking.

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

Now, please answer the following question in a similar style, using the data I provide:
{user_question}

{data_summary}

Provide a detailed analysis and recommendations based on this data.
"""

# LLM Prompts
DECISION_PROMPT = """
You are an AI assistant specialized in vessel performance analysis. Your task is to determine what type of information is needed to answer the user's query. The options are:

1. Hull performance
2. Speed consumption
3. Combined performance (both hull and speed)
4. General vessel information

Based on the user's query, output your decision as a JSON object with the following structure:
{{
    "decision": "hull_performance" or "speed_consumption" or "combined_performance" or "general_info",
    "explanation": "Brief explanation of why you made this decision"
}}

User Query: {query}

Decision:
"""

def get_llm_decision(query: str) -> Dict[str, str]:
    """Get the LLM's decision on what type of information is needed."""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a vessel performance analysis expert."},
                {"role": "user", "content": DECISION_PROMPT.format(query=query)}
            ],
            temperature=0.3,
        )
        decision_text = response.choices[0].message['content'].strip()
        return json.loads(decision_text)
    except json.JSONDecodeError:
        return {
            "decision": "general_info",
            "explanation": "Failed to parse LLM response, defaulting to general info."
        }
    except Exception as e:
        st.error(f"Error in LLM decision: {str(e)}")
        return {
            "decision": "general_info",
            "explanation": "An error occurred, defaulting to general info."
        }

def get_llm_analysis(query: str, vessel_name: str, data_summary: str) -> str:
    """Get the LLM's analysis based on the query and available data."""
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a vessel performance analysis expert."},
            {"role": "user", "content": FEW_SHOT_EXAMPLES.format(
                user_question=query, data_summary=data_summary)}
        ],
        temperature=0.5,
    )
    return response.choices[0].message['content']

def fetch_six_months_data(vessel_name: str) -> pd.DataFrame:
    """Fetch data from hull_performance_six_months table for a specific vessel."""
    conn = get_db_connection()
    try:
        query = """
        SELECT *
        FROM hull_performance_six_months
        WHERE vessel_name = %s
        """
        data = pd.read_sql(query, conn, params=(vessel_name,))
    finally:
        conn.close()
    return data

def fetch_performance_data(vessel_name: str) -> pd.DataFrame:
    """Fetch performance data for a specific vessel."""
    conn = get_db_connection()
    try:
        query = """
        SELECT report_date, hull_roughness_power_loss
        FROM hull_performance
        WHERE vessel_name = %s
        ORDER BY report_date DESC
        LIMIT 180  -- Assuming we want the last 6 months of data
        """
        data = pd.read_sql(query, conn, params=(vessel_name,))
    finally:
        conn.close()
    return data

def analyze_hull_performance(vessel_name: str) -> Tuple[Any, float, str]:
    """Analyze hull performance for a specific vessel."""
    performance_data = fetch_performance_data(vessel_name)
    
    if performance_data.empty:
        return None, None, "Unknown"
    
    # Generate chart (you'll need to implement this part)
    chart = generate_hull_performance_chart(performance_data)
    
    # Calculate current power loss (using the most recent data point)
    current_power_loss = performance_data['hull_roughness_power_loss'].iloc[0]
    
    # Determine hull condition
    hull_condition = determine_hull_condition(current_power_loss)
    
    return chart, current_power_loss, hull_condition

def determine_hull_condition(power_loss: float) -> str:
    """Determine hull condition based on power loss."""
    if power_loss is None:
        return "Unknown"
    elif power_loss < 5:
        return "Excellent"
    elif power_loss < 10:
        return "Good"
    elif power_loss < 15:
        return "Fair"
    else:
        return "Poor"

def generate_hull_performance_chart(performance_data: pd.DataFrame):
    # Implement chart generation logic here
    # For now, we'll return None as a placeholder
    return None

def fetch_hull_performance_data(vessel_name: str) -> Dict[str, Any]:
    """Fetch hull performance data and generate summary."""
    chart, power_loss, hull_condition = analyze_hull_performance(vessel_name)
    performance_data = fetch_performance_data(vessel_name)
    six_months_data = fetch_six_months_data(vessel_name)
    
    forecasted_cleaning = None
    six_months_data_available = False
    if not six_months_data.empty:
        six_months_data_available = True
        if 'forecasted_hull_cleaning_date' in six_months_data.columns:
            forecasted_cleaning = six_months_data['forecasted_hull_cleaning_date'].iloc[0]
            if pd.isnull(forecasted_cleaning):
                forecasted_cleaning = "Not available (insufficient data)"
    
    return {
        "chart_available": chart is not None,
        "power_loss": power_loss,
        "hull_condition": hull_condition,
        "performance_data_available": not performance_data.empty,
        "six_months_data_available": six_months_data_available,
        "forecasted_cleaning": forecasted_cleaning,
    }

def fetch_speed_consumption_data(vessel_name: str) -> Dict[str, Any]:
    """Fetch speed consumption data and generate summary."""
    # Implement speed consumption data fetching logic here
    # For now, we'll return a placeholder
    return {
        "chart_available": False,
    }

def generate_data_summary(vessel_name: str, decision: str) -> str:
    """Generate a summary of available data based on the LLM's decision."""
    summary = f"Vessel Name: {vessel_name}\n"
    
    if decision in ["hull_performance", "combined_performance"]:
        hull_data = fetch_hull_performance_data(vessel_name)
        summary += f"Hull Performance Data:\n"
        summary += f"- Chart available: {'Yes' if hull_data['chart_available'] else 'No'}\n"
        summary += f"- Current excess power: {hull_data['power_loss']:.2f}% if hull_data['power_loss'] is not None else 'Not available'}\n"
        summary += f"- Hull condition: {hull_data['hull_condition'] if hull_data['hull_condition'] is not None else 'Not available'}\n"
        summary += f"- Historical performance data available: {'Yes' if hull_data['performance_data_available'] else 'No'}\n"
        summary += f"- Six months summary data available: {'Yes' if hull_data['six_months_data_available'] else 'No'}\n"
        summary += f"- Forecasted hull cleaning: {hull_data['forecasted_cleaning'] if hull_data['forecasted_cleaning'] else 'Not available'}\n"
        if hull_data['forecasted_cleaning'] == "Not available (insufficient data)":
            summary += "  Note: Insufficient valid data points after the last event for accurate forecasting.\n"
    
    if decision in ["speed_consumption", "combined_performance"]:
        speed_data = fetch_speed_consumption_data(vessel_name)
        summary += f"Speed Consumption Data:\n"
        summary += f"- Chart available: {'Yes' if speed_data['chart_available'] else 'No'}\n"
    
    return summary

def handle_user_query(query: str) -> Tuple[str, str, str]:
    """Main function to handle user queries."""
    vessel_name = clean_vessel_name(extract_vessel_name(query))
    if not vessel_name:
        return "I couldn't identify a vessel name in your query. Could you please provide a specific vessel name?", "general_info", None

    llm_decision = get_llm_decision(query)
    data_summary = generate_data_summary(vessel_name, llm_decision['decision'])
    analysis = get_llm_analysis(query, vessel_name, data_summary)

    return analysis, llm_decision['decision'], vessel_name

def display_charts(decision: str, vessel_name: str):
    """Display relevant charts based on the LLM's decision."""
    if decision in ["hull_performance", "combined_performance"]:
        chart, _, _ = analyze_hull_performance(vessel_name)
        if chart:
            st.pyplot(chart)
    if decision in ["speed_consumption", "combined_performance"]:
        # Implement speed consumption chart display here
        pass

def main():
    st.title("Advanced Vessel Performance Chatbot")

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What would you like to know about vessel performance?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        analysis, decision, vessel_name = handle_user_query(prompt)

        st.session_state.messages.append({"role": "assistant", "content": analysis})
        with st.chat_message("assistant"):
            st.markdown(analysis)

        display_charts(decision, vessel_name)

if __name__ == "__main__":
    main()
