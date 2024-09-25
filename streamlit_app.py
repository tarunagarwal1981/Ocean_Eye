import streamlit as st
import openai
import os
import pandas as pd
import json
from agents.hull_performance_agent import analyze_hull_performance
from agents.speed_consumption_agent import analyze_speed_consumption
from utils.nlp_utils import extract_vessel_name, clean_vessel_name

# LLM Prompts
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
"""

DECISION_PROMPT = """
You are an AI assistant specialized in vessel performance analysis. Your task is to determine what type of information is needed to answer the user's query. The options are:

1. Hull performance
2. Speed consumption
3. Combined performance (both hull and speed)
4. General vessel information

Based on the user's query, output your decision as a JSON object with the following structure:
{
    "decision": "hull_performance" or "speed_consumption" or "combined_performance" or "general_info",
    "explanation": "Brief explanation of why you made this decision"
}

Respond only with the JSON object, no other text.
"""

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

# Function to call ChatGPT for decision making
def get_llm_decision(query: str) -> Dict[str, str]:
    prompt = f"{DECISION_PROMPT}\n\nUser Query: {query}\n\nDecision:"
    response = openai.Completion.create(
        model="gpt-3.5-turbo",
        prompt=prompt,
        max_tokens=100,
        temperature=0.3
    )
    decision_text = response.choices[0].text.strip()
    try:
        return json.loads(decision_text)
    except json.JSONDecodeError:
        return {
            "decision": "general_info",
            "explanation": "Failed to parse GPT response, defaulting to general info."
        }

# Function to get the analysis from ChatGPT
def get_llm_analysis(query: str, vessel_name: str, data_summary: str) -> str:
    prompt = f"{FEW_SHOT_EXAMPLES}\n\nUser Question: {query}\n\nVessel Data:\n{data_summary}\n\nAnalysis:"
    response = openai.Completion.create(
        model="gpt-3.5-turbo",
        prompt=prompt,
        max_tokens=1000,
        temperature=0.5
    )
    return response.choices[0].text.strip()

# Function to generate the vessel data summary
def generate_data_summary(vessel_name: str, decision: str) -> str:
    summary = f"Vessel Name: {vessel_name}\n"
    
    if decision in ["hull_performance", "combined_performance"]:
        hull_chart, power_loss_pct_ed, hull_condition = analyze_hull_performance(vessel_name)
        summary += "Hull Performance Data:\n"
        summary += f"- Chart available: {'Yes' if hull_chart is not None else 'No'}\n"
        if power_loss_pct_ed is not None:
            summary += f"- Current excess power: {power_loss_pct_ed:.2f}%\n"
        else:
            summary += "- Current excess power: Not available\n"
        summary += f"- Hull condition: {hull_condition if hull_condition is not None else 'Not available'}\n"
    
    if decision in ["speed_consumption", "combined_performance"]:
        speed_chart, speed_stats = analyze_speed_consumption(vessel_name)
        summary += "Speed Consumption Data:\n"
        summary += f"- Chart available: {'Yes' if speed_chart is not None else 'No'}\n"
        if speed_stats:
            summary += f"- Overall speed range: {speed_stats['overall']['speed_range'][0]:.2f} to {speed_stats['overall']['speed_range'][1]:.2f} knots\n"
            summary += f"- Overall consumption range: {speed_stats['overall']['consumption_range'][0]:.2f} to {speed_stats['overall']['consumption_range'][1]:.2f} mT/d\n"
            for condition in ['laden', 'ballast']:
                if condition in speed_stats:
                    summary += f"- {condition.capitalize()} condition:\n"
                    summary += f"  - Speed range: {speed_stats[condition]['speed_range'][0]:.2f} to {speed_stats[condition]['speed_range'][1]:.2f} knots\n"
                    summary += f"  - Consumption range: {speed_stats[condition]['consumption_range'][0]:.2f} to {speed_stats[condition]['consumption_range'][1]:.2f} mT/d\n"
                    summary += f"  - Slope of fit: {speed_stats[condition]['slope']:.4f}\n"
                    summary += f"  - R-squared: {speed_stats[condition]['r_squared']:.4f}\n"
    
    return summary

# Function to handle user query and return analysis
def handle_user_query(query: str):
    vessel_name = clean_vessel_name(extract_vessel_name(query))
    if not vessel_name:
        return "I couldn't identify a vessel name in your query."

    # Get the decision from the LLM (ChatGPT)
    llm_decision = get_llm_decision(query)
    data_summary = generate_data_summary(vessel_name, llm_decision['decision'])
    analysis = get_llm_analysis(query, vessel_name, data_summary)

    # Display charts based on the decision
    display_charts(llm_decision['decision'], vessel_name)
    return analysis

# Function to display the charts based on the LLM's decision
def display_charts(decision: str, vessel_name: str):
    if decision in ["speed_consumption", "combined_performance"]:
        try:
            speed_chart, _ = analyze_speed_consumption(vessel_name)
            if speed_chart is not None and hasattr(speed_chart, 'savefig'):
                st.pyplot(speed_chart)
            else:
                st.warning("Speed consumption chart is not available for this vessel.")
        except Exception as e:
            st.error(f"An error occurred while generating the speed consumption chart: {str(e)}")
    
    if decision in ["hull_performance", "combined_performance"]:
        try:
            hull_chart, _, _ = analyze_hull_performance(vessel_name)
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
