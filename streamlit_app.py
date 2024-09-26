import streamlit as st
import openai
from utils.database_utils import fetch_data_from_db
from agents.hull_performance_agent import analyze_hull_performance
from agents.speed_consumption_agent import analyze_speed_consumption

# Initialize OpenAI API key
def get_api_key():
    if 'openai' in st.secrets:
        return st.secrets['openai']['api_key']
    api_key = os.getenv('OPENAI_API_KEY')
    if api_key is None:
        raise ValueError("API key not found. Set OPENAI_API_KEY as an environment variable.")
    return api_key

openai.api_key = get_api_key()


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

# LLM decision-making function
def get_llm_decision(query: str):
    prompt = f"""
    You are a vessel performance analysis assistant. The user query is: '{query}'
    
    Your task is to identify whether the query relates to hull performance, speed consumption, or a combined performance of a vessel.
    The options are:
    1. Hull performance
    2. Speed consumption
    3. Combined performance (both hull and speed)
    
    Please respond in the following JSON format:
    {{
        "vessel_name": "<vessel_name>",
        "decision": "<hull_performance|speed_consumption|combined_performance>",
        "explanation": "<brief_explanation>"
    }}
    """
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a vessel performance analysis assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=300,
        temperature=0.5
    )
    
    decision_text = response['choices'][0]['message']['content']
    
    # Parse the LLM's response into JSON format
    decision_json = eval(decision_text)  # Convert string to dictionary
    
    return decision_json


# LLM detailed analysis function
def get_llm_analysis(query: str, hull_analysis: str, speed_analysis: str, hull_condition: str):
    prompt = f"""
    {FEW_SHOT_EXAMPLES}
    
    User Question: {query}

    Vessel Data:
    - Hull Analysis: {hull_analysis}
    - Speed Analysis: {speed_analysis}
    - Hull Condition: {hull_condition}

    Provide a detailed analysis and recommendations based on this data.
    """
    
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


# Main function to handle user query
def handle_user_query(query: str):
    detailed_analysis = "No detailed analysis available."  # Initialize with a default value
    
    # Get the decision and vessel name from the LLM (ChatGPT)
    llm_decision = get_llm_decision(query)
    
    vessel_name = llm_decision.get("vessel_name")
    if not vessel_name:
        return "I couldn't identify a vessel name in your query."

    # Based on the decision, call the appropriate agent
    if llm_decision['decision'] == 'hull_performance':
        # Unpack 4 values now (analysis, power_loss_pct, hull_condition, hull_chart)
        analysis, power_loss_pct, hull_condition, hull_chart = analyze_hull_performance(vessel_name)
        st.write(f"Hull performance analysis executed for {vessel_name}.")
        st.write(f"{analysis}")  # Display only the analysis summary
        
        if hull_chart is not None and hasattr(hull_chart, 'savefig'):
            st.pyplot(hull_chart)
        else:
            st.warning("Hull performance chart is not available for this vessel.")
        
        detailed_analysis = analysis  # Update detailed_analysis with the hull analysis

    elif llm_decision['decision'] == 'speed_consumption':
        analysis, speed_chart = analyze_speed_consumption(vessel_name)
        st.write(f"{analysis}")  # Display speed analysis
        
        if speed_chart is not None and hasattr(speed_chart, 'savefig'):
            st.pyplot(speed_chart)
        else:
            st.warning("Speed consumption chart is not available for this vessel.")
        
        detailed_analysis = analysis  # Update detailed_analysis with the speed analysis

    elif llm_decision['decision'] == 'combined_performance':
        # Call both hull and speed consumption agents and combine the analysis
        hull_analysis, _, hull_condition, hull_chart = analyze_hull_performance(vessel_name)
        speed_analysis, speed_chart = analyze_speed_consumption(vessel_name)
        
        # Display combined analysis
        combined_analysis = f"{hull_analysis}\n\n{speed_analysis}"
        st.write(combined_analysis)
        
        # Call the LLM to generate a comprehensive analysis
        detailed_analysis = get_llm_analysis(query, hull_analysis, speed_analysis, hull_condition)
        st.write(detailed_analysis)
        
        # Display both charts
        if hull_chart is not None and hasattr(hull_chart, 'savefig'):
            st.pyplot(hull_chart)
        else:
            st.warning("Hull performance chart is not available for this vessel.")
        
        if speed_chart is not None and hasattr(speed_chart, 'savefig'):
            st.pyplot(speed_chart)
        else:
            st.warning("Speed consumption chart is not available for this vessel.")

    else:
        st.warning("The query seems to require general vessel information or is unclear. Please refine the query.")
        detailed_analysis = "No specific analysis could be generated from the provided query."
    
    return detailed_analysis  # Now, this will always return a value



# Streamlit App Layout
def main():
    st.title("Advanced Vessel Performance Chatbot (Powered by GPT-3.5)")

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What would you like to know about vessel performance?"):
        st.session_state.messages.append({"role": "human", "content": prompt})
        with st.chat_message("human"):
            st.markdown(prompt)

        analysis = handle_user_query(prompt)

        st.session_state.messages.append({"role": "assistant", "content": analysis})
        with st.chat_message("assistant"):
            st.markdown(analysis)


if __name__ == "__main__":
    main()
