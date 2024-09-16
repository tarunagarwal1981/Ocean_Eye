import streamlit as st
import openai
import os
import pandas as pd
from datetime import datetime, timedelta
from utils.nlp_utils import process_user_input, extract_vessel_name, clean_vessel_name
from modules.hull_performance import analyze_hull_performance, fetch_performance_data, fetch_six_months_data
from modules.speed_consumption import analyze_speed_consumption

def get_api_key():
    if 'openai' in st.secrets:
        return st.secrets['openai']['api_key']
    api_key = os.getenv('OPENAI_API_KEY')
    if api_key is None:
        raise ValueError("API key not found. Set OPENAI_API_KEY as an environment variable.")
    return api_key

openai.api_key = get_api_key()

FEW_SHOT_PROMPT = """
You are an AI assistant specialized in vessel performance analysis. Here are some example Q&As:

Q: What's the hull performance of the vessel Oceanica Explorer?
A: I've analyzed the hull performance data for Oceanica Explorer. Here's what I found:

1. Current Excess Power: 8.7%. This means the vessel currently requires 8.7% more power to maintain its speed compared to a clean hull condition.
2. Fouling Rate: Approximately 0.5% increase in power loss per month over the last 6 months, indicating a moderate level of fouling accumulation.
3. Forecasted Hull Cleaning: Scheduled for 2023-11-15. It's important to plan operations around this date to optimize performance.
4. Data Confidence: We have sufficient valid data points after the last hull cleaning event to make these assessments with high confidence.
5. Performance Impact: The 8.7% excess power requirement translates to approximately 6-7% increased fuel consumption, assuming typical operating conditions.

Recommendations:
1. Monitor the hull condition closely as you approach the forecasted cleaning date.
2. Consider performing underwater hull inspections to validate the fouling rate and adjust the cleaning date if necessary.
3. Implement operational measures like speed optimization to mitigate the impact of increased power requirements.
4. After the next hull cleaning, ensure proper data collection to maintain accurate performance tracking.

Q: Can you provide the hull performance analysis for the vessel Starlight Voyager?
A: I've looked into the hull performance data for Starlight Voyager. Here's what I found:

1. Current Excess Power: Data unavailable. We don't have enough valid data points after the last hull cleaning event to calculate a reliable current excess power figure.
2. Fouling Rate: Despite the lack of a current excess power figure, the data shows an average increase of approximately 0.3% in power loss per month over the last 6 months.
3. Forecasted Hull Cleaning: Date not available due to insufficient data to accurately predict the next optimal cleaning date.
4. Data Confidence: Our confidence in the current hull performance assessment is low due to insufficient recent data.
5. Performance Impact: Without a current excess power figure, we can't accurately quantify the performance impact. However, based on the fouling rate, we can estimate that the impact is likely increasing over time.

Recommendations:
1. Urgently review the data collection process for Starlight Voyager to ensure all required performance data is being recorded accurately.
2. Conduct a manual hull inspection to assess the current condition, as we lack reliable data-driven insights.
3. If a hull cleaning has been performed recently, ensure that this event is properly recorded in the system to reset the performance baseline.
4. Implement a more rigorous data validation process to prevent gaps in critical performance metrics.
5. Consider using historical data or fleet averages to estimate the current hull condition until more reliable data is available.

Now, please answer the following question in a similar style, using the data I provide:
{user_question}

{data_summary}

Provide a detailed analysis and recommendations based on this data.
"""

def get_gpt_response(prompt):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an AI assistant specialized in vessel performance."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            n=1,
            stop=None,
            temperature=0.7,
        )
        return response.choices[0].message['content'].strip()
    except Exception as e:
        st.error(f"Error in GPT response: {str(e)}")
        return "I'm sorry, I encountered an error while processing your request."

def calculate_fouling_rate(performance_data):
    if performance_data.empty:
        return 0
    
    performance_data['report_date'] = pd.to_datetime(performance_data['report_date'])
    performance_data = performance_data.sort_values('report_date')
    
    latest_date = performance_data['report_date'].max()
    six_months_ago = latest_date - timedelta(days=180)
    
    last_six_months_data = performance_data[performance_data['report_date'] > six_months_ago]
    
    if len(last_six_months_data) < 2:
        return 0  # Not enough data points in the last 6 months
    
    first_date = last_six_months_data['report_date'].iloc[0]
    last_date = last_six_months_data['report_date'].iloc[-1]
    first_power_loss = last_six_months_data['hull_roughness_power_loss'].iloc[0]
    last_power_loss = last_six_months_data['hull_roughness_power_loss'].iloc[-1]
    
    days_difference = (last_date - first_date).days
    if days_difference == 0:
        return 0
    
    monthly_rate = (last_power_loss - first_power_loss) / days_difference * 30  # Assuming 30 days per month
    return monthly_rate

def handle_user_query(user_input: str) -> str:
    intent, vessel_present = process_user_input(user_input)
    vessel_name = extract_vessel_name(user_input)
    cleaned_vessel_name = clean_vessel_name(vessel_name)

    if not cleaned_vessel_name:
        return "I couldn't identify a vessel name in your query. Could you please provide a specific vessel name?"

    if intent == "hull_performance":
        return process_hull_performance(cleaned_vessel_name)
    elif intent == "speed_consumption":
        return process_speed_consumption(cleaned_vessel_name)
    elif intent == "hull_performance_and_speed_consumption" or intent == "vessel_performance":
        return process_combined_performance(cleaned_vessel_name)
    else:
        return get_gpt_response(FEW_SHOT_PROMPT.format(
            user_question=user_input,
            data_summary=f"Vessel Name: {cleaned_vessel_name}\nIntent: General vessel performance query"
        ))

def process_hull_performance(vessel_name: str) -> str:
    try:
        chart, power_loss, hull_condition = analyze_hull_performance(vessel_name)
        performance_data = fetch_performance_data(vessel_name)
        six_months_data = fetch_six_months_data(vessel_name)
        
        if chart:
            st.pyplot(chart)
        
        if power_loss is None or hull_condition is None:
            return f"Sorry, I couldn't find specific hull performance data for {vessel_name}."
        
        fouling_rate = calculate_fouling_rate(performance_data)
        forecasted_cleaning = six_months_data['forecasted_hull_cleaning_date'].iloc[0] if not six_months_data.empty else None
        
        data_summary = f"""
        Vessel Name: {vessel_name}
        Current Excess Power: {power_loss:.2f}%
        Hull Condition: {hull_condition}
        Fouling Rate (last 6 months): {fouling_rate:.2f}% per month
        Forecasted Hull Cleaning: {forecasted_cleaning if forecasted_cleaning else 'Not available'}
        """
        
        gpt_response = get_gpt_response(FEW_SHOT_PROMPT.format(
            user_question=f"Analyze the hull performance of {vessel_name}",
            data_summary=data_summary
        ))
        
        return gpt_response
    
    except Exception as e:
        st.error(f"An error occurred while processing hull performance: {str(e)}")
        return "I encountered an error while analyzing hull performance. Please try again or check if the vessel name is correct."

def process_speed_consumption(vessel_name: str) -> str:
    try:
        chart = analyze_speed_consumption(vessel_name)
        if chart:
            st.pyplot(chart)
            return get_gpt_response(FEW_SHOT_PROMPT.format(
                user_question=f"Analyze the speed consumption profile of {vessel_name}",
                data_summary=f"Vessel Name: {vessel_name}\nSpeed consumption data is available and plotted."
            ))
        else:
            return f"Sorry, I couldn't find specific speed consumption data for {vessel_name}."
    except Exception as e:
        st.error(f"An error occurred while processing speed consumption: {str(e)}")
        return "I encountered an error while analyzing speed consumption. Please try again or check if the vessel name is correct."

def process_combined_performance(vessel_name: str) -> str:
    hull_response = process_hull_performance(vessel_name)
    speed_response = process_speed_consumption(vessel_name)
    
    combined_prompt = f"""
    Provide a combined analysis of hull performance and speed consumption for the vessel {vessel_name}.
    
    Hull Performance Analysis:
    {hull_response}
    
    Speed Consumption Analysis:
    {speed_response}
    
    Summarize the overall vessel performance and provide recommendations based on both analyses.
    """
    
    return get_gpt_response(combined_prompt)

def main():
    st.title("Vessel Performance Chatbot")

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What would you like to know about vessel performance?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        response = handle_user_query(prompt)

        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)

if __name__ == "__main__":
    main()
