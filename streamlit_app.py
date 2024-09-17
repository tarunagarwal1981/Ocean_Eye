import streamlit as st
import openai
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import Dict, Any, Tuple
from utils.database_utils import fetch_data_from_db
from utils.nlp_utils import extract_vessel_name, clean_vessel_name
from scipy.optimize import curve_fit

# Initialize OpenAI API
def get_api_key():
    if 'openai' in st.secrets:
        return st.secrets['openai']['api_key']
    api_key = os.getenv('OPENAI_API_KEY')
    if api_key is None:
        raise ValueError("API key not found. Set OPENAI_API_KEY as an environment variable.")
    return api_key

openai.api_key = get_api_key()

# Hull Performance Functions
def fetch_performance_data(vessel_name):
    query = f"""
    SELECT vessel_name, report_date, hull_roughness_power_loss
    FROM hull_performance
    WHERE UPPER(vessel_name) = '{vessel_name.upper()}'
    """
    return fetch_data_from_db(query)

def fetch_six_months_data(vessel_name):
    query = f"""
    SELECT vessel_name, hull_rough_power_loss_pct_ed
    FROM hull_performance_six_months
    WHERE UPPER(vessel_name) = '{vessel_name.upper()}'
    """
    return fetch_data_from_db(query)

def plot_hull_roughness(vessel_name, data):
    if data.empty:
        return None
    
    data['report_date'] = pd.to_datetime(data['report_date'], errors='coerce')
    today = datetime.today().date()
    six_months_ago = today - timedelta(days=180)
    
    filtered_data = data[(data['report_date'].dt.date >= six_months_ago) & (data['hull_roughness_power_loss'].notnull())]
    
    if filtered_data.empty:
        return None
    
    dates = pd.to_datetime(filtered_data['report_date'])
    power_loss = filtered_data['hull_roughness_power_loss']
    
    x_numeric = (dates - dates.min()).dt.days
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(dates, power_loss, c='cyan', edgecolors='white', s=50, alpha=0.8)
    
    coeffs = np.polyfit(x_numeric, power_loss, 1)
    best_fit_line = np.poly1d(coeffs)
    
    x_smooth = np.linspace(x_numeric.min(), x_numeric.max(), 200)
    ax.plot(dates.min() + pd.to_timedelta(x_smooth, unit='D'), best_fit_line(x_smooth), color='#00FF00', linewidth=2, linestyle='-', label='Best Fit Line')
    
    ax.set_facecolor('#000C20')
    fig.patch.set_facecolor('#000C20')
    
    ax.set_xlabel('Dates', fontsize=12, color='white')
    ax.set_ylabel('Excess Power %', fontsize=12, color='white')
    ax.set_title(f'Hull Roughness Power Loss - {vessel_name}', fontsize=14, color='white')
    
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%b'))
    ax.xaxis.set_major_locator(plt.matplotlib.dates.MonthLocator())
    
    plt.xticks(color='white', fontsize=10)
    plt.yticks(color='white', fontsize=10)
    
    ax.set_xlim(dates.min(), dates.max())
    ax.set_ylim(power_loss.min() - 0.05 * (power_loss.max() - power_loss.min()), power_loss.max() + 0.05 * (power_loss.max() - power_loss.min()))
    
    ax.legend(loc='upper left', fontsize=10, frameon=False, facecolor='none', edgecolor='none', labelcolor='white')
    
    return fig

def get_hull_condition(power_loss_pct):
    if power_loss_pct > 25:
        return "Poor"
    elif 15 <= power_loss_pct <= 25:
        return "Average"
    else:
        return "Good"

def analyze_hull_performance(vessel_name):
    performance_data = fetch_performance_data(vessel_name)
    six_months_data = fetch_six_months_data(vessel_name)
    
    chart = plot_hull_roughness(vessel_name, performance_data)
    
    if not six_months_data.empty:
        power_loss_pct_ed = six_months_data['hull_rough_power_loss_pct_ed'].iloc[-1]
        hull_condition = get_hull_condition(power_loss_pct_ed)
    else:
        power_loss_pct_ed = None
        hull_condition = None
    
    return chart, power_loss_pct_ed, hull_condition

# Speed Consumption Functions
def fetch_speed_consumption_data(vessel_name):
    query = f"""
    SELECT vessel_name, report_date, speed, normalised_consumption, loading_condition
    FROM hull_performance
    WHERE UPPER(vessel_name) = '{vessel_name.upper()}'
    """
    return fetch_data_from_db(query)

def plot_speed_consumption(vessel_name, data):
    if data.empty:
        logging.warning("Input data is empty.")
        return None
    
    data['report_date'] = pd.to_datetime(data['report_date'], errors='coerce')
    today = datetime.today().date()
    six_months_ago = today - timedelta(days=180)
    
    filtered_data = data[(data['report_date'].dt.date >= six_months_ago)]
    
    if filtered_data.empty:
        logging.warning("Filtered data is empty.")
        return None
    
    laden_data = filtered_data[filtered_data['loading_condition'].str.lower() == 'laden']
    ballast_data = filtered_data[filtered_data['loading_condition'].str.lower() == 'ballast']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    def exp_func_2nd_order(x, a, b, c, d, e):
        return a * np.exp(b * x) + c * np.exp(d * x) + e
    
    for ax, condition_data, title in [(ax1, laden_data, 'Laden Condition'), (ax2, ballast_data, 'Ballast Condition')]:
        if not condition_data.empty:
            dates = pd.to_datetime(condition_data['report_date'])
            x = condition_data['speed'].values
            y = condition_data['normalised_consumption'].values
            
            scatter = ax.scatter(x, y, c=(dates - dates.min()).dt.days, cmap='viridis', s=50, alpha=0.8)
            
            try:
                logging.info(f"Attempting to fit 2nd order exponential curve for {title}")
                logging.info(f"Data shape: x={x.shape}, y={y.shape}")
                logging.info(f"x range: {x.min()} to {x.max()}")
                logging.info(f"y range: {y.min()} to {y.max()}")
                
                # Fit 2nd order exponential curve
                popt, pcov = curve_fit(exp_func_2nd_order, x, y, p0=[1, 0.1, 1, 0.1, 1], maxfev=10000)
                
                # Generate points for smooth curve
                x_smooth = np.linspace(x.min(), x.max(), 100)
                y_smooth = exp_func_2nd_order(x_smooth, *popt)
                
                # Plot 2nd order exponential best fit curve
                ax.plot(x_smooth, y_smooth, 'r-', label='2nd Order Exponential Fit')
                logging.info(f"Successfully plotted 2nd order exponential fit for {title}")
            except RuntimeError as e:
                logging.error(f"Error fitting 2nd order exponential curve for {title}: {str(e)}")
                logging.info("Falling back to polynomial fit")
                try:
                    # Fallback to polynomial fit
                    z = np.polyfit(x, y, 3)
                    p = np.poly1d(z)
                    x_smooth = np.linspace(x.min(), x.max(), 100)
                    ax.plot(x_smooth, p(x_smooth), 'g-', label='Polynomial Fit (Fallback)')
                    logging.info(f"Successfully plotted polynomial fit for {title}")
                except Exception as poly_e:
                    logging.error(f"Error fitting polynomial curve: {str(poly_e)}")
            except Exception as e:
                logging.error(f"Unexpected error in curve fitting for {title}: {str(e)}")
            
            ax.legend(fontsize=8)
            ax.set_title(title)
            ax.set_xlabel('Speed (knots)')
            ax.set_ylabel('ME Consumption (mT/d)')
            plt.colorbar(scatter, ax=ax, label="Time Progression (days)")
    
    plt.tight_layout()
    fig.suptitle(f"Speed vs Consumption - {vessel_name}", fontsize=16)
    plt.subplots_adjust(top=0.93)
    return fig
def analyze_speed_consumption(vessel_name):
    speed_data = fetch_speed_consumption_data(vessel_name)
    chart = plot_speed_consumption(vessel_name, speed_data)
    return chart

# LLM Prompts and Functions
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

def generate_data_summary(vessel_name: str, decision: str) -> str:
    summary = f"Vessel Name: {vessel_name}\n"
    
    if decision in ["hull_performance", "combined_performance"]:
        chart, power_loss, hull_condition = analyze_hull_performance(vessel_name)
        summary += "Hull Performance Data:\n"
        summary += f"- Chart available: {'Yes' if chart is not None else 'No'}\n"
        summary += "- Current excess power: {}\n".format(
            f"{power_loss:.2f}%" if power_loss is not None else "Not available"
        )
        summary += f"- Hull condition: {hull_condition if hull_condition is not None else 'Not available'}\n"
    
    if decision in ["speed_consumption", "combined_performance"]:
        speed_chart = analyze_speed_consumption(vessel_name)
        summary += "Speed Consumption Data:\n"
        summary += f"- Chart available: {'Yes' if speed_chart is not None else 'No'}\n"
    
    return summary

def handle_user_query(query: str) -> Tuple[str, str, str]:
    vessel_name = clean_vessel_name(extract_vessel_name(query))
    if not vessel_name:
        return "I couldn't identify a vessel name in your query. Could you please provide a specific vessel name?", "general_info", None

    llm_decision = get_llm_decision(query)
    data_summary = generate_data_summary(vessel_name, llm_decision['decision'])
    analysis = get_llm_analysis(query, vessel_name, data_summary)

    return analysis, llm_decision['decision'], vessel_name

def display_charts(decision: str, vessel_name: str):
    if decision in ["hull_performance", "combined_performance"]:
        chart, _, _ = analyze_hull_performance(vessel_name)
        if chart:
            st.pyplot(chart)
    if decision in ["speed_consumption", "combined_performance"]:
        speed_chart = analyze_speed_consumption(vessel_name)
        if speed_chart:
            st.pyplot(speed_chart)

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

        if vessel_name:
            display_charts(decision, vessel_name)

if __name__ == "__main__":
    main()
