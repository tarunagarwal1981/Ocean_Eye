import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from datetime import datetime, timedelta
from utils.database_utils import fetch_hull_performance_data
from utils.nlp_utils import clean_vessel_name, extract_vessel_name, get_llm_analysis
from agents.base_agent import Agent
import logging

logger = logging.getLogger(__name__)

class HullPerformanceAgent(Agent):
    def __init__(self):
        self.chart = None

    def process_query(self, query: str, engine):
        vessel_name = clean_vessel_name(extract_vessel_name(query))
        if not vessel_name:
            return "I couldn't identify a vessel name in your query. Could you please provide a specific vessel name?"

        from_date, to_date = self.extract_date_range(query)
        
        data = fetch_hull_performance_data(vessel_name, engine, from_date, to_date)
        if not data:
            return f"I'm sorry, but I couldn't find any hull performance data for the vessel '{vessel_name}' in the specified time range. Could you please check the vessel name and dates, then try again?"

        data_summary = self.generate_data_summary(data)
        self.chart = self.create_chart(data)
        analysis = self.analyze_hull_condition(data)
        
        llm_analysis = get_llm_analysis(query, vessel_name, data_summary, "hull performance")
        
        return f"{llm_analysis}\n\n{analysis}"

    def extract_date_range(self, query):
        # TODO: Implement logic to extract from_date and to_date from query
        # For now, we'll use the last 6 months as default
        to_date = datetime.now()
        from_date = to_date - timedelta(days=180)
        return from_date, to_date

    def generate_data_summary(self, data):
        df = pd.DataFrame(data, columns=['vessel_name', 'report_date', 'hull_roughness_power_loss'])
        df['report_date'] = pd.to_datetime(df['report_date'])
        df = df.sort_values('report_date')
        df = df.dropna()

        summary = f"Data from {df['report_date'].min().strftime('%Y-%m-%d')} to {df['report_date'].max().strftime('%Y-%m-%d')}\n"
        summary += f"Number of data points: {len(df)}\n"
        summary += f"Average hull roughness power loss: {df['hull_roughness_power_loss'].mean():.2f}%\n"
        summary += f"Minimum hull roughness power loss: {df['hull_roughness_power_loss'].min():.2f}%\n"
        summary += f"Maximum hull roughness power loss: {df['hull_roughness_power_loss'].max():.2f}%\n"

        return summary

    def create_chart(self, data):
        df = pd.DataFrame(data, columns=['vessel_name', 'report_date', 'hull_roughness_power_loss'])
        df['report_date'] = pd.to_datetime(df['report_date'])
        df = df.sort_values('report_date')
        df = df.dropna()

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_facecolor('#000C20')
        fig.patch.set_facecolor('#000C20')

        # Scatter plot
        ax.scatter(df['report_date'], df['hull_roughness_power_loss'], color='#00FFFF', s=50, alpha=0.8)

        # Best fit line
        x = (df['report_date'] - df['report_date'].min()).dt.days
        slope, intercept, _, _, _ = stats.linregress(x, df['hull_roughness_power_loss'])
        line = slope * x + intercept
        ax.plot(df['report_date'], line, color='#FF00FF', linewidth=2)

        ax.set_xlabel('Dates', color='white')
        ax.set_ylabel('Excess Power % (compared to baseline)', color='white')
        ax.set_title('Hull Roughness Power Loss Over Time', color='white')

        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')

        plt.xticks(rotation=45)
        plt.tight_layout()

        return fig

    def analyze_hull_condition(self, data):
        df = pd.DataFrame(data, columns=['vessel_name', 'report_date', 'hull_roughness_power_loss'])
        df['report_date'] = pd.to_datetime(df['report_date'])
        df = df.sort_values('report_date')
        df = df.dropna()

        x = (df['report_date'] - df['report_date'].min()).dt.days
        slope, intercept, _, _, _ = stats.linregress(x, df['hull_roughness_power_loss'])
        
        last_date = df['report_date'].max()
        days_since_start = (last_date - df['report_date'].min()).days
        excess_power = slope * days_since_start + intercept

        analysis = f"Excess Power % = {excess_power:.2f}%\n\n"

        if excess_power < 15:
            condition = "good"
            recommendation = "Continue to monitor the hull condition of the vessel."
        elif 15 <= excess_power <= 25:
            condition = "average"
            recommendation = "Propeller polishing can be considered. Continue to monitor the hull condition."
        else:
            condition = "poor"
            recommendation = "Hull cleaning and propeller polishing are recommended."

        analysis += f"The hull condition is {condition}. {recommendation}"

        return analysis

    def display_charts(self, st):
        if self.chart is not None:
            st.pyplot(self.chart)
        else:
            st.write("No chart available for hull performance.")

    def generate_report_section(self):
        return "## Hull Performance Report\n\n[Hull performance details here]"
