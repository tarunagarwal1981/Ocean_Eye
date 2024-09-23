# agents/hull_performance_agent.py

from .agent_base import Agent
from utils.database_utils import fetch_performance_data
from utils.nlp_utils import extract_vessel_name
import matplotlib.pyplot as plt
import streamlit as st

class HullPerformanceAgent(Agent):
    def handle(self, query: str, vessel_name: str) -> str:
        performance_data = fetch_performance_data(vessel_name)
        if performance_data.empty:
            return f"No hull performance data available for {vessel_name}."

        chart, power_loss_pct_ed, hull_condition = self.analyze_hull_performance(vessel_name, performance_data)
        if chart:
            st.pyplot(chart)
        return f"**Hull Performance for {vessel_name}:**\n- Current Excess Power: {power_loss_pct_ed}%\n- Hull Condition: {hull_condition}"

    def analyze_hull_performance(self, vessel_name, data):
        # Implement your hull performance analysis logic here
        # For now, we'll use placeholder values
        power_loss_pct_ed = 10  # Example value
        hull_condition = "Average"  # Example value

        # Generate a sample chart (you should replace this with your actual plotting code)
        fig, ax = plt.subplots()
        ax.plot(data['report_date'], data['hull_roughness_power_loss'])
        ax.set_title(f'Hull Performance of {vessel_name}')
        ax.set_xlabel('Date')
        ax.set_ylabel('Power Loss (%)')

        return fig, power_loss_pct_ed, hull_condition
