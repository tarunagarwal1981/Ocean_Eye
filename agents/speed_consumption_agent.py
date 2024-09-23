# agents/speed_consumption_agent.py

from .agent_base import Agent
from utils.database_utils import fetch_speed_consumption_data
from utils.nlp_utils import extract_vessel_name
import matplotlib.pyplot as plt
import streamlit as st

class SpeedConsumptionAgent(Agent):
    def handle(self, query: str, vessel_name: str) -> str:
        speed_data = fetch_speed_consumption_data(vessel_name)
        if speed_data.empty:
            return f"No speed consumption data available for {vessel_name}."

        chart, stats = self.analyze_speed_consumption(vessel_name, speed_data)
        if chart:
            st.pyplot(chart)
        stats_str = self.format_stats(stats)
        return f"**Speed Consumption for {vessel_name}:**\n{stats_str}"

    def analyze_speed_consumption(self, vessel_name, data):
        # Implement your speed consumption analysis logic here
        # For now, we'll use placeholder values
        stats = {
            'average_speed': data['speed'].mean(),
            'average_consumption': data['normalised_consumption'].mean()
        }

        # Generate a sample chart (you should replace this with your actual plotting code)
        fig, ax = plt.subplots()
        ax.scatter(data['speed'], data['normalised_consumption'])
        ax.set_title(f'Speed vs Consumption for {vessel_name}')
        ax.set_xlabel('Speed (knots)')
        ax.set_ylabel('Consumption (mT/d)')

        return fig, stats

    def format_stats(self, stats):
        return f"- Average Speed: {stats['average_speed']:.2f} knots\n- Average Consumption: {stats['average_consumption']:.2f} mT/d"
