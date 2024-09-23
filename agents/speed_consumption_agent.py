from .base_agent import Agent
from utils.database_utils import fetch_speed_consumption_data
from utils.nlp_utils import clean_vessel_name, extract_vessel_name, get_llm_analysis

class SpeedConsumptionAgent(Agent):
    def process_query(self, query: str, engine):
        vessel_name = clean_vessel_name(extract_vessel_name(query))
        if not vessel_name:
            return "I couldn't identify a vessel name in your query. Could you please provide a specific vessel name?"

        data = fetch_speed_consumption_data(vessel_name, engine)
        data_summary = self.generate_data_summary(data)
        
        analysis = get_llm_analysis(query, vessel_name, data_summary, "speed consumption")
        return analysis

    def generate_data_summary(self, data):
        # Implement data summary generation for speed consumption
        pass

    def display_charts(self, st):
        # Implement speed consumption chart display logic using Streamlit
        pass
