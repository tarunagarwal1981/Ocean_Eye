from utils.database_utils import fetch_hull_performance_data
from utils.nlp_utils import clean_vessel_name, extract_vessel_name, get_llm_analysis
from agents.base_agent import Agent

class HullPerformanceAgent(Agent):
    def process_query(self, query: str, engine):
        vessel_name = clean_vessel_name(extract_vessel_name(query))
        if not vessel_name:
            return "I couldn't identify a vessel name in your query. Could you please provide a specific vessel name?"

        data = fetch_hull_performance_data(vessel_name, engine)
        data_summary = self.generate_data_summary(data)
        
        analysis = get_llm_analysis(query, vessel_name, data_summary, "hull performance")
        return analysis

    def generate_data_summary(self, data):
        # Implement data summary generation for hull performance
        return "Hull performance data summary"  # Placeholder

    def display_charts(self, st):
        # Implement hull performance chart display logic using Streamlit
        st.write("Hull performance chart would be displayed here")  # Placeholder

    def generate_report_section(self):
        return "## Hull Performance Report\n\n[Hull performance details here]"
