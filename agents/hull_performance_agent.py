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

    # ... (rest of the methods remain the same)
