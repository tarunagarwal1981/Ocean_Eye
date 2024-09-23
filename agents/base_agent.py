# agents/base_agent.py

from abc import ABC, abstractmethod

class Agent(ABC):
    @abstractmethod
    def process_query(self, query: str, engine):
        """
        Process the user query and return a response.
        
        :param query: The user's query string
        :param engine: The database engine for data retrieval
        :return: A string response to the user's query
        """
        pass

    @abstractmethod
    def display_charts(self, st):
        """
        Display relevant charts using Streamlit.
        
        :param st: The Streamlit object for rendering
        """
        pass

    @abstractmethod
    def generate_data_summary(self, data):
        """
        Generate a summary of the data retrieved from the database.
        
        :param data: The data retrieved from the database
        :return: A string summary of the data
        """
        pass

    @abstractmethod
    def generate_report_section(self):
        """
        Generate a section of the report specific to this agent's domain.
        
        :return: A string or object representing the report section
        """
        pass
