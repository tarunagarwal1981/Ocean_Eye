from abc import ABC, abstractmethod

class Agent(ABC):
    @abstractmethod
    def process_query(self, query: str, engine):
        pass

    @abstractmethod
    def display_charts(self, st):
        pass
