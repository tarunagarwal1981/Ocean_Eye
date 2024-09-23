# agents/agent_base.py

from abc import ABC, abstractmethod

class Agent(ABC):
    @abstractmethod
    def handle(self, query: str, vessel_name: str) -> str:
        pass
