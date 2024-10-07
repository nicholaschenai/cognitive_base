"""
Soar decision procedure is an abstract class that defines the structure of a decision procedure.
Currently linear, future implementations should allow for parallel proposal and evaluation of actions.
"""
from abc import ABC, abstractmethod


class SoarDecisionProcedure(ABC):
    def run(self, agent, data):
        proposed_actions = self.propose_actions(agent, data)
        evaluated_actions = self.evaluate_actions(agent, proposed_actions)
        selected_action = self.select_action(agent, evaluated_actions)
        return selected_action

    @abstractmethod
    def propose_actions(self, agent, data):
        pass

    @abstractmethod
    def evaluate_actions(self, agent, actions):
        pass

    @abstractmethod
    def select_action(self, agent, actions):
        pass
