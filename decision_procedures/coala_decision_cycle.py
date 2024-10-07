"""
decision cycle from CoALA
"""
from abc import ABC, abstractmethod


class CoalaDecisionCycle(ABC):
    def run_cycle(self, agent, data):
        while True:
            # Observation phase
            observation_data = self.observation_phase(data)

            # Planning loop (alternating proposal and evaluation)
            while True:
                proposed_actions = self.propose_actions(agent, observation_data)
                evaluated_actions = self.evaluate_actions(agent, proposed_actions)
                if self.should_break_planning_loop(agent):
                    break

            # Selection phase
            selected_action = self.select_action(agent, evaluated_actions)

            # Execution phase
            self.execution_phase(agent, selected_action)

            # Return to planning loop
            if self.should_break_decision_cycle(agent):
                break

    @abstractmethod
    def observation_phase(self, data):
        pass

    @abstractmethod
    def propose_actions(self, agent, data):
        pass

    @abstractmethod
    def evaluate_actions(self, agent, actions):
        pass

    @abstractmethod
    def select_action(self, agent, actions):
        pass

    @abstractmethod
    def execution_phase(self, agent, action):
        pass

    @abstractmethod
    def should_break_planning_loop(self, agent):
        pass

    @abstractmethod
    def should_break_decision_cycle(self, agent):
        pass
