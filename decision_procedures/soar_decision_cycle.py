"""
Soar decision cycle
Currently linear, future implementations should allow for parallel application of actions.
loop handled outside this class
"""
from abc import ABC, abstractmethod


class SoarDecisionCycle(ABC):
    def __init__(self, decision_procedure):
        self.decision_procedure = decision_procedure

    def run_cycle(self, agent, data):
        # Input phase
        input_data = self.input_phase(data)

        # Decision procedure phase (propose, evaluate, select)
        selected_action = self.decision_procedure.run(agent, input_data)

        # Apply phase
        self.apply_phase(agent, selected_action)

        # Output phase
        output_data = self.output_phase(agent)

        return output_data

    @abstractmethod
    def input_phase(self, data):
        pass

    @abstractmethod
    def apply_phase(self, agent, action):
        pass

    @abstractmethod
    def output_phase(self, agent):
        pass
