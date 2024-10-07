# Decision Procedures and Cycles

## Overview

This document provides an overview of the decision procedures and cycles used in our project. The decision cycle is a structured process that an agent follows to make decisions based on observations, planning, selection, and execution.

## CoALA Decision Cycle

The CoALA Decision Cycle is an abstract base class that defines the structure of the decision-making process. It includes the following phases:

1. **Observation Phase**: The agent observes the environment and gathers data.
2. **Planning Loop**: The agent alternates between proposing and evaluating actions until a certain condition is met.
3. **Selection Phase**: The agent selects the best action based on the evaluated actions.
4. **Execution Phase**: The agent executes the selected action.
5. **Return to Planning Loop**: The cycle repeats, starting from the observation phase.

## Decision Procedure

The decision procedure is a set of methods that the agent uses to propose, evaluate, and select actions. These methods are implemented in the `CoalaDecisionCycle` class.

### Methods

- **observation\_phase(data)**: Gathers data from the environment.
- **propose\_actions(agent, data)**: Proposes possible actions based on the observed data.
- **evaluate\_actions(agent, actions)**: Evaluates the proposed actions.
- **select\_action(agent, actions)**: Selects the best action from the evaluated actions.
- **execution\_phase(agent, action)**: Executes the selected action.
- **should\_break\_planning\_loop(agent, data)**: Determines whether to break the planning loop.
- **should\_break\_decision\_cycle(agent, data)**: Determines whether to break the decision cycle.

## Example Usage

Here is an example of how to use the `CoalaDecisionCycle` class:

```python
from cognitive_base.decision_procedures.coala_decision_cycle import CoalaDecisionCycle


class MyDecisionCycle(CoalaDecisionCycle):
    def observation_phase(self, data):
        # Custom observation logic
        return data

    def propose_actions(self, agent, data):
        # Custom proposal logic
        return ["action1", "action2"]

    def evaluate_actions(self, agent, actions):
        # Custom evaluation logic
        return {action: len(action) for action in actions}

    def select_action(self, agent, actions):
        # Custom selection logic
        return max(actions, key=actions.get)

    def execution_phase(self, agent, action):
        # Custom execution logic
        print(f"Executing {action}")

    def should_break_planning_loop(self, agent):
        # Custom condition to break the planning loop
        return True
    
    def should_break_decision_cycle(self, agent):
        # Custom condition to break the decision cycle
        return True

# Initialize and run the decision cycle
agent = None  # Replace with actual agent
data = {"key": "value"}
cycle = MyDecisionCycle()
cycle.run_cycle(agent, data)
```