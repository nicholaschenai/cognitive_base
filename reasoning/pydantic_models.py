from typing import List, Union, Dict, Any
from langchain_core.pydantic_v1 import BaseModel, Field


class BaseRule(BaseModel):
    rigid_conditions: Dict[str, Union[str, List[str]]] = Field(
        description="Criteria that must be met exactly for the rule to apply"
    )
    flexible_conditions: Dict[str, Union[str, List[str]]] = Field(
        description="Criteria that must be met via semantic similarity for the rule to apply"
    )
    actions: List[str] = Field(description="The actions that follow once the conditions are met")


class BaseRules(BaseModel):
    __root__: List[BaseRule]


# deprecated
class ProceduralRule:
    """
    Represents a procedural rule where conditions are in a list

    Args:
        conditions (list): A list of conditions for the rule.
        action (str): The action to be performed if the conditions are met.
        weights (list, optional): A list of weights for each condition. Defaults to None.
        priority (int, optional): The priority of the rule. Defaults to 1.
        rigidity (int, optional): The rigidity of the rule. Defaults to 1.
    """
    def __init__(self, conditions, action, weights=None, priority=1, rigidity=1):
        self.conditions = conditions
        self.action = action
        self.weights = weights if weights else [1] * len(conditions)
        self.priority = priority
        # self.embedding = get_embedding(conditions)
        self.rigidity = rigidity
