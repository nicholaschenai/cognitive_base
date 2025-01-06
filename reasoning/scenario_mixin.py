"""
LM reason about scenarios where this code is useful to help it connect to future tasks

CoALA: "reason about 'in what situations would this knowledge be useful?', 
and append the reasoning results to the knowledge to help later connect the knowledge
to new situations."
"""
from pydantic import BaseModel, Field
from typing import List, TypeVar, Generic

from ..utils.formatting import dict_indent_format
from .base_lm_reasoning import BaseLMReasoning

T = TypeVar('T', bound=BaseLMReasoning)

DEFAULT_SCENARIO_SYS_PROMPT = """
You are a helpful assistant that extracts important information from problems and solutions to aid in searchability and education.

## Instructions
- Read through the problem and solution carefully.
- Reason out, step by step, in what scenarios this knowledge would be useful.
- Focus on general patterns and situations to help connect this knowledge to new situations.
- Then, include a list of keywords that are useful for searching this problem/solution pair.

## Response format
Respond in JSON, and follow the keys and expected format of the values strictly.

{format_instructions}
"""

DEFAULT_SCENARIO_H_TEMPLATE = """
[Problem]
{task}
[/Problem]

[Solution]
{code}
[/Solution]
"""


class Concept(BaseModel):
    scenarios: str = Field(description="In what scenarios will this knowledge be useful? Answer in a paragraph.")
    keywords: List[str] = Field(description="List of keywords that are useful for searching this problem/solution pair")


class ScenarioMixin(Generic[T]):
    # Reference the default prompts defined above
    _default_scenario_sys_prompt = DEFAULT_SCENARIO_SYS_PROMPT
    _default_scenario_h_template = DEFAULT_SCENARIO_H_TEMPLATE

    @property
    def scenario_sys_prompt(self: T) -> str:
        """Returns instance's scenario_sys_prompt if set, otherwise returns default"""
        return getattr(self, '_scenario_sys_prompt', self._default_scenario_sys_prompt)
    
    @property
    def scenario_h_template(self: T) -> str:
        """Returns instance's scenario_h_template if set, otherwise returns default"""
        return getattr(self, '_scenario_h_template', self._default_scenario_h_template)

    def get_scenario(self: T, task: str, code: str, llm=None) -> dict:
        """
        LM reason about scenarios where this code is useful to help it connect to future tasks
        
        CoALA: "reason about 'in what situations would this knowledge be useful?', 
        and append the reasoning results to the knowledge to help later connect the knowledge
        to new situations."
        """
        scenario_out = self.lm_reason(
            sys_template=self.scenario_sys_prompt,
            human_template=dict_indent_format(
                self.scenario_h_template, 
                {'task': task, 'code': code}
            ),
            llm=llm,
            structured=True,
            pydantic_model=Concept
        )

        return scenario_out
