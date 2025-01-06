"""
Example CoALA reasoning module that implements core cognitive functions:
- Code generation
- Summarization 
- Reflection
- Critique

This module demonstrates how to use BaseLMReasoning for implementing cognitive capabilities
as described in the CoALA paper.
"""
import logging

from typing import Dict, Any, Tuple, List

from ..voyager_coder.base_coding_module import BaseCodingModule
from .utils.coala_message_thread import CoalaMessageThread

logger = logging.getLogger("logger")

# adapted from USACO-Bench prompt
sys_prompt = """
You are an expert programmer tasked with solving the problem below.

Follow the user's instructions to solve the problem.
The user can ask for analysis, code, or both.

## Instructions
When asked to output code,
- Make sure to wrap your code in '```python' and '```' Markdown delimiters, 
- include exactly one block of code with the entire solution
- No outside libraries are allowed. 
- Builtins are allowed.
- If the task specifies a function name to be used, follow it strictly (be case sensitive!) and declare the that function last (helper functions are declared first).
- DO NOT write any assert statements / tests.

## Problem
[BEGIN PROBLEM]
{}
[END PROBLEM]
"""

human_initial_prompt = """
## Instructions

Reason through the problem and:
1. Restate the problem in plain English
2. Conceptualize a solution first in plain English
3. Write a pseudocode solution
4. Output the final Python 3 solution with your solution steps in comments.

Make sure to wrap your code in '```python' and '```' Markdown delimiters, 
and include exactly one block of code with the entire solution.

No outside libraries are allowed.
Builtins are allowed.
"""

human_prompt = """
## Instructions
Now, given your insights, try to fix the solution. 
Output a block of correct python3 code to be executed and evaluated again. 

Make sure to wrap your code in '```python' and '```' Markdown delimiters.
"""

critique_prompt = """
The code doesn't pass all the tests.

## Instructions
- First, think step-by-step about why your code is wrong.
- Then, think step-by-step about where you went wrong in your latest solution.
"""

summarize_prompt = """
## Instructions
Now, summarize your attempts as a **standalone** document for your own future reference. (can skip formalities, be concise)

- Only include the most relevant information and code snippets that could be useful in the future.
- If you did not learn much from the current experience (e.g. the problem was too easy), feel free to write less or nothing at all.
- Here are some ideas if the experience was useful:
    - Identifying the core concepts and patterns that problem appears to test for
    - Describing the strategies that seemed to work but did not, and vice versa
    - Describing unexpected errors and how they were fixed

Your summary should be at most a few paragraphs.
"""

reflection_prompt = """
Below is the official solution (do exercise discretion as official solutions can also have mistakes).

You will be reflecting on your attempts to solve the problem.
This will be a **standalone** document for your own future reference. (can skip formalities, be concise)

## Instructions
- Only include the most relevant insights and code snippets that could be useful in the future.
- If you did not learn much from the current experience (e.g. the problem was too easy / official solution was obvious), feel free to write less or nothing at all.
- Here are some ideas if the experience was useful:
    - If the official solution is insightful (e.g. better than yours in time and space complexity or more effective in breaking down the problem statement), distil the key approach of the official solution, step by step.
    - If your approach failed the test cases, reason why the official solution works but yours does not

Your reflection should be at most a few paragraphs.

## Official solution
{}
"""

mem_prompt = """
Also, here are some of your memories.
Feel free to use the given information to aid your problem solving process if necessary.
Do not confuse the memories with the problem statement.

## Memories
{}
"""


class CoalaReasoning(BaseCodingModule):
    """
    A reasoning module implementing core cognitive functions for the CoALA agent.
    """
    def __init__(
        self,
        model_name: str = "gpt-3.5-turbo",
        temperature: float = 0,
        request_timeout: int = 120,
        verbose: bool = True,
        callbacks = None,
        debug_mode: bool = False,
        name: str = 'coala_reasoning',
        generic_code_env: bool = False,
        agent_type: str = 'coala',
        **kwargs
    ):
        """Initialize the CoALA reasoning module"""
        super().__init__(
            model_name=model_name,
            temperature=temperature,
            request_timeout=request_timeout,
            verbose=verbose,
            callbacks=callbacks,
            debug_mode=debug_mode,
            name=name,
            generic_code_env=generic_code_env,
            **kwargs
        )

        self.agent_type = agent_type

    """
    Reasoning Actions (from and to working mem)
    """
    def gen_code(
        self,
        message_thread: CoalaMessageThread,
        retrieved: List[str],
        attempt_idx: int,
    ) -> Tuple[Dict[str, Any], str]:
        """
        Generate code 
        
        Returns:
            Dict containing the generated code and metadata
        """
        if attempt_idx > 0:
            message_thread.add("user", human_prompt)
        
        # Add retrieved memories as temporary message
        if retrieved:
            message_thread.add("user", mem_prompt.format('\n'.join(retrieved)), is_temp=True)
        
        parsed_result = self.lm_reason(
            messages=message_thread.to_msg_with_temp(),
            parse_fn=self.parse_ai_code,
        )

        message_thread.record_raw_msg(parsed_result)
        return parsed_result

    def reflect(
        self,
        message_thread: CoalaMessageThread,
        official_solution: str
    ) -> str:
        """Generate reflection on solution attempt"""
        message_thread.add("user", reflection_prompt.format(official_solution), is_temp=True)
        reflection = self.lm_reason(messages=message_thread.to_msg_with_temp())
        message_thread.record_reflection(reflection)
        return reflection

    def summarize(
        self,
        message_thread: CoalaMessageThread
    ) -> str:
        """Generate a summary of the solution attempt"""
        message_thread.add("user", summarize_prompt, is_temp=True)
        summary = self.lm_reason(messages=message_thread.to_msg_with_temp())
        message_thread.record_summary(summary)
        return summary

    def initial_solve(self) -> Tuple[str, CoalaMessageThread]:
        """
        Initial attempt to solve the problem without any retrieved context.
        Similar to solve_prompt_fn in USACO prompts.
        """
        message_thread = CoalaMessageThread()
        message_thread.add("system", sys_prompt.format(self.task_prompt))
        message_thread.add("user", human_initial_prompt)

        if self.agent_type == 'react':
            return '', message_thread

        initial_soln = self.lm_reason(messages=message_thread.to_msg())
        message_thread.add("assistant", initial_soln, is_canonical=False)

        return initial_soln, message_thread

    def critique(self, message_thread: CoalaMessageThread):
        message_thread.add("user", critique_prompt)
        critique = self.lm_reason(messages=message_thread.to_msg())
        message_thread.record_critique(critique)
        return critique
