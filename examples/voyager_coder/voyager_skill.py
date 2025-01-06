"""
Voyager skill description

heavily adapted from Voyager: An Open-Ended Embodied Agent with Large Language Models by
Wang et. al. (2023) https://github.com/MineDojo/Voyager
"""
from ...reasoning.base_lm_reasoning import BaseLMReasoning

from .voyager_skill_prompts import desc_template, skill_sys_prompt


class VoyagerSkill(BaseLMReasoning):
    def __init__(
        self,
        desc_model_name="gpt-3.5-turbo",
        temperature=0,
        request_timeout=120,
        verbose=True,
        callbacks=None,
        debug_mode=False,
        model_name="gpt-3.5-turbo",
        name='skill',
        **kwargs,
    ):
        super().__init__(
            model_name=desc_model_name,
            temperature=temperature,
            request_timeout=request_timeout,
            verbose=verbose,
            callbacks=callbacks,
            debug_mode=debug_mode,
            name=name,
            **kwargs,
        )

    """
    helper fns
    """
    def format_desc(self, code_desc, program_name, task):
        """
        Formats a description string for a given code description, program name, and task.

        Args:
            code_desc (str): The description of the code.
            program_name (str): The name of the program or function. If None or empty, no program name will be included in the formatted description.
            task (str): The task associated with the code description.

        Returns:
            str: A formatted description string.
        """
        name_str = f" for function: {program_name}" if program_name else ''
        desc = desc_template.format(task_str='', name_str=name_str, code_desc=code_desc)
        return desc

    """
    Reasoning Actions (from and to working mem)
    """

    def gen_code_desc(self, program_code, program_name="", task="", **kwargs):
        """
        Generates a description for the given program code.
        Args:
            program_code (str): The code of the program to generate a description for.
            program_name (str, optional): The name of the main function in the program. Defaults to an empty string.
            task (str, optional): The task associated with the program. Defaults to an empty string.
            **kwargs: Additional keyword arguments.
        Returns:
            str: The formatted description of the program code.
        """
        prog_str = f"\n\nThe main function is `{program_name}`." if program_name else ''

        code_desc = self.lm_reason(
            sys_template=skill_sys_prompt, 
            human_template=program_code+prog_str
        )

        formatted_desc = self.format_desc(code_desc, program_name, task)

        print(f"\033[33m generated desc for {program_name}:\n{formatted_desc}\033[0m")
        return formatted_desc
