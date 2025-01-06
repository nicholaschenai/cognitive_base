from ..voyager_coder.voyager_skill import VoyagerSkill
from ...reasoning.scenario_mixin import ScenarioMixin


class CoalaDesc(VoyagerSkill, ScenarioMixin):
    def __init__(
        self,
        model_name="gpt-3.5-turbo",
        temperature=0,
        request_timeout=120,
        verbose=True,
        callbacks=None,
        debug_mode=False,
        **kwargs,
    ):
        super().__init__(
            model_name=model_name,
            desc_model_name=model_name,
            temperature=temperature,
            request_timeout=request_timeout,
            verbose=verbose,
            callbacks=callbacks,
            debug_mode=debug_mode,
            **kwargs,
        )

    def run(self, parsed_result, task):
        """
        Generates a description for a task, including conditions for any subtasks.

        Args:
            parsed_result (dict): The parsed result containing program code and name.
            task (str): The current task description.
            task_stack (list): Stack of tasks, including current task and any subtasks.

        Returns:
            str: Generated description for the task, incorporating additional insights from the language model.
        """
        program_code = parsed_result.get('program_code', '')
        program_name = parsed_result.get('program_name', '')
        desc = super().gen_code_desc(program_code, program_name=program_name, task=task)
        
        scenario_out = self.get_scenario(task, program_code)
        desc += scenario_out['scenarios']

        return desc
    