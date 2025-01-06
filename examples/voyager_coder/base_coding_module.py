"""
Code parsing functionality

heavily adapted from Voyager: An Open-Ended Embodied Agent with Large Language Models by
Wang et. al. (2023) https://github.com/MineDojo/Voyager
"""
import logging

from ...reasoning.base_lm_reasoning import BaseLMReasoning

from ...knowledge_sources.parsers import extract_blocks

from ...utils.formatting import truncate_str
from ...utils.code_parse import extract_from_ast, assert_modules_in_whitelist


logger = logging.getLogger("logger")


class BaseCodingModule(BaseLMReasoning):
    def __init__(
        self,
        model_name="gpt-3.5-turbo",
        temperature=0,
        request_timeout=120,
        verbose=True,
        callbacks=None,
        debug_mode=False,
        name='coding',
        generic_code_env=False,
        **kwargs,
    ):
        super().__init__(
            model_name=model_name,
            temperature=temperature,
            request_timeout=request_timeout,
            verbose=verbose,
            callbacks=callbacks,
            debug_mode=debug_mode,
            name=name,
            **kwargs,
        )

        self.gt_fn_name = ''
        self.task_prompt = ''

        self.check_imports = not generic_code_env
        self.assert_fns = not generic_code_env
        self.rebuild_code_from_ast = False

    """
    helper fns
    """
    def reset(self, full_task):
        self.gt_fn_name = full_task.get('gt_fn_name', '')
        self.task_prompt = full_task.get('task_prompt', full_task['task'])
        logger.info(f'The task prompt is {truncate_str(self.task_prompt)}\n')

    def validate_code(self, imported_modules, functions, main_fns, fn_name):
        """
        Validates the generated code by checking imports, functions, and main function name.

        Args:
            imported_modules (list): List of imported modules.
            functions (list): List of functions in the code.
            main_fns (list): List of main functions in the code.
            fn_name (str): The name of the main function.

        Returns:
            bool: True if no parent function is found, False otherwise.
        """
        if self.check_imports:
            assert_modules_in_whitelist(imported_modules)
        if self.assert_fns:
            assert functions, 'Error: No functions found. please try again\n'

        gt_err_msg = (f"expected main function name {self.gt_fn_name} but got function name {fn_name}, try again. "
                      "Your response should declare helper functions first, then the main function last.\n")
        
        no_parent = False
        if self.gt_fn_name:
            if self.assert_fns:
                if not main_fns:
                    raise Exception("could not find any main functions (those without parent)")
                if self.gt_fn_name != fn_name:
                    raise Exception(gt_err_msg)
                no_parent = True
            else:
                no_parent = find_gt_fn(functions, self.gt_fn_name)
        return no_parent

    def parse_ai_code(self, message):
        """
        Parses AI-generated code from a message, extracting functions, imports, and dependencies.

        Parameters:
            message (AIMessage): The message containing the AI-generated code.

        Returns:
            dict: A dictionary containing parsed code information, including program code, program name, dependencies,
            and more.
        """
        code = extract_blocks(message.content, identifier='python|py')
        assert code, 'regex fails to extract Python code. check your formatting and try again\n'

        functions, import_statements, dependencies, imported_modules = extract_from_ast(code)

        main_fns = [fn for fn in functions if fn["no_parent"]]
        # main_fns can be blank if the fns are within a class Solution, so main_fns[-1] gives error
        fn_name = ''
        if main_fns:
            # TODO: if not self.assert_fns (generic code env), 
            # only checks for no_parent so last main fn might not be the gt fn!
            fn_name = main_fns[-1]['name']

        no_parent = self.validate_code(imported_modules, functions, main_fns, fn_name)

        if self.rebuild_code_from_ast:
            program_code = "\n".join(import_statements) + "\n\n".join(fn["body"] for fn in main_fns)
        else:
            program_code = code

        parsed_result = {
            "program_code": program_code,
            "program_name": fn_name,
            "dependencies": list(dependencies),
            "raw_msg": message.content,
            "no_parent": no_parent,
        }
        
        if self.verbose:
            for k, v in parsed_result.items():
                logger.info(f'{k}:\n {v}\n')
        return parsed_result


def find_gt_fn(functions, gt_fn_name):
    """
    Finds the ground truth function in the list of functions.

    Args:
        functions (list): List of functions in the code.
        gt_fn_name (str): The name of the ground truth function.

    Returns:
        bool: True if the ground truth function has no parent, False otherwise.

    Raises:
        Exception: If the ground truth function is not found.
    """
    for fn in functions:
        if fn['name'] == gt_fn_name:
            return fn["no_parent"]
    raise Exception(f"could not find any function with the required name {gt_fn_name}")
