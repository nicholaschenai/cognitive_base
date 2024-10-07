"""
This module defines the base class for language model (LM) based reasoning within the cognitive architecture. 
It encapsulates the interaction with a language model to perform reasoning tasks, including generating responses 
based on templates and parsing the output.

The class is designed to be extended for specific reasoning tasks that require interaction with a language model.
"""
import asyncio

from langchain.output_parsers import PydanticOutputParser
from langchain.schema import HumanMessage, SystemMessage
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate

from pprint import pp

from ..utils import parse_retry_loop, pydantic_parse_fn, async_parse_retry_loop
from ..utils.llm import construct_chat_model
from ..knowledge_sources.parsers import extract_blocks


class BaseLMReasoning:
    """
    A base class for reasoning using a language model (LM).

    This class provides foundational functionalities to interact with a language model for reasoning purposes. 
    It includes initializing the model, extracting code blocks from text, and a method for reasoning with the model 
    using system and human message templates.

    Attributes:
        llm: The language model instance configured for chat.
        name (str): The name of the reasoning instance, useful for debugging.
        debug_mode (bool): Flag to enable debug mode for additional logging.
        verbose (bool): Flag to enable verbose output.

    Methods:
        post_init(): A hook for additional initialization after the main initialization.
        extract_blocks(text, identifier=''): Static method to extract code blocks from text.
        lm_reason(sys_template, human_template, ...): Performs reasoning by calling the LM with provided templates and parsing the output.
    """
    def __init__(
            self,
            model_name="gpt-3.5-turbo",
            temperature=0,
            request_timeout=120,
            verbose=True,
            callbacks=None,
            debug_mode=False,
            name='base_reasoning',
            parallel_api=False,
            **kwargs,
    ):
        """
        Initializes the BaseLMReasoning class with a specified language model and configuration settings.

        Parameters:
            model_name (str): The name of the language model to use.
            temperature (float): The sampling temperature for generating responses from the language model.
            request_timeout (int): The timeout in seconds for language model requests.
            verbose (bool): Enables verbose output if True.
            callbacks (list): A list of callback functions to be called on the language model's response.
            debug_mode (bool): Enables debug mode if True, providing additional output for debugging.
            name (str): The name of the reasoning instance.
            **kwargs: Additional keyword arguments for future extensions.
        """
        # Models
        self.llm = construct_chat_model(
            model_name=model_name,
            temperature=temperature,
            request_timeout=request_timeout,
            verbose=verbose,
            callbacks=callbacks,
        )

        # misc settings
        self.name = name
        self.debug_mode = debug_mode
        self.verbose = verbose
        self.parallel_api = parallel_api

        self.post_init()

    """
    helper fns
    """

    def post_init(self):
        """
        A hook for additional initialization actions after the main initialization process.

        This method is called at the end of the __init__ method. It's intended for subclasses to add any additional 
        initialization steps without needing to override the __init__ method.
        """
        if self.verbose:
            print(f'{self.name}\n')
            pp(self.llm.dict())

    # TODO: eventually deprecate this when refactor legacy code
    @staticmethod
    def extract_blocks(text, identifier=''):
        return extract_blocks(text, identifier)

    @staticmethod
    def construct_messages(sys_template, human_template, sys_vars=None, human_vars=None, parser=None):
        def create_message_thread(single_sys_vars, single_human_vars):
            if single_sys_vars or ('{format_instructions}' in sys_template):
                sys_prompt_template = SystemMessagePromptTemplate.from_template(sys_template)
                if single_sys_vars is None:
                    single_sys_vars = {}
                if ('{format_instructions}' in sys_template) and ('format_instructions' not in single_sys_vars):
                    assert parser is not None, "format_instructions expected but not found in vars or parser"
                    single_sys_vars['format_instructions'] = parser.get_format_instructions()
                sys_message = sys_prompt_template.format(**single_sys_vars)
            else:
                sys_message = SystemMessage(content=sys_template)

            if single_human_vars:
                human_prompt_template = HumanMessagePromptTemplate.from_template(human_template)
                human_message = human_prompt_template.format(**single_human_vars)
            else:
                human_message = HumanMessage(content=human_template)

            return [sys_message, human_message]

        if isinstance(sys_vars, list) or isinstance(human_vars, list):
            if isinstance(sys_vars, list) and isinstance(human_vars, list):
                assert len(sys_vars) == len(human_vars), "sys_vars and human_vars lists must have the same length"
                message_threads = [create_message_thread(sv, hv) for sv, hv in zip(sys_vars, human_vars)]
            elif isinstance(sys_vars, list):
                message_threads = [create_message_thread(sv, human_vars) for sv in sys_vars]
            else:
                message_threads = [create_message_thread(sys_vars, hv) for hv in human_vars]
        else:
            message_threads = [create_message_thread(sys_vars, human_vars)]

        return message_threads

    """
    Reasoning Actions (from and to working mem)
    """

    # TODO: allow LCEL (so can parallel chain)
    def lm_reason(
            self,
            sys_template='',
            human_template='',
            parse_fn=None,
            llm=None,
            sys_vars=None,
            human_vars=None,
            parse_tries=3,
            fallback=None,
            parser=None,
            return_messages=False,
            return_json=False,
            pydantic_model=None,
            messages=None,
            structured=False,
            messages_list=None,
    ):
        """
        Performs reasoning by interacting with the language model using provided templates.

        This method sends messages to the language model based on the provided system and human templates, 
        parses the response, and optionally retries if parsing fails.

        Parameters:
            sys_template (str): The str template for system messages.
            human_template (str): The str template for human messages.
            parse_fn (callable): The function to parse the language model's response.
            llm: The language model instance to use. If None, uses the instance initialized during class instantiation.
            parse_tries (int): The number of retries for parsing the language model's response.
            fallback: A fallback value to return in case parsing fails after all retries.
            parser: An optional parser to use instead of the default.
            return_messages (bool): If True, returns the messages sent to and received from the language model.
            return_json (bool): If True, returns the response in JSON format.
            pydantic_model: An optional Pydantic model to validate the parsed response.

        Returns:
            The parsed response from the language model, optionally validated by a Pydantic model.
        """
        # structured_lm_reason stuff here
        if structured:
            if parse_fn is None:
                parse_fn = pydantic_parse_fn
            parser = PydanticOutputParser(pydantic_object=pydantic_model)

        if parse_fn is None:
            parse_fn = return_content

        if messages is None and messages_list is None:
            message_threads = self.construct_messages(sys_template, human_template, sys_vars, human_vars, parser)
            if len(message_threads) == 1:
                messages = message_threads[0]
            else:
                messages_list = message_threads

        return_type = 'ai_message'
        if llm is None:
            llm = self.llm
        if pydantic_model is not None:
            llm = llm.with_structured_output(pydantic_model, include_raw=True)
            return_type = 'pydantic_object'
        elif return_json:
            llm = llm.bind(response_format={"type": "json_object"})

        # TODO: langgraph for retries
        # TODO: handle case where forgot to turn on parallel_api
        if self.parallel_api and messages_list is not None:
            async def async_eval_loop():
                tasks = []
                for msgs in messages_list:
                    tasks.append(
                        async_parse_retry_loop(
                            msgs,
                            parse_fn,
                            llm,
                            parse_tries=parse_tries,
                            fallback=fallback,
                            debug_mode=self.debug_mode,
                            parser=parser,
                            return_messages=return_messages,
                            verbose=self.verbose,
                            return_type=return_type,
                            name=self.name,
                        )
                    )
                results = await asyncio.gather(*tasks)
                return results
            out = asyncio.run(async_eval_loop())
        else:
            out = parse_retry_loop(
                messages,
                parse_fn,
                llm,
                parse_tries=parse_tries,
                fallback=fallback,
                debug_mode=self.debug_mode,
                parser=parser,
                return_messages=return_messages,
                verbose=self.verbose,
                return_type=return_type,
                name=self.name,
            )

        return out

    # TODO: eventually deprecate below for having lm_reason with structured=True
    def structured_lm_reason(
            self,
            sys_template='',
            human_template='',
            parse_fn=pydantic_parse_fn,
            llm=None,
            sys_vars=None,
            human_vars=None,
            fallback=None,
            pydantic_model=None,
            messages=None,
            **kwargs,
    ):
        """
        LM reason with structured output.

        Args:
            sys_template (str): The system prompt template
            human_template (str): The human prompt template
            parse_fn (callable): The function used to parse the language model output. Defaults to pydantic_parse_fn.
            llm (LanguageModel, optional): The language model to use for reasoning. Defaults to None.
            sys_vars (dict, optional): Variables related to the system template. Defaults to None.
            human_vars (dict, optional): Variables related to the human template. Defaults to None.
            fallback (dict, optional): A fallback mechanism in case of failure. Defaults to {'output': []}.
            pydantic_model (PydanticModel): The Pydantic model to parse the structured output into.
            messages (list): A list of messages to be sent to the language model, to override sys and human templates
            **kwargs: Arbitrary keyword arguments passed to the language model reasoning method.

        Returns:
            dict: The output from the language model reasoning, parsed into the specified Pydantic model converted to json.
        """
        parser = PydanticOutputParser(pydantic_object=pydantic_model)

        out = self.lm_reason(
            sys_template=sys_template,
            human_template=human_template,
            parse_fn=parse_fn,
            llm=llm,
            sys_vars=sys_vars,
            human_vars=human_vars,
            fallback=fallback,
            parser=parser,
            pydantic_model=pydantic_model,
            messages=messages,
            structured=True,
            **kwargs,
        )

        return out


def return_content(x):
    return x.content
