from langchain.output_parsers import PydanticOutputParser

from .base_lm_reasoning import BaseLMReasoning
from .pydantic_models import BaseRule

from ..utils import pydantic_parse_fn


class RuleExtraction(BaseLMReasoning):
    """
    A class for extracting rules from templates using language models.

    This class extends BaseLMReasoning to provide functionality for extracting
    rules based on a language model
    
    """
    def __init__(
            self,
            name='rule_extraction',
            **kwargs,
    ):
        super().__init__(
            name=name,
            **kwargs,
        )

    """
    helper fns
    """

    """
    Reasoning Actions (from and to working mem)
    """

    def extract_rules(
            self,
            sys_template,
            human_template,
            parse_fn=pydantic_parse_fn,
            llm=None,
            sys_vars=None,
            human_vars=None,
            fallback=None,
            pydantic_model=BaseRule,
            **kwargs,
    ):
        """
        Extracts rules using a language model.

        Args:
            sys_template (str): The system prompt template for rule extraction.
            human_template (str): The human prompt template for rule extraction.
            parse_fn (callable): The function used to parse the language model output. Defaults to pydantic_parse_fn.
            llm (LanguageModel, optional): The language model to use for reasoning. Defaults to None.
            sys_vars (dict, optional): Variables related to the system template. Defaults to None.
            human_vars (dict, optional): Variables related to the human template. Defaults to None.
            fallback (dict, optional): A fallback mechanism in case of failure. Defaults to {'output': []}.
            pydantic_model (PydanticModel): The Pydantic model to parse the extracted rules into.
            **kwargs: Arbitrary keyword arguments passed to the language model reasoning method.

        Returns:
            dict: The output from the language model reasoning, parsed into the specified Pydantic model converted to json.
        """
        parser = PydanticOutputParser(pydantic_object=pydantic_model)
        if fallback is None:
            fallback = {'output': []}

        out = self.lm_reason(
            sys_template,
            human_template,
            parse_fn=parse_fn,
            llm=llm,
            sys_vars=sys_vars,
            human_vars=human_vars,
            fallback=fallback,
            parser=parser,
            pydantic_model=pydantic_model,
            **kwargs,
        )

        return out
