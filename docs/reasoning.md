# Reasoning

## Overview
The `BaseLMReasoning` class contains a `lm_reason` method that calls an LLM for reasoning. Here are some typical patterns

## Usage

### Example 1: Using Prompt Templates and Variables, and Defined Parse Functions
```python
from cognitive_base.reasoning.base_lm_reasoning import BaseLMReasoning

# Instantiate the BaseLMReasoning class
reasoning_instance = BaseLMReasoning()

# Define the input variables
sys_template = """
You are a coding assistant, specializing in writing {programming_language} functions.
"""
human_template = """
Please write a function named {function_name} that {function_description}.
"""
sys_vars = {'programming_language': 'Python'}
human_vars = {'function_name': 'add_numbers', 'function_description': 'takes two arguments and returns their sum'}

def simple_parse_fn(message):
    code = message.content
    assert "def add_numbers" in code, "Function definition not found"
    return {"parsed_code": code.strip()}

# Perform reasoning
generated_code = reasoning_instance.lm_reason(
    sys_template=sys_template,
    human_template=human_template,
    sys_vars=sys_vars,
    human_vars=human_vars,
    parse_fn=simple_parse_fn,
)
```

#### Explanation

##### Input Variables:

- `sys_template` (str): A system message template instructing the language model about its overall goal and role.
- `human_template` (str): A human message template providing more details about the specific task to be generated.
- `sys_vars` (dict) and `human_vars` (dict): Dictionaries containing variables to be formatted into the templates.
- `parse_fn` (callable): An optional function that is applied directly to the raw result, typically used for parsing, assertions, and transformations.

### Example 2: Using Messages List and Structured Outputs with Pydantic Validation

Let's use LM reasoning to check if a code is written correctly from the code's execution outputs.

```python
from pydantic import BaseModel, Field

# Define a Pydantic model for structured output
class CodeCritique(BaseModel):
    reasoning: str = Field(description="Reason out step by step if my code succeeded or failed")
    success: bool = Field(description="The answer to whether my code succeeded or failed")
    critique: str = Field(description="Critique to help me improve my code")

# Messages which were constructed elsewhere, showing the code's execution outputs
messages = ...

# Perform reasoning
result = reasoning_instance.lm_reason(
    messages=messages,
    structured=True,
    pydantic_model=CodeCritique,
    fallback={'success': False, 'critique': "", 'reasoning': ""},
)
```

#### Explanation
- `messages` (list): A list of messages (System, Human, AI) that have been constructed elsewhere. 

- `structured` (bool): This flag indicates calling a model's structured output function, validated using a Pydantic model.

- `pydantic_model` (Pydantic model): Validation via Pydantic model

- `fallback` (dict): A fallback dictionary that provides default values in case the reasoning process fails.

- `parse_fn` is automatically defined when you use structured outputs with Pydantic validation.

### Important Notes
- Either `messages` or the combination of `sys_template`, `human_template`, `sys_vars`, and `human_vars` must be used, not both at the same time. The vars can be skipped if the template does not contain anything to format
- The rest of the parameters are optional.