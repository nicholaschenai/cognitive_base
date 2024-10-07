# Simple Usage of the Cognitive Base Library to define an agent

This document provides a bare-bones example of how to use the Cognitive Base Library to define a reasoning module and a memory module, and how to integrate them into an agent.

## Reasoning Module

A reasoning module in the Cognitive Base Library inherits from the [`BaseLMReasoning`](../reasoning/base_lm_reasoning.py) class. See the [reasoning docs](reasoning.md) for more details. Below is an example of a simple reasoning module:

```python
from cognitive_base.reasoning.base_lm_reasoning import BaseLMReasoning

class SimpleReasoningModule(BaseLMReasoning):
    def __init__(self, name='simple_reasoning', **kwargs):
        super().__init__(name=name, **kwargs)
    
    def simple_reason(self, messages):
        sys_template = ...
        human_template = ...
        out = self.lm_reason(
            sys_template=sys_template,
            human_template=human_template,
        )
        return out
```

This module leverages the language model to generate responses

## Memory Module

A memory module in the Cognitive Base Library inherits from the [`BaseMem`](../memories/base_mem.py) class which already comes with vector retrieve and update. See the [memory docs](memories.md) for more details. Below is an example of a simple procedural memory module:

```python
from cognitive_base.memories.base_mem import BaseMem

class SimpleProceduralMem(BaseMem):
    def __init__(self, ckpt_dir="ckpt", **kwargs):
        super().__init__(ckpt_dir=ckpt_dir, **kwargs)

    def retrieve_rule(self, cue):
        return self.retrieval_methods['vector'].retrieve(cue)
```

## Agent

An agent integrates reasoning and memory modules and defines decision procedures as methods. Below is an example of a simple agent:

```python
class SimpleAgent:
    def __init__(self, model_name, ckpt_dir):
        self.reasoning_module = SimpleReasoningModule(model_name=model_name)
        self.procedural_mem = SimpleProceduralMem(ckpt_dir=ckpt_dir)
    
    def make_decision(self, messages):
        # Use the reasoning module to generate a response
        response = self.reasoning_module.simple_reason(messages)
        
        # Use the memory module to retrieve a rule
        rule = self.procedural_mem.retrieve_rule(response)
        
```

In this example, the `make_decision` method uses the reasoning module to generate a response and the memory module to retrieve a rule based on that response.
