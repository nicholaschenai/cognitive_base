# Cognitive Base Library

`cognitive_base` is a collection of implementations of processes inspired by cognitive science and cognitive architectures. 

## Features
- **Memory Elements**: Components for storing and retrieving data.
- **Reasoning**: Tools for logical reasoning and decision-making.
- **Learning**: Mechanisms for learning from data and experiences.
- **Retrieval**: Efficient data retrieval methods.
- **Planning**: Tools for creating and executing plans.

## Getting Started

### Installation

See examples

## Documentation

- [Memory Elements](docs/memories.md)
- [Reasoning](docs/reasoning.md)
- [Learning](docs/learning.md)
- [Retrieval](docs/retrieval.md)
- [Planning](docs/planning.md)
- [Decision Procedures](docs/decision_procedures.md)

## Examples
See this [repo](https://github.com/nicholaschenai/coala_coder_demo) for a prototype of a CoALA coding agent.

## Disclaimer

This is currently a hobby project and is not intended for any serious use.

## References
The library is inspired by various cognitive architectures, including:
- CoALA (Sumers et. al. 2023)
- Soar (various iterations, see [references](https://soar.eecs.umich.edu/home/About/))

## Roadmap
We aim to implement more of the features from the above references over time.

We might also implement modules from "Building Machines That Learn and Think Like People"
(Lake et. al. 2016) such as
- Intuitive Physics module
- Intuitive Psychology module
- Causal world models
- Compositionality

The skills library in the Voyager coder example has elements of causal models 
(LLM expected to choose skills in order) and compositionality 
(LLM expected to compose new skills from existing skills).

