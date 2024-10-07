"""
This file constructs the agent in layers for modularity

we implement the working memory as attributes
Learning and retrieval actions are implemented as methods within their respective memories
"""


from . import get_cls

from agent_expt_suite.eval_setup.log import VerboseHandler


def construct_agent(args, agent_config):
    """
    Constructs an agent by dynamically loading and initializing its components.

    This function reads the agent configuration based on the agent type specified
    in the arguments. It then dynamically loads and initializes the components
    specified in the configuration, including memories and reasoning components,
    and finally constructs the agent with these components.

    Parameters:
    - args (Namespace): The command line arguments or configuration namespace
      containing settings for the agent construction, including the agent type
      and any other necessary parameters.

    Returns:
    - agent: An instance of the agent class, constructed with the specified
      components and settings.
    """

    components = {}
    handler = VerboseHandler(verbose=args.verbose)
    # TODO: as complexity increases, consider config builder to get right args for components
    for component_config in agent_config['memories']:
        component_cls = get_cls(component_config)
        components[component_config['name']] = component_cls(**vars(args))
    for component_config in agent_config['reasoning']:
        component_cls = get_cls(component_config)
        components[component_config['name']] = component_cls(callbacks=[handler], **vars(args))
    agent_cls = get_cls(agent_config.get('agents', agent_config['decisions']))
    agent = agent_cls(args, components, handler)
    return agent
