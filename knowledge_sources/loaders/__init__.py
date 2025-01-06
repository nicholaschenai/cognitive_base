from .composite_memory_loader import CompositeMemoryLoader


def load_agent_memory(args, agent):
    """
    Loads documents into semantic memory
    """
    if args.load_db:
        # Can specify multiple sources in args
        memory_sources = args.memory_sources.split(',')
        kwargs = vars(args)
        memory_loader = CompositeMemoryLoader(
            sources=memory_sources,
            **kwargs
        )
        memory_loader.load(agent)
