from .comp_prog import create_comp_prog_loader


class CompositeMemoryLoader:
    """Handles loading from multiple memory sources"""
    def __init__(self, sources: list[str], **kwargs):
        self.loaders = []
        for source_name in sources:
            if source_name == "comp_prog":
                self.loaders.append(create_comp_prog_loader(**kwargs))
            # Add more sources as needed
    
    def load(self, agent):
        for loader in self.loaders:
            loader.load_into_agent(agent)

# TODO: future: parallel loading
# class CompositeMemoryLoader:
#     def __init__(self, sources: list[str], load_strategy: str = "sequential", **kwargs):
#         self.sources = []
#         self.load_strategy = load_strategy
#         # ... source initialization ...
    
#     def load(self, agent):
#         if self.load_strategy == "sequential":
#             self._load_sequential(agent)
#         elif self.load_strategy == "parallel":
#             self._load_parallel(agent)
    
#     def _load_sequential(self, agent):
#         for source in self.sources:
#             source.load_into_agent(agent)
    
#     def _load_parallel(self, agent):
#         # Could implement parallel loading if needed
#         pass
