"""
Retrieval classes so that we can adjust things like params, transforms, etc while still using base retrieval method
"""


class BaseRetrieval:
    def __init__(self, db, transform=None, verbose=False, debug_mode=False, retrieval_top_k=5):
        self.db = db
        self.transform = transform
        self.verbose = verbose
        self.debug_mode = debug_mode
        self.retrieval_top_k = retrieval_top_k

    def retrieve(self, query, **kwargs):
        raise NotImplementedError("Subclasses should implement this method")
