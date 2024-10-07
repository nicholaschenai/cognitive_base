"""
Retrieval classes so that we can adjust things like params, transforms, etc while still using base retrieval method
"""


class BaseRetrieval:
    def __init__(self, db, transform=None):
        self.db = db
        self.transform = transform

    def retrieve(self, query, **kwargs):
        raise NotImplementedError("Subclasses should implement this method")
