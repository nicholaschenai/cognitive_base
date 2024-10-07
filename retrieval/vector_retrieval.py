from .base_retrieval import BaseRetrieval


class VectorRetrieval(BaseRetrieval):
    def retrieve(self, query, k=5, **kwargs):
        return self.db.retrieve(query, k=k, **kwargs)
