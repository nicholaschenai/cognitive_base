from .base_retrieval import BaseRetrieval


class VectorRetrieval(BaseRetrieval):
    def retrieve(self, query, k_new=0, verbose=False, debug_mode=False, **kwargs):
        k = k_new if k_new else self.retrieval_top_k
        return self.db.retrieve(query, k=k, **kwargs)
