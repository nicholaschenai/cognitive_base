"""
scaffold for vector database
"""
from ..base_db import BaseDB


class BaseVectorDB(BaseDB):
    def retrieve(self, query, top_k=5, **kwargs):
        """
        Queries the vector database.

        Args:
            top_k (int): Number of top results to return.

        Returns:
            list: List of top_k results.
        """
        raise NotImplementedError("Subclasses should implement this method")

    def update(self, entry, **kwargs):
        """
        Adds an entry to the vector database.
        """
        raise NotImplementedError("Subclasses should implement this method")