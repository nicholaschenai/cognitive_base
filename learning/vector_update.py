from .base_update import BaseUpdate


class VectorUpdate(BaseUpdate):
    def update(self, entry, metadata=None, **kwargs):
        """
        Stores an embedding in the vector database.

        Args:
            entry: The embedding to store.
            metadata (dict): Optional metadata associated with the embedding.
        """
        self.db.update(entry, metadata)
