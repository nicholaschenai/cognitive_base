from .base_update import BaseUpdate


class VectorUpdate(BaseUpdate):
    def __init__(self, db, verbose=False, debug_mode=False):
        super().__init__(db)
        self.verbose = verbose
        self.debug_mode = debug_mode

    def update(self, entry, metadata=None, doc_id=None, **kwargs):
        """
        Stores an embedding in the vector database.

        Args:
            entry: The embedding to store.
            metadata (dict): Optional metadata associated with the embedding.
        """
        ids = self.db.update(entry, metadata, doc_id, **kwargs)
        # if self.verbose or self.debug_mode:
        #     logger.info(f"Updated entry: {truncate_str(entry)}, Metadata: {metadata}")
        return ids
