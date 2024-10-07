import logging
from langchain.vectorstores import Chroma

from .base_vector_db import BaseVectorDB

from ...llm import get_embedding_fn
from ....utils import f_mkdir

logger = logging.getLogger("logger")


class ChromaVectorDB(BaseVectorDB):
    # TODO: future: all dbs should have config rather than individual args
    def __init__(
            self,
            vectordb_name: str = 'base',
            ckpt_dir: str = 'ckpt',
            collection_name: str = '',
            persist_directory: str = '',
            **kwargs
    ):
        embedding_fn = get_embedding_fn()
        if not persist_directory:
            persist_directory = f"{ckpt_dir}/{vectordb_name}/vectordb"
        if not collection_name:
            collection_name = f"{vectordb_name}_vectordb"
        f_mkdir(persist_directory)
        self.db = Chroma(
            collection_name=collection_name,
            embedding_function=embedding_fn,
            persist_directory=persist_directory,
        )
        self.db_name = vectordb_name

    def count(self):
        return self.db._collection.count()

    def retrieve(self, query, k=5, **kwargs):
        """
        Retrieves entries from the vector database based on similarity to a query text.

        Args:
            query: The query embedding.

        Returns:
            list: A list of documents retrieved from the database.
        """

        k = min(self.db.count(), k)
        docs = []
        if k:
            logger.info(f"\033[33m Retrieving {k} entries for db: {self.db_name} \n \033[0m")
            docs = self.db.similarity_search(query, k=k)
        return docs

    def update(self, entry, metadata=None, **kwargs):
        """
        Stores an embedding in the vector database.

        Args:
            entry: The embedding to store.
            metadata (dict): Optional metadata associated with the embedding.
        """
        self.db.add_texts(texts=[entry], metadatas=[metadata])
        self.db.persist()
