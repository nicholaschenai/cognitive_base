import logging
import json

from langchain.vectorstores import Chroma

from pprint import pp

from .base_vector_db import BaseVectorDB

from ...llm import get_embedding_fn
from ....utils import f_mkdir
from ....utils.formatting import truncate_str

logger = logging.getLogger("logger")


class ChromaVectorDB(BaseVectorDB):
    # TODO: future: all dbs should have config rather than individual args
    def __init__(
        self,
        vectordb_name: str = 'base',
        ckpt_dir: str = 'ckpt',
        collection_name: str = '',
        persist_directory: str = '',
        verbose: bool = False,
        debug_mode: bool = False,
        **kwargs
    ):
        self.verbose = verbose
        self.debug_mode = debug_mode

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

    def retrieve(self, query, k=5, with_scores=False, **kwargs):
        """
        Retrieves entries from the vector database based on similarity to a query text.

        Args:
            query: The query embedding.

        Returns:
            list: A list of documents retrieved from the database.
        """

        k = min(self.count(), k)
        docs = []
        if k:
            logger.info(f"\033[33m Retrieving {k} entries for db: {self.db_name} \n \033[0m")
            if with_scores:
                docs = self.db.similarity_search_with_score(query, k=k, **kwargs)
                for doc, score in docs:
                    logger.info(f"Retrieved (score={score:.4f}):\n{truncate_str(doc.page_content)}\n\n")
            else:
                docs = self.db.similarity_search(query, k=k, **kwargs)
                for doc in docs:
                    logger.info(f"Retrieved doc:\n{truncate_str(doc.page_content)}\n\n")
        return docs

    def update(self, entry, metadata=None, doc_id=None, **kwargs):
        """
        Stores an embedding in the vector database.

        Args:
            entry: The embedding to store.
            metadata (dict): Optional metadata associated with the embedding.
        """
        # Note: this is upsert
        # Note: if not specified, ids will be uuid4 which is not deterministic
        ids = [doc_id] if doc_id else None
        ids = self.db.add_texts(texts=[entry], metadatas=[metadata], ids=ids, **kwargs)
        self.db.persist()
        logger.info(f"Updated entry: {truncate_str(entry)},\n")
        logger.info(f"Metadata: {truncate_str(json.dumps(metadata, indent=4))}\n")
        return ids
