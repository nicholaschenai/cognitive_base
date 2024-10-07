"""
Base class for memories.

This class provides foundational methods for memory retrieval and learning
operations

Attributes:
    retrieval_top_k (int): The number of top results to retrieve in retrieval operations.
    ckpt_dir (str): The directory where checkpoints are stored.
    vectordb (object): The vector database object used for memory operations.
    vectordb_name (str): The name of the vector database.

Args:
    retrieval_top_k (int): The number of top results to retrieve in retrieval operations. Defaults to 5.
    ckpt_dir (str): The directory where checkpoints are stored. Defaults to "ckpt".
    vectordb_name (str): The name of the vector database to use. Defaults to "base".
    resume (bool): Whether to resume from an existing checkpoint. Defaults to True.
    **kwargs: Additional keyword arguments passed to the database factory.
"""
import logging

from ..retrieval.vector_retrieval import VectorRetrieval
from ..learning.vector_update import VectorUpdate

# from ..utils.database.database_factory import get_database
from ..utils.database.database_wrapper import DatabaseWrapper
from ..utils.database.vector_db.chroma_vector_db import ChromaVectorDB


logger = logging.getLogger("logger")


class BaseMem:
    def __init__(
            self,
            retrieval_top_k=5,
            ckpt_dir="ckpt",
            vectordb_name="base",
            resume=True,
            verbose=False,
            debug_mode=False,
            **kwargs,
    ):

        self.retrieval_top_k = retrieval_top_k
        self.ckpt_dir = ckpt_dir

        vectordb = ChromaVectorDB(vectordb_name=vectordb_name, ckpt_dir=ckpt_dir, **kwargs)
        # self.vectordb = get_database(vectordb_name=vectordb_name, ckpt_dir=ckpt_dir, **kwargs)
        # TODO: slowly deprecate self.vectordb and self.vectordb_name
        self.vectordb = vectordb.db
        self.vectordb_name = vectordb_name

        self.dbs = {}
        self.register_db(vectordb_name, vectordb)
        # Retrieval Actions (to working mem / decision procedure)
        self.retrieval_methods = {
            'vector': VectorRetrieval(self.dbs[vectordb_name])
        }
        # Learning Actions (from working mem)
        self.update_methods = {
            'vector': VectorUpdate(self.dbs[vectordb_name])
        }

        self.verbose = verbose
        self.debug_mode = debug_mode

    """
    helper fns
    """
    @staticmethod
    def print_one_doc_count(db_name: str, db: DatabaseWrapper):
        """
        Logs the document count of a given database.
        """
        # Note: must define wrapper around db to ensure consistent interface for db.count()
        logger.info(f'DB {db_name} doc count: {db.count()}\n')

    def register_db(self, db_name, db):
        """
        Registers a database with the memory.
        """
        self.dbs.update({db_name: db})
        self.print_one_doc_count(db_name, db)

    def print_doc_count(self):
        """ 
        Logs the document count of all registered databases.
        """
        for db_name, db in self.dbs.items():
            self.print_one_doc_count(db_name, db)

    """
    Retrieval Actions (to working mem / decision procedure)
    """
    # unless kwargs are v custom
    def retrieve_by_ebd(self, query, **kwargs):
        """
        Retrieves entries from the vector database based on similarity to a text.

        Args:
            query: The query text.

        Returns:
            list: A list of documents retrieved from the database.
        """
        return self.retrieval_methods['vector'].retrieve(query, **kwargs)

    """
    Learning Actions (from working mem)
    """
    def update_ebd(self, entry, **kwargs):
        """
        Stores an embedding in the vector database.

        Args:
            entry: The embedding to store.
            metadata (dict): Optional metadata associated with the embedding.
            db: The database where the embedding should be stored. If None, the default database is used.
        """
        self.update_methods['vector'].update(entry, **kwargs)
