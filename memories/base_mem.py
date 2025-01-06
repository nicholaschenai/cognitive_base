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

from ..utils.formatting import tag_indent_format
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
        # args
        self.ckpt_dir = ckpt_dir
        self.kwargs = kwargs
        self.verbose = verbose
        self.debug_mode = debug_mode

        # params
        self.retrieval_top_k = retrieval_top_k

        self.dbs = {}
        self.register_vectordb(vectordb_name)

        # TODO: slowly deprecate self.vectordb and self.vectordb_name
        self.vectordb = self.dbs[vectordb_name].db
        self.vectordb_name = vectordb_name

        # Retrieval Actions (to working mem / decision procedure)
        self.retrieval_methods = {
            'vector': VectorRetrieval(
                self.dbs[vectordb_name], 
                retrieval_top_k=self.retrieval_top_k
            )
        }
        # Learning Actions (from working mem)
        self.update_methods = {
            'vector': VectorUpdate(self.dbs[vectordb_name])
        }

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

    def register_vectordb(self, vectordb_name):
        """
        Registers a vector database with the memory.
        """
        vectordb = ChromaVectorDB(
            vectordb_name=vectordb_name, 
            ckpt_dir=self.ckpt_dir,
            verbose=self.verbose,
            debug_mode=self.debug_mode,
            **self.kwargs
        )
        self.register_db(vectordb_name, vectordb)

    def register_vectordb_with_methods(self, vectordb_name):
        """
        Registers a vector database with the memory and sets up corresponding retrieval and update methods.
        
        Args:
            vectordb_name (str): The name of the vector database to register.
        """
        self.register_vectordb(vectordb_name)
        self.retrieval_methods.update({
            vectordb_name: VectorRetrieval(
                self.dbs[vectordb_name], 
                retrieval_top_k=self.retrieval_top_k
            )
        })
        self.update_methods.update({
            vectordb_name: VectorUpdate(self.dbs[vectordb_name])
        })

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
    
    def retrieve(self, query, **kwargs):
        """
        default retrieves entries from the vector database based on similarity to a text.

        Args:
            query: The query text.

        Returns:
            list: A list of documents retrieved from the database.
        """
        return self.retrieve_by_ebd(query, **kwargs)

    def _retrieve_and_format(self, query, retrieval_method, tag, transform_fn=None, **kwargs):
        """
        Generic method to retrieve and format documents from a specific database.

        Args:
            query: The query text.
            retrieval_method: Name of retrieval method to use.
            tag: Tag to use in formatting the results.
            transform_fn: Optional function to transform doc_obj into content string.
                        If None, uses doc_obj.page_content
            **kwargs: Additional arguments passed to the retrieval method.

        Returns:
            Union[List[str], List[Tuple[str, float]]]: Either:
                - A list of formatted strings with proper indentation
                - A list of tuples (formatted_string, score) if scores were requested
        """
        docs = self.retrieval_methods[retrieval_method].retrieve(query, **kwargs)
        
        if transform_fn is None:
            transform_fn = lambda doc_obj: doc_obj.page_content
        # Format each document's content
        formatted_docs = []
        for doc in docs:
            # Extract document object and optional score
            doc_obj = doc[0] if isinstance(doc, tuple) else doc
            # Transform content using provided function or default to page_content
            content = tag_indent_format(tag, [transform_fn(doc_obj)])
            
            # Maintain original return type (with or without score)
            if isinstance(doc, tuple):
                formatted_docs.append((content, doc[1]))
            else:
                formatted_docs.append(content)
        
        return formatted_docs
    
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

    def update(self, entry, **kwargs):
        """
        default update Stores an embedding in the vector database.

        Args:
            entry: The embedding to store.
            metadata (dict): Optional metadata associated with the embedding.
            db: The database where the embedding should be stored. If None, the default database is used.
        """
        self.update_ebd(entry, **kwargs)
