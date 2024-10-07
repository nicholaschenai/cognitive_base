"""
This module provides a factory function for creating database instances with a common interface.
It currently supports creating instances of the Chroma vector database,
with plans to support additional databases such as PGVector in the future.

Functions:
    get_database(vectordb_name: str, ckpt_dir: str, db_type: str, collection_name: str, persist_directory: str, **kwargs) -> DatabaseWrapper:
        Creates and returns a database instance wrapped in a DatabaseWrapper for a common interface. The type of database created depends on the `db_type` parameter.

# TODO:
    - Integrate support for PGVector database, which requires a connection string and additional setup for use with PostgreSQL.
    - Explore and add support for other database types as needed, ensuring they can be integrated with the common interface provided by DatabaseWrapper.

"""

from langchain.vectorstores import Chroma
# from langchain_postgres.vectorstores import PGVector

from .database_wrapper import DatabaseWrapper
from ..llm import get_embedding_fn
from ...utils import f_mkdir


def get_database(
        vectordb_name: str = 'base',
        ckpt_dir: str = 'ckpt',
        db_type: str = 'chroma',
        collection_name: str = '',
        persist_directory: str = '',
        **kwargs
):
    """
    Creates and returns a langchain vector database instance wrapped in a DatabaseWrapper for a common interface.

    Args:
        vectordb_name (str): The name of the vector database. Defaults to 'base'.
        ckpt_dir (str): The directory where checkpoints are stored. Defaults to 'ckpt'.
        db_type (str): The type of database to create. Currently supports 'chroma'. Defaults to 'chroma'.
        collection_name (str): The name of the collection within the database. If not provided, it defaults to a name based on `vectordb_name`.
        persist_directory (str): The directory where the database should persist its data. If not provided, it defaults to a directory based on `ckpt_dir` and `vectordb_name`.
        **kwargs: Additional keyword arguments that may be required by specific database types.

    Returns:
        DatabaseWrapper: A wrapper around the created langchain vectordb instance, providing a common interface.

    Raises:
        ValueError: If an unsupported `db_type` is specified.

    TODO:
        - Implement support for the 'pgvector' database type, requiring additional parameters such as a connection string.
    """
    embedding_fn = get_embedding_fn()
    if db_type == 'chroma':
        if not persist_directory:
            persist_directory = f"{ckpt_dir}/{vectordb_name}/vectordb"
        if not collection_name:
            collection_name = f"{vectordb_name}_vectordb"
        f_mkdir(persist_directory)
        db = Chroma(
            collection_name=collection_name,
            embedding_function=embedding_fn,
            persist_directory=persist_directory,
        )
        return DatabaseWrapper(db, lambda x: x._collection.count())
    # TODO: pgvector. requires connection string
    # elif db_type == 'pgvector':
    #     return PGVector(
    #         embeddings=embedding_fn,
    #         collection_name=collection_name,
    #         connection=connection,
    #         use_jsonb=True,
    #     )
    else:
        raise ValueError(f"Unsupported database type: {db_type}")
