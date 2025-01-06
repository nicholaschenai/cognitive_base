"""
Base class for memories. standard methods for retrieval and learning (read, write)
Legacy class as BaseMem is the main one
"""
import logging

from ...memories.base_mem import BaseMem

from ...utils import dump_json, load_json
from ...utils.code_parse import get_fn_name

logger = logging.getLogger("logger")


class BaseVectorMem(BaseMem):
    """
    A memory module for vector-based retrieval and storage of documents or code snippets.
    
    This class extends `BaseMem` to provide functionalities specific to handling vectorized representations
    of documents or code, facilitating efficient retrieval and storage operations based on semantic similarity.
    
    Attributes:
        retrieval_top_k (int): The number of top-k results to retrieve in a similarity search.
        ckpt_dir (str): The directory path for saving and loading checkpoints.
        vectordb_name (str): The name of the vector database.
        fn_str_map (dict): A mapping from function names to their string representations.
    
    Methods:
        log_content(docs): Logs the content of documents.
        log_name(docs): Logs the names of documents.
        retrieve(query, db=None, db_name="", metadata_filter=None, k_new=0, log_fn=None, new_only=False,
                 id_key='doc_hash', log_docs=True): Retrieves documents based on a query.
        retrieve_code(query, metadata_filter=None, k_new=0, new_only=False, return_formatted=False): Retrieves code snippets based on a query.
        add_code(raw_data, mapping, description, prevent_duplicates=False): Adds a new code snippet to the database.
    """
    def __init__(
        self,
        retrieval_top_k=5,
        ckpt_dir="ckpt",
        vectordb_name="na",
        resume=False,
        **kwargs,
    ):
        """
        Initializes the BaseVectorMem instance.

        Args:
            retrieval_top_k (int): The number of top-k results to retrieve in a similarity search.
            ckpt_dir (str): The directory path for saving and loading checkpoints.
            vectordb_name (str): The name of the vector database.
            resume (bool): Whether to resume from a checkpoint.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(
            retrieval_top_k=retrieval_top_k,
            ckpt_dir=ckpt_dir,
            vectordb_name=vectordb_name,
            **kwargs,
        )

        if resume:
            print(f"\033[33mLoading {vectordb_name} Manager from {ckpt_dir}/{vectordb_name}\033[0m")
            self.fn_str_map = load_json(f"{ckpt_dir}/{vectordb_name}/entries.json")
        else:
            self.fn_str_map = {}

    """
    helper fns
    """
    @staticmethod
    def log_content(docs):
        """
        formats the content of documents for logging

        Args:
            docs (list): A list of document objects.

        Returns:
            str: A formatted string of document contents.
        """
        return f"\033[33m" + '\n'.join([doc.page_content[:100] + '...' for doc in docs]) + "\033[0m"
    
    def _check_vectordb_sync(self, error_message):
        """Helper method to check if vectordb is in sync with fn_str_map"""
        assert self.vectordb._collection.count() == len(self.fn_str_map), error_message
    
    def get_docs(self, query, db, k, **kwargs):
        return db.similarity_search(query, k=k)

    """
    Retrieval Actions (to working mem / decision procedure)
    """
    def retrieve(
        self,
        query,
        db=None,
        db_name="",
        k_new=0,
        log_fn=None,
        log_docs=True,
        **kwargs,
    ):
        """
        Retrieves documents based on a query.

        Args:
            query (str): The query string for retrieval.
            db (object, optional): The database object. Defaults to None.
            db_name (str, optional): The name of the database. Defaults to "".
            k_new (int, optional): The number of new top-k results to retrieve. Defaults to 0.
            log_fn (function, optional): The function to log the documents. Defaults to None.
            log_docs (bool, optional): Whether to log the documents. Defaults to True.

        Returns:
            list: A list of retrieved documents.
        """
        if db is None:
            db = self.vectordb

        k = min(db._collection.count(), k_new if k_new else self.retrieval_top_k)

        docs = []
        if k:
            if not db_name:
                db_name = self.vectordb_name
            logger.info(f"\033[33m Retrieving {k} entries for db: {db_name} \n \033[0m")
            
            docs = self.get_docs(query, db, k, **kwargs)

        if log_docs and docs:
            if not log_fn:
                log_fn = self.log_content
            logger.info(log_fn(docs))

        return docs

    def retrieve_code(self, query, **kwargs):
        """
        Retrieves code snippets based on a query.
        meant for Voyager style code entries

        Args:
            query (str): The query string for retrieval.

        Returns:
            list: A list of retrieved code snippets.
        """
        docs = self.retrieve(query, log_fn=log_name, **kwargs)
        entries = [self.fn_str_map[doc.metadata['name']] for doc in docs]
        return entries

    """
    Learning Actions (from working mem)
    """
    def add_code(self, raw_data, mapping, description, prevent_duplicates=False):
        """
        Adds a new code snippet to the database.
        meant for Voyager style code entries

        Args:
            raw_data (dict): The raw data containing code snippets.
            mapping (list): A list of tuples mapping source keys to destination keys.
            description (str): A description of the code snippet.
            prevent_duplicates (bool, optional): Whether to prevent duplicate entries. Defaults to False.

        Returns:
            str: The name of the added code snippet.
        """
        processed_data = {}
        for source_key, destination_key in mapping:
            processed_data[destination_key] = raw_data.get(source_key, '')

        code = processed_data['code']
        name = processed_data.get("name", get_fn_name(code))

        name, old_name, exit_fn = new_name_if_exists(name, self.fn_str_map, prevent_duplicates, code)
        if exit_fn:
            return

        update_processed_data(processed_data, name, old_name, description)

        self.vectordb.add_texts(
            texts=[description],
            ids=[name],
            metadatas=[{"name": name}],
        )

        self.fn_str_map[name] = {k: v for k, v in processed_data.items() if k != "name"}
        
        self._check_vectordb_sync("vectordb is not synced with entries.json")
        
        dump_json(self.fn_str_map, f"{self.ckpt_dir}/{self.vectordb_name}/entries.json", indent=4)
        self.vectordb.persist()
        
        return name


def log_name(docs):
    """
    formats the names of documents for logging

    Args:
        docs (list): A list of document objects.

    Returns:
        str: A formatted string of document names.
    """
    return f"\033[33m {', '.join([doc.metadata['name'] for doc in docs])}\033[0m"


def check_duplicate_code(name, old_name, code, fn_str_map):
    curr_header = f'{name}('
    old_header = f'{old_name}('
    if code == fn_str_map[name]['code'].replace(curr_header, old_header):
        logger.info(f'{name} already in db so skip duplicates\n')
        return True
    return False


def new_name_if_exists(name, fn_str_map, prevent_duplicates, code):
    old_name = name
    ver = 1
    while name in fn_str_map:
        if prevent_duplicates:
            if check_duplicate_code(name, old_name, code, fn_str_map):
                return '', '', True

        ver += 1
        logger.info(f'incrementing ver to {ver}')
        name = f'{old_name}_v{ver}'

    return name, old_name, False


def update_processed_data(processed_data, name, old_name, description):
    if name != old_name:
        processed_data['code'] = processed_data['code'].replace(f'{old_name}(', f'{name}(')
        processed_data["name"] = name
        logger.info(f'{old_name} exists in db but different code so rename to {name}')

    processed_data["description"] = description
