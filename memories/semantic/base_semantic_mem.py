"""
scaffold for semantic memory. WIP
"""
# TODO: import more from original codebase
from ..base_mem import BaseMem


class BaseSemanticMem(BaseMem):
    def __init__(
        self,
        retrieval_top_k=5,
        ckpt_dir="ckpt",
        vectordb_name="semantic",
        resume=True,
        **kwargs,
    ):
        super().__init__(
            retrieval_top_k=retrieval_top_k,
            ckpt_dir=ckpt_dir,
            vectordb_name=vectordb_name,
            resume=resume,
            **kwargs,
        )
        
        self.knowledge_sources = []
        
        self.register_vectordb_with_methods('summaries')
        self.register_vectordb_with_methods('reflections')

    """
    Retrieval Actions (to working mem / decision procedure)
    """
    def get_knowledge_sources(self):
        """
        Retrieves all knowledge sources from the semantic memory.

        Returns:
            list: A list of dictionaries containing knowledge sources.
        """
        return self.knowledge_sources

    def retrieve_summaries(self, query, **kwargs):
        """
        Retrieves summaries from the vector database based on similarity to a text.

        Args:
            query: The query text.
            **kwargs: Additional arguments passed to the retrieval method.

        Returns:
            Union[List[str], List[Tuple[str, float]]]: Either:
                - A list of formatted strings with proper indentation
                - A list of tuples (formatted_string, score) if scores were requested
        """
        return self._retrieve_and_format(query, 'summaries', 'Summary', **kwargs)

    def retrieve_reflections(self, query, **kwargs):
        """
        Retrieves reflections from the vector database based on similarity to a text.

        Args:
            query: The query text.
            **kwargs: Additional arguments passed to the retrieval method.

        Returns:
            Union[List[str], List[Tuple[str, float]]]: Either:
                - A list of formatted strings with proper indentation
                - A list of tuples (formatted_string, score) if scores were requested
        """
        return self._retrieve_and_format(query, 'reflections', 'Reflection', **kwargs)
    
    def retrieve_textbook(self, query, **kwargs):
        """
        Retrieves textbook reference material from the vector database and formats it with proper indentation.

        Args:
            query: The query text.
            **kwargs: Additional arguments passed to the retrieval method.

        Returns:
            Union[List[str], List[Tuple[str, float]]]: Either:
                - A list of formatted strings with proper indentation
                - A list of tuples (formatted_string, score) if scores were requested
        """
        return self._retrieve_and_format(query, 'vector', 'Textbook Reference Material', **kwargs)

    """
    Learning Actions (from working mem)
    """
    def add_knowledge_source(self, source):
        """
        Adds a knowledge source to the semantic memory.

        Args:
            source (dict): A dictionary containing information about the knowledge source.
        """
        self.knowledge_sources.append(source)

    def update_summaries(self, entry, **kwargs):
        """
        Stores an embedding in the summaries vector database.

        Args:
            entry: The embedding to store.
            **kwargs: Additional arguments passed to the update method.
        """
        self.update_methods['summaries'].update(entry, **kwargs)

    def update_reflections(self, entry, **kwargs):
        """
        Stores an embedding in the reflections vector database.

        Args:
            entry: The embedding to store.
            **kwargs: Additional arguments passed to the update method.
        """
        self.update_methods['reflections'].update(entry, **kwargs)


# TODO: merge below with above
# from cognitive_base.utils.database.database_manager import DatabaseManager
#
# class SemanticMemory:
#     def __init__(self, vector_db_config, relational_db_path):
#         self.db_manager = DatabaseManager(vector_db_config, relational_db_path)
#
#     def add_reflection(self, table_name, entry):
#         """
#         Adds a reflection or summary to the semantic memory.
#
#         Args:
#             table_name (str): Name of the table.
#             entry (dict): Dictionary containing column-value pairs.
#         """
#         self.db_manager.add_to_relational_db(table_name, entry)
#
#     def get_reflections(self, table_name, query_conditions):
#         """
#         Retrieves reflections and summaries from the semantic memory.
#
#         Args:
#             table_name (str): Name of the table.
#             query_conditions (str): SQL conditions for the query.
#
#         Returns:
#             list: List of query results.
#         """
#         return self.db_manager.query_relational_db(table_name, query_conditions)
