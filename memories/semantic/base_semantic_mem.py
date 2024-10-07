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
        self.reflections = []

    def add_knowledge_source(self, source):
        """
        Adds a knowledge source to the semantic memory.

        Args:
            source (dict): A dictionary containing information about the knowledge source.
        """
        self.knowledge_sources.append(source)

    def add_reflection(self, reflection):
        """
        Adds a reflection or summary to the semantic memory.

        Args:
            reflection (dict): A dictionary containing information about the reflection or summary.
        """
        self.reflections.append(reflection)

    def get_knowledge_sources(self):
        """
        Retrieves all knowledge sources from the semantic memory.

        Returns:
            list: A list of dictionaries containing knowledge sources.
        """
        return self.knowledge_sources

    def get_reflections(self):
        """
        Retrieves all reflections and summaries from the semantic memory.

        Returns:
            list: A list of dictionaries containing reflections and summaries.
        """
        return self.reflections

# TODO: merge below with above
# from cognitive_base.utils.database.database_manager import DatabaseManager
#
# class SemanticMemory:
#     def __init__(self, vector_db_config, relational_db_path):
#         self.db_manager = DatabaseManager(vector_db_config, relational_db_path)
#
#     def add_knowledge_source(self, vector, metadata):
#         """
#         Adds a knowledge source to the semantic memory.
#
#         Args:
#             vector (list): The vector representation of the data.
#             metadata (dict): Metadata associated with the vector.
#         """
#         self.db_manager.add_to_vector_db(vector, metadata)
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
#     def get_knowledge_sources(self, vector, top_k=10):
#         """
#         Retrieves knowledge sources from the semantic memory.
#
#         Args:
#             vector (list): The vector to query.
#             top_k (int): Number of top results to return.
#
#         Returns:
#             list: List of top_k results.
#         """
#         return self.db_manager.query_vector_db(vector, top_k)
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
