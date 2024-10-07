"""
scaffold for relational database
"""
from ..base_db import BaseDB


class BaseRelationalDB(BaseDB):
    def update(self, table_name, entry, **kwargs):
        """
        Inserts an entry into the relational database.

        Args:
            table_name (str): Name of the table.
            entry (dict): Dictionary containing column-value pairs.
        """
        raise NotImplementedError("Subclasses should implement this method")

    def retrieve(self, table_name, query_conditions, **kwargs):
        """
        Queries the relational database.

        Args:
            table_name (str): Name of the table.
            query_conditions (str): SQL conditions for the query.

        Returns:
            list: List of query results.
        """
        raise NotImplementedError("Subclasses should implement this method")

