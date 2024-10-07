"""
scaffold for vector database
"""
from ..base_db import BaseDB


class BaseGraphDB(BaseDB):
    def update_attributes(self, node_id: str, attributes: dict) -> None:
        """
        Update attributes of an node in the graph.

        Args:
            node_id (str): The ID of the node to update.
            attributes (dict): A dictionary of attributes to update.
        """
        raise NotImplementedError("Subclasses should implement this method")

    def add_node(self, node_id: str, attributes: dict) -> None:
        """
        Add a node to the graph.

        Args:
            node_id (str): The ID of the node to add.
            attributes (dict): A dictionary of attributes for the node.
        """

    def remove_node(self, node_id: str) -> None:
        """
        Remove a node from the graph.

        Args:
            node_id (str): The ID of the node to remove.
        """
        raise NotImplementedError("Subclasses should implement this method")

    def add_edge(self, subject: str, obj: str, relation: str, attributes: dict) -> None:
        """
        Add an edge to the graph.

        Args:
            subject (str): The ID of the first node.
            obj (str): The ID of the second node.
            relation (str): The type of relation between the entities.
            attributes (dict): A dictionary of attributes for the edge.
        """
        raise NotImplementedError("Subclasses should implement this method")

    def remove_edge(self, subject: str, obj: str) -> None:
        """
        Remove an edge from the graph.

        Args:
            subject (str): The ID of the first node.
            obj (str): The ID of the second node.
        """
        raise NotImplementedError("Subclasses should implement this method")
