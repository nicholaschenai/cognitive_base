import networkx as nx

from pprint import pp
from .base_graph_db import BaseGraphDB


class NxDb(BaseGraphDB):
    def __init__(self, graph_type="directed", **kwargs):
        # TODO: future: more graph types
        self.graph = nx.DiGraph() if graph_type == "directed" else nx.Graph()

    def count(self):
        # get number of nodes and edges
        return self.graph.number_of_nodes(), self.graph.number_of_edges()

    def print_node_attributes(self, node_id):
        if node_id in self.graph.nodes:
            node_attributes = self.graph.nodes[node_id]
            print(f"Node {node_id} attributes:\n")
            pp(node_attributes)
        else:
            print(f"Node {node_id} not found in the graph.\n")

    def get_nodes_by_attribute(self, attribute_name, attribute_value):
        return [node for node, attr in self.graph.nodes(data=True) if attr.get(attribute_name) == attribute_value]

    def get_node(self, node_id: str) -> dict:
        """
        Get a node from the graph.

        Args:
            node_id (str): The ID of the node to get.

        Returns:
            dict: The attributes of the node.
        """
        return self.graph.nodes[node_id]

    def update_attributes(self, node_id: str, attributes: dict) -> None:
        """
        Update attributes of a node in the graph.

        Args:
            node_id (str): The ID of the node to update.
            attributes (dict): A dictionary of attributes to update.
        """
        # Note: this is full override. if you want update as in dict.update, use add_node
        nx.set_node_attributes(self.graph, {node_id: attributes})

    def add_node(self, node_id: str, verbose=False, **attributes: dict) -> None:
        """
        Add a node to the graph.

        Args:
            node_id (str): The ID of the node to add.
            attributes (dict): A dictionary of attributes for the node.
        """
        if verbose:
            print('adding / updating node. Before:\n')
            self.print_node_attributes(node_id)
        self.graph.add_node(node_id, **attributes)
        if verbose:
            print('After:\n')
            self.print_node_attributes(node_id)
            print(f"new number of nodes {self.graph.number_of_nodes()}\n\n")

    def remove_node(self, node_id: str) -> None:
        """
        Remove a node from the graph.

        Args:
            node_id (str): The ID of the node to remove.
        """
        self.graph.remove_node(node_id)

    def add_edge(self, subject: str, relation: str, obj: str, verbose=False, update=True, **attributes) -> None:
        """
        Add an edge to the graph.

        Args:
            subject (str): The ID of the first node.
            obj (str): The ID of the second node.
            relation (str): The type of relation between the entities.
        """
        # if attributes is None:
        #     attributes = {}

        if verbose:
            print(f"Adding edge {subject} {relation} {obj}\n")
            print(f"with attributes:\n")
            pp(attributes)

        if self.graph.has_edge(subject, obj):
            print(f"Warning: Edge from {subject} to {obj} already exists.\n")
            existing_attributes = self.graph.get_edge_data(subject, obj)
            if verbose:
                print(f"Existing edge attributes:\n")
                pp(existing_attributes)
            if update:
                print("Updating edge attributes.\n")
                # Update existing edge attributes
                existing_attributes.update(attributes)
                existing_attributes.pop('relation')
                attributes = existing_attributes

        self.graph.add_edge(subject, obj, relation=relation, **attributes)

        if verbose:
            print("Edge added. New Edge attributes:\n")
            pp(self.graph.get_edge_data(subject, obj))
            for node_id in [subject, obj]:
                pp([(u, d['relation'], v) for u, v, d in self.graph.out_edges(node_id, data=True)])
                pp([(u, d['relation'], v) for u, v, d in self.graph.in_edges(node_id, data=True)])

            print(f"new number of edges {self.graph.number_of_edges()}\n\n")

    def remove_edge(self, subject: str, obj: str) -> None:
        """
        Remove an edge from the graph.

        Args:
            subject (str): The ID of the first node.
            obj (str): The ID of the second node.
        """
        self.graph.remove_edge(subject, obj)
