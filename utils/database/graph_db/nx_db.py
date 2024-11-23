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

    # Node operations
    def get_nodes_by_attribute(self, attribute_name, attribute_value):
        return [node for node, attr in self.graph.nodes(data=True) if attr.get(attribute_name) == attribute_value]

    def get_node(self, node_id: str, return_id=False) -> dict:
        """
        Get a node from the graph.

        Args:
            node_id (str): The ID of the node to get.

        Returns:
            dict: The attributes of the node.
        """
        node_attributes = self.graph.nodes[node_id]
        if return_id:
            node_attributes['node_id'] = node_id
        return node_attributes

    def update_attributes(self, node_id: str, attributes: dict) -> None:
        """
        Update attributes of a node in the graph.

        Args:
            node_id (str): The ID of the node to update.
            attributes (dict): A dictionary of attributes to update.
        """
        # Note: this is full override. if you want update as in dict.update, use add_node
        nx.set_node_attributes(self.graph, {node_id: attributes})

    def add_node(self, node_id: str, verbose=False, **attributes: dict) -> bool:
        """
        Add a node to the graph.

        Args:
            node_id (str): The ID of the node to add.
            attributes (dict): A dictionary of attributes for the node.
        """
        has_diff = True
        if node_id in self.graph.nodes:
            has_diff = self.compare_attributes(self.graph.nodes[node_id], attributes, union=False)
            if has_diff:
                print(f"Node attributes differ for node {node_id}.\n")
            else:
                if verbose:
                    print(f"Node {node_id} already exists with the same attributes. Skipping update.\n")
                return has_diff
        if verbose:
            print('adding / updating node. Before:\n')
            self.print_node_attributes(node_id)

        self.graph.add_node(node_id, **attributes)

        if verbose:
            print('After:\n')
            self.print_node_attributes(node_id)
            print(f"new number of nodes {self.graph.number_of_nodes()}\n\n")

        return has_diff

    def remove_node(self, node_id: str) -> None:
        """
        Remove a node from the graph.

        Args:
            node_id (str): The ID of the node to remove.
        """
        self.graph.remove_node(node_id)

    # Edge operations
    def add_edge(self, subject: str, obj: str, relation: str, verbose=False, update=True, **attributes) -> None:
        """
        Add an edge to the graph.

        Args:
            subject (str): The ID of the first node.
            obj (str): The ID of the second node.
            relation (str): The type of relation between the entities.
        """
        # if attributes is None:
        #     attributes = {}
        # TODO: need to standardize the relation thing
        attributes['relation'] = relation

        if verbose:
            print(f"Adding edge {subject} {relation} {obj}\n")
            print(f"with attributes:\n")
            pp(attributes)

        has_diff = True
        if self.graph.has_edge(subject, obj):
            print(f"Warning: Edge from {subject} to {obj} already exists.\n")
            existing_attributes = self.graph.get_edge_data(subject, obj)
            has_diff = self.compare_attributes(existing_attributes, attributes)
            if not has_diff:
                if verbose:
                    print(f"Edge attributes are the same. Skipping update.\n")
                return has_diff, existing_attributes
            
            if verbose:
                print(f"Existing edge attributes:\n")
                pp(existing_attributes)
            if update:
                print("Updating edge attributes.\n")
                existing_attributes.update(attributes)
                attributes = existing_attributes

        self.graph.add_edge(subject, obj, **attributes)

        if verbose:
            print("Edge added. New Edge attributes:\n")
            pp(self.graph.get_edge_data(subject, obj))
            for node_id in [subject, obj]:
                pp([(u, d['relation'], v) for u, v, d in self.graph.out_edges(node_id, data=True)])
                pp([(u, d['relation'], v) for u, v, d in self.graph.in_edges(node_id, data=True)])

            print(f"new number of edges {self.graph.number_of_edges()}\n\n")

        return has_diff, attributes

    def remove_edge(self, subject: str, obj: str) -> None:
        """
        Remove an edge from the graph.

        Args:
            subject (str): The ID of the first node.
            obj (str): The ID of the second node.
        """
        self.graph.remove_edge(subject, obj)

    def get_edges_by_attribute(self, attr_name, attr_value):
        return [(u, v, attr) for u, v, attr in self.graph.edges(data=True) if attr.get(attr_name) == attr_value]
    
    # Search methods
    def search_keyword(self, keyword: str):
        """
        Search for a keyword in the attributes of nodes and edges in the knowledge graph.

        Args:
            keyword (str): The keyword to search for.

        Returns:
            dict: A dictionary with two keys 'nodes' and 'edges', each containing a list of matching nodes and edges.
        """
        
        matching_nodes = []
        matching_edges = []

        # Search in nodes
        for node, attrs in self.graph.nodes(data=True):
            if any(keyword in str(value) for value in attrs.values()):
                matching_nodes.append((node, attrs))

        # Search in edges
        for u, v, attrs in self.graph.edges(data=True):
            if any(keyword in str(value) for value in attrs.values()):
                matching_edges.append((u, v, attrs))

        return {'nodes': matching_nodes, 'edges': matching_edges}
    
    # Utility methods
    def get_neighbors(self, node_id):
        return list(self.graph.neighbors(node_id))
    
    def get_neighbor_edges(self, node_id, data=False):
        out_edges = list(self.graph.out_edges(node_id, data=data))
        in_edges = list(self.graph.in_edges(node_id, data=data))
        return out_edges + in_edges
    
    def get_path(self, source_id, target_id):
        return nx.shortest_path(self.graph, source=source_id, target=target_id)
    
    def print_node_attributes(self, node_id):
        if node_id in self.graph.nodes:
            node_attributes = self.graph.nodes[node_id]
            print(f"Node {node_id} attributes:\n")
            pp(node_attributes)
        else:
            print(f"Node {node_id} not found in the graph.\n")

    def compare_attributes(self, existing_attrs, new_attrs, union=True, exclude_keys=None):
        """
        Compare attributes of a node or edge.

        Args:
            existing_attrs (dict): The existing attributes.
            new_attrs (dict): The new attributes to compare.
            union (bool): If True, compare using the union of keys from both dictionaries.
                          If False, compare using only the keys from new_attrs.
        """
        keys = set(existing_attrs.keys()).union(new_attrs.keys()) if union else set(new_attrs.keys())
        if exclude_keys:
            keys = keys - set(exclude_keys)

        has_diff = False
        for key in keys:
            existing_value = existing_attrs.get(key)
            new_value = new_attrs.get(key)
            if existing_value != new_value:
                print(f"\nAttribute '{key}' differs:\n existing='{existing_value}'\n\n new='{new_value}'")
                has_diff = True

        return has_diff
