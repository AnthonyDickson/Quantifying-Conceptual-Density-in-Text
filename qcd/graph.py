from abc import ABC
from typing import Type, Optional, Tuple

Node = str


class EdgeI(ABC):
    """The abstract interface for an edge in a graph."""

    @property
    def nodes(self) -> Tuple[Node, Node]:
        """Get the nodes that make up an edge.

        :return: The pair of nodes that make up the edge.
        """
        raise NotImplementedError


class DirectedEdgeI(EdgeI, ABC):
    """The interface for a directed edge in a graph."""

    @property
    def tail(self) -> Node:
        """Get the tail node of an edge.
        The tail node is the node that the edge points away from, or originates from.

        :return: The tail node of the edge.
        """
        raise NotImplementedError

    @property
    def head(self) -> Node:
        """Get the head node of an edge.
        The head node is the node that the edge points towards.

        :return: The head node of the edge.
        """
        raise NotImplementedError


class GraphI:
    """An interface describing the guaranteed functionality of a graph object."""

    def add_node(self, node: Node, section: str):
        """Add a node to the graph.

        :param node: The node to add.
        :param section: The section that the node appeared in.
        """
        raise NotImplementedError

    def add_edge(self, tail: str, head: str, edge_type: Type[EdgeI] = EdgeI) -> Optional[EdgeI]:
        """Add an edge between two nodes to the graph.

        :param tail: The node that the edge originates from.
        :param head: The node that the edge points to.
        :param edge_type: The type of edge to be created.
        :return: The edge instance, possibly None.
        """
        raise NotImplementedError
