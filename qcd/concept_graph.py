"""This file implements two main components: a 'concept graph' and a parser for building up a concept graph."""

from collections import defaultdict
from math import log2
from typing import Type, Set, Dict, Optional, List, Tuple, NewType, Union

import graphviz
import networkx as nx

from qcd.graph import Node, DirectedEdgeI, GraphI, Section
from qcd.parser import ParserI


class DirectedEdge(DirectedEdgeI):
    """The base class representing a connection between two nodes in a graph."""

    def __init__(self, tail: Node, head: Node, weight: float = 1):
        """Create an edge between two nodes.

        :param tail: The node from which the edge originates from, or points
                     aways from.
        :param head: The node that the edge points towards.
        :param weight: The weighting of edge frequency (how often the edge is
                       added to, or appears in, a graph.
        """
        self._tail: Node = tail
        self._head: Node = head
        self.weight: float = weight
        self.frequency: int = 1  # the number of the times this edge occurs

        self._colour: str = 'black'
        self._style: str = 'solid'
        self._label: str = ''

    def __eq__(self, other: 'DirectedEdge'):
        return self._tail == other._tail and self._head == other._head

    def __hash__(self):
        return hash(self._tail + self._head)

    def __str__(self):
        class_ = self.__class__.__name__
        return '%s(\'%s\', \'%s\', %.2f)' % (class_, self._tail, self._head, self.weight)

    def __repr__(self):
        return str(self)

    @property
    def nodes(self) -> Tuple[Node, Node]:
        return self._tail, self._head

    @property
    def tail(self) -> Node:
        return self._tail

    @property
    def head(self) -> Node:
        return self._head

    @property
    def colour(self) -> str:
        return self._colour

    @colour.setter
    def colour(self, value: str):
        self._colour = value

    @property
    def style(self) -> str:
        return self._style

    @style.setter
    def style(self, value: str):
        self._style = value

    @property
    def label(self) -> str:
        return self._label

    @label.setter
    def label(self, value: str):
        self._label = value

    @property
    def weighted_frequency(self):
        """The weighted frequency of the edge."""
        return self.weight * self.frequency

    @property
    def log_weighted_frequency(self):
        """A logarithmically scaled version of the edge's frequency count.

        This is used when rendering an edge so that the edges do not get
        arbitrarily thick as the weight gets large.
        """
        # Log weight:
        # > Start at one so that edges are always at least one unit thick
        # > Add one to log argument so that the return value is strictly
        # non-negative, weight=0 gives a return value of 0 and avoids
        # log(0) which is undefined.
        return 1 + log2(1 + self.weighted_frequency)

    def render(self, g: graphviz.Digraph, colour: Optional[str] = None):
        """Render/draw the edge using GraphViz.

        :param g: The graphviz graph instance.
        :param colour: The colour to render the edge. If None then the edge's colour attribute is used.
        """
        g.edge(self._tail, self._head,
               penwidth=str(self.log_weighted_frequency),
               color=colour if colour else self.colour,
               style=self.style,
               label=self.label)


Relation = NewType('Relation', str)


class RelationalEdge(DirectedEdge):
    def __init__(self, tail: Node, head: Node, relation: Relation = '', weight: float = 1.0):
        super().__init__(tail, head, weight)

        self._relation = relation

    @property
    def label(self) -> str:
        # NOTE: The parent class's label property which has both a setter and getter, by overriding just the getter here
        #       the property is turned into a read-only property (i.e. no setter is inherited from the parent class).
        return self.relation

    @property
    def relation(self) -> Relation:
        return self._relation

    def __eq__(self, other: Union[DirectedEdge, 'RelationalEdge']):
        if isinstance(other, RelationalEdge):
            return super(RelationalEdge, self).__eq__(other) and self.relation == other.relation
        else:
            return super(RelationalEdge, self).__eq__(other)

    def __hash__(self):
        return hash(self._tail + self._head + self.relation)

    def __str__(self):
        class_ = self.__class__.__name__
        return '%s(\'%s\', \'%s\', \'%s\', %.2f)' % (class_, self._tail, self._head, self.relation, self.weight)


class ForwardReference(RelationalEdge):
    """An edge that references a node in a section that comes after the section
    that the tail node is in."""

    def __init__(self, tail: Node, head: Node, weight: float = 2.0):
        super().__init__(tail, head, weight=weight)

        self.colour = 'blue'


class BackwardReference(RelationalEdge):
    """An edge that references a node in a section that comes before the section
    that the tail node is in."""

    def __init__(self, tail: Node, head: Node, weight: float = 1.5):
        super().__init__(tail, head, weight=weight)

        self.colour = 'red'


# TODO: Get rid of this somehow? Maybe?
class ImplicitReference(DirectedEdge):
    def __init__(self, tail: Node, head: Node, weight: float = 1.0):
        super().__init__(tail, head, weight=weight)


class ConceptGraph(GraphI):
    # TODO: Add 'dirty' flag to indicate the graph has been changed since postprocessing() was last called.
    def __init__(self, parser: ParserI, mark_references=True):
        """Create an empty graph.

        :param parser: The parser to use.
        :param mark_references: Whether or not to mark forward and backward references after parsing.
        """
        # The set of all nodes (vertices) in the graph.
        self.nodes: Set[Node] = set()

        # Maps section to nodes found in that section.
        self.section_listings: Dict[Section, Set[Node]] = defaultdict(set)
        # Maps node names to sections:
        self.section_index: Dict[Node, Optional[Section]] = defaultdict(lambda: None)
        # Set of sections nodes (the main concept of a given section).
        self.section_nodes: Set[Node] = set()
        # List of sections used to record order that sections are introduced.
        self.sections: List[Section] = list()
        # Maps nodes to the frequency they appear in each section.
        self.section_counts: Dict[Node, Dict[Section, int]] = defaultdict(lambda: defaultdict(int))

        # Maps tail to head nodes
        self.adjacency_list: Dict[Node, Set[Node]] = defaultdict(set)
        # Maps head to tail nodes
        self.adjacency_index: Dict[Node: Set[Node]] = defaultdict(set)
        # The set of all edges in the graph
        self.edges: Set[DirectedEdge] = set()
        # Maps (tail, head) pairs to edge instance
        self.edge_index: Dict[Tuple[Node, Node], Optional[DirectedEdge]] = defaultdict(lambda: None)

        # Set of forward references
        self.forward_references: Set[DirectedEdge] = set()
        # Set of backward references
        self.backward_references: Set[DirectedEdge] = set()
        # Set of a priori concepts (nodes)
        self.a_priori_concepts: Set[Node] = set()
        # Set of emerging entities (nodes)
        self.emerging_concepts: Set[Node] = set()
        # Set of self-referential loops
        self.cycles: List[List[Node]] = list()
        # Set of disjointed_subgraphs
        self.subgraphs: List[Set[Node]] = list()

        # Parse Options #
        self.parser: ParserI = parser
        self.mark_references: bool = mark_references

        # Misc #
        self.nx: Optional[nx.DiGraph] = None

    @property
    def mean_outdegree(self):
        return sum([len(self.adjacency_list[node]) for node in self.nodes]) / len(self.nodes)

    @property
    def mean_section_outdegree(self):
        avg_degree = 0

        for section in self.sections:
            avg_degree += sum(
                [len(self.adjacency_list[node]) for node in self.section_listings[section]]) / len(
                self.section_listings[section])

        avg_degree /= len(self.sections)

        return avg_degree

    @property
    def mean_weighted_outdegree(self):
        return sum([edge.weight for edge in self.edges]) / len(self.nodes)

    @property
    def mean_weighted_section_outdegree(self):
        avg_degree = 0

        for section in self.sections:
            section_degree = 0

            for tail in self.section_listings[section]:
                for head in self.adjacency_list[tail]:
                    section_degree += self.get_edge(tail, head).weight

            section_degree /= len(self.section_listings[section])
            avg_degree += section_degree

        avg_degree /= len(self.sections)

        return avg_degree

    @property
    def mean_cycle_length(self):
        if len(self.cycles) > 0:
            return sum([len(cycle) for cycle in self.cycles]) / len(self.cycles)
        else:
            return 0

    @property
    def mean_subgraph_size(self):
        if len(self.subgraphs) > 0:
            return sum([len(subgraph) for subgraph in self.subgraphs]) / len(self.subgraphs)
        else:
            return 0

    def add_node(self, node: Node, section: Section):
        """Add a node to the graph.

        :param node: The node to add.
        :param section: The section that the node appeared in.
        """
        # Main section node was referenced before but now is being added under its own section,
        # update the relevant section details.
        if node in self.nodes and node == section:
            self.section_listings[self.section_index[node]].remove(node)
            self.section_index[node] = section
            self.section_listings[section].add(node)
            self.section_nodes.add(node)
        elif node not in self.nodes:
            self.nodes.add(node)
            self.section_index[node] = section
            self.section_listings[section].add(node)

            if section not in self.sections:
                self.sections.append(section)

            if node == section:
                self.section_nodes.add(node)

        self.update_section_count(node, section)

    def add_edge(self, tail: Node = None, head: Node = None, edge_type: Type[DirectedEdge] = DirectedEdge,
                 edge: DirectedEdge = None) -> Optional[DirectedEdge]:
        """Add an edge between two nodes to the graph.

        :param tail: The node that the edge originates from.
        :param head: The node that the edge points to.
        :param edge_type: The type of edge to be created.
        :return: The edge instance, possibly None.
        """
        if edge is None:
            assert tail is not None and head is not None, \
                'The parameters `head` and `tail` must both be set if no edge instance is given.'

            # Ignore spurious edges, a concept cannot be defined directly in terms of itself
            if tail == head:
                return None

            # Get edge if it exists in the graph already...
            edge = self.get_edge(tail, head)

            # Otherwise create a new edge instance
            if edge is None:
                edge = edge_type(tail, head)
        else:
            tail = edge.tail
            head = edge.head

        assert tail in self.nodes and head in self.nodes, 'Both nodes in the edge must exist within the graph.'

        if edge in self.edges:
            # Duplicate edges only increase a count and each unique edge is
            # only rendered once.
            edge.frequency += 1
        else:
            self.adjacency_list[tail].add(head)
            self.adjacency_index[head].add(tail)
            self.edges.add(edge)
            self.edge_index[(tail, head)] = edge

        return edge

    def add_relation(self, subject: Node, relation: Relation, object_: Node, section: Section):
        """Add a relation between two concepts to the graph.
        This is shorthand for adding the both of the concepts as nodes and adding an edge between those nodes.

        :param subject: The subject of the relation.
        :param relation: The verb of the relation.
        :param object_: The object of the relation.
        :param section: The section that the concepts appear in.
        """
        self.add_node(subject, section)
        self.add_node(object_, section)
        self.add_edge(edge=RelationalEdge(subject, object_, relation=relation))

    def remove_edge(self, tail: Optional[Node] = None, head: Optional[Node] = None,
                    edge: Optional[DirectedEdge] = None):
        """Remove an edge from the graph.

        :param tail: The tail node of the edge to remove.
        :param head: The head node of the edge to remove.
        :param edge: The edge object to remove. The parameters `tail` and `head`
                     are ignored if this parameter is not `None`.
        """
        try:
            if not edge:
                assert tail is not None and head is not None, \
                    'The parameters `head` and `tail` must both be set if no edge instance is given.'

                edge = self.edge_index[(tail, head)]

            self.edges.discard(edge)
            del self.edge_index[(tail, head)]

            self.adjacency_list[tail].discard(head)
            self.adjacency_index[head].discard(tail)
        except KeyError:
            # non-existent edge, probably
            pass

    def get_edge(self, tail: Node, head: Node) -> DirectedEdge:
        """Get the edge that connects the nodes corresponding to `tail` and `head`.

        :param tail: The name of the node that the edge originates from.
        :param head: The name of the node that the edge points to.
        :return: The corresponding edge if it exists in the graph, None otherwise.
        """
        return self.edge_index[(tail, head)]

    def set_edge(self, edge: DirectedEdge) -> DirectedEdge:
        """Set (delete and add) an edge in the graph.

        The edge that is deleted will be the edge that has the same tail and
        head as the function argument `edge`.

        :param edge: The new edge that should replace the one in the graph.
        :return: The new edge.
        """
        self.remove_edge(edge=edge)

        return self.add_edge(edge=edge)

    def update_section_count(self, node: Node, section: Section):
        """Update the count of times a node appears in a given section by one.

        :param node: The node.
        :param section: The section the node was found in.
        """
        self.section_counts[node][section] += 1

    def parse(self, filename: str):
        """Parse a XML document and build up a graph structure.

        :param filename: The XML file to parse.
        """
        self.parser.parse(filename, self)
        self.postprocessing()

    def postprocessing(self):
        """Perform tasks to fix/normalise the graph structure"""
        self._reassign_implicit_entities()
        self._reassign_sections()
        self._categorise_nodes()
        self.nx = self.to_nx()

        self.mark_edges()
        self.find_cycles()
        self.find_subgraphs()

    def to_nx(self):
        """Convert the graph to a NetworkX graph.

        :return: The graph as an NetworkX graph.
        """
        g = nx.DiGraph()
        g.add_nodes_from([node for node in self.nodes])
        g.add_edges_from([(edge.tail, edge.head) for edge in self.edges])

        return g

    def _reassign_implicit_entities(self):
        """Correct the sections for entities derived from a main section node.

        If an implicit entity that is derived from a main section node and it is referenced before that section in a
        preceding section, then it will be incorrectly assigned to the preceding section.
        For example, in `bread.xml` the section on bread references (the main entity) wheat flour, which creates three
        nodes: 'wheat flour', 'wheat', and 'flour'. Initially these are all assigned the section 'bread',
        which is incorrect since 'wheat flour' is its own section and any nodes derived from this should also be of
        the same section.
        """
        for node in self.section_nodes:
            for child in self.adjacency_list[node]:
                edge = self.get_edge(node, child)

                if isinstance(edge, ImplicitReference):
                    child_section = self.section_index[child]
                    node_section = self.section_index[node]

                    try:
                        self.section_listings[child_section].remove(child)
                    except KeyError:
                        pass

                    self.section_listings[node_section].add(child)
                    self.section_index[child] = node_section

    def _reassign_sections(self):
        """Reassign nodes to another section if the node appears more times in that section."""
        for node in self.nodes:
            if node == self.section_index[node]:
                continue

            prev_section = self.section_index[node]
            new_section = max(self.section_counts[node], key=lambda key: self.section_counts[node][key])

            if new_section != prev_section:
                self.section_listings[prev_section].remove(node)
                self.section_index[node] = new_section
                self.section_listings[new_section].add(node)

    def _categorise_nodes(self):
        """Categorise nodes in the graph into 'a priori' and 'emerging' concepts.

        Nodes that are only referenced from one section represent 'a priori references',
        all other nodes represent 'emerging concepts'.
        """
        for section in self.sections:
            for node in self.section_listings[section]:
                referencing_sections = set()

                for tail in self.adjacency_index[node]:
                    tail_section = self.section_index[tail]
                    referencing_sections.add(tail_section)

                if len(referencing_sections) == 1:
                    for tail in self.adjacency_index[node]:
                        the_edge = self.get_edge(tail, node)
                        the_edge.weight = 0.5

                    self.a_priori_concepts.add(node)
                else:
                    self.emerging_concepts.add(node)

    def mark_edges(self):
        """Colour edges as either forward or backward edges."""
        self.forward_references = set()
        self.backward_references = set()

        visited = set()

        for node in self.nodes:
            self._mark_edges(node, None, visited)

    def _mark_edges(self, curr: Node, prev: Optional[Node], visited: set):
        """Recursively mark edges.

        :param curr: The next node to vist.
        :param prev: The node that was previously visited.
        :param visited: The set of nodes that have already been visited.
        """
        # We have reached a 'leaf node' which is a node belonging to another section
        if prev and self.section_index[curr] != self.section_index[prev]:
            # Check if the path goes forward from section to a later section, or vice versa
            curr_i = self.sections.index(self.section_index[curr])
            prev_i = self.sections.index(self.section_index[prev])

            if curr_i < prev_i:
                self._mark_edge(prev, curr, BackwardReference)
            elif curr_i > prev_i:
                self._mark_edge(prev, curr, ForwardReference)

        if curr not in visited:
            # Otherwise we continue the depth-first traversal
            visited.add(curr)

            for child in self.adjacency_list[curr]:
                self._mark_edges(child, curr, visited)

    def _mark_edge(self, tail: Node, head: Node, edge_type: Type[DirectedEdge] = DirectedEdge):
        """Mark an edge.

        :param tail: The node from which the edge originates from, or points
                     aways from.
        :param head: The node that the edge points towards.
        :param edge_type: The type to mark the edge as.
        """
        edge = self.get_edge(tail, head)

        new_edge = edge_type(tail, head)
        new_edge = self.set_edge(new_edge)
        # preserve edge style for cases where there are implicit
        # forward/backward edges
        new_edge.style = edge.style
        new_edge.frequency = edge.frequency

        if isinstance(new_edge, ForwardReference):
            self.forward_references.add(new_edge)
        elif isinstance(new_edge, BackwardReference):
            self.backward_references.add(new_edge)

    def find_cycles(self) -> List[List[Node]]:
        """Find cycles in the graph."""
        self.cycles = [cycle for cycle in nx.simple_cycles(self.nx)]

        return self.cycles

    def find_subgraphs(self) -> List[Set[Node]]:
        """Find disjointed subgraphs in the graph.

        :return: A list of the subgraphs.
        """
        self.subgraphs = [nodes for nodes in nx.connected_components(self.nx.to_undirected())]

        return self.subgraphs

    def print_summary(self):
        sep = '=' * 80

        print(sep)
        print('Summary of Graph')
        print(sep)
        print('Nodes:', len(self.nodes))
        print('Edges:', len(self.edges))

        print(sep)
        print('Avg. Outdegree: %.2f' % self.mean_outdegree)
        print('Avg. Section Outdegree: %.2f' % self.mean_section_outdegree)
        print('Avg. Weighted Outdegree: %.2f' % self.mean_weighted_outdegree)
        print('Avg. Weighted Section Outdegree: %.2f' % self.mean_weighted_section_outdegree)

        if len(self.subgraphs) > 1:
            print(sep)
            print('Disjointed Subgraphs:', len(self.subgraphs))
            print('Avg. Disjointed Subgraph Size: %.2f' % (
                    sum(len(subgraph) for subgraph in self.subgraphs) / len(self.subgraphs)))

        print(sep)
        print('Forward References:', len(self.forward_references))
        print('Backward References:', len(self.backward_references))
        print('A Priori Concepts:', len(self.a_priori_concepts))
        print('Emerging Concepts:', len(self.emerging_concepts))

        if len(self.cycles) > 0:
            print(sep)
            print('Cycles:', len(self.cycles))
            print('Avg. Cycle Length: %.2f' % (sum([len(cycle) for cycle in self.cycles]) / len(self.cycles)))

        print(sep)

    def score(self) -> float:
        """Calculate a score of conceptual density for the given graph.

        :return: The score for the graph as a non-negative scalar.
        """
        return self.mean_weighted_outdegree

    # TODO: Add debug rendering mode that shows more info such as edge and node frequency
    def render(self, filename='concept_graph', view=True):
        """Render the graph using GraphViz.

        :param filename: The name of the file to save the output to.
        :param view: Whether or not to show the output in a window.
        """
        try:
            g = graphviz.Digraph(engine='neato')
            g.attr(overlap='false')

            for node in self.nodes:
                g.node(node)

            if self.mark_references:
                for edge in self.edges:
                    edge.render(g)
            else:
                for edge in self.edges:
                    edge.render(g, colour='black')

            g.render(filename, format='png', view=view, cleanup=True)
        except graphviz.backend.ExecutableNotFound:
            print('Could not display graph -- GraphViz does not seem to be installed.')
