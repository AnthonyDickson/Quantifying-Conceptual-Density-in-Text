import argparse
import xml.etree.ElementTree as ET
from collections import defaultdict
from math import log2
from typing import Set, List, Dict, Optional, Tuple, Type

import graphviz
import networkx as nx
import neuralcoref
import nltk
import spacy

from concept_graph import Parser


class Node:
    """A node/vertex in a graph."""

    def __eq__(self, other):
        return self.name == other.name

    def __init__(self, name: str):
        """Create a node representing a named entity that appears in a section
        of a given text.

        :param name: The name of the entity that the node represents.
        """
        self.name = name

    def __hash__(self):
        return hash(self.name)

    def __str__(self):
        return self.name

    def __repr__(self):
        return "Node('%s')" % self.name

    def render(self, g: graphviz.Digraph):
        """Render/draw the node using GraphViz.

        :param g: The graphviz graph instance.
        """
        g.node(self.name)


class Edge:
    """A connection between two nodes in a graph."""

    def __init__(self, tail: Node, head: Node, weight=1):
        """Create an edge between two nodes.

        :param tail: The node from which the edge originates from, or points
                     aways from.
        :param head: The node that the edge points towards.
        :param weight: The weighting of edge frequency (how often the edge is
                       added to, or appears in, a graph.
        """
        self.tail = tail
        self.head = head
        self.weight = weight
        self.frequency = 1  # the number of the times this edge occurs

        self.color = 'black'
        self.style = 'solid'

    def __eq__(self, other):
        return self.tail == other.tail and self.head == other.head

    def __hash__(self):
        return hash(self.tail.name + self.head.name)

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

    def render(self, g: graphviz.Digraph):
        """Render/draw the edge using GraphViz.

        :param g: The graphviz graph instance.
        """
        g.edge(self.tail.name, self.head.name,
               penwidth=str(self.log_weighted_frequency),
               color=self.color,
               style=self.style)


class ForwardEdge(Edge):
    """An edge that references a node in a section that comes after the section
    that the tail node is in."""

    def __init__(self, tail, head, weight=2.0):
        super().__init__(tail, head, weight)

        self.color = 'blue'


class BackwardEdge(Edge):
    """An edge that references a node in a section that comes before the section
    that the tail node is in."""

    def __init__(self, tail, head, weight=1.5):
        super().__init__(tail, head, weight)

        self.color = 'red'


class ImplicitEdge(Edge):
    def __init__(self, tail, head, weight=0.5):
        super().__init__(tail, head, weight)

        self.style = 'dashed'


class Graph:
    def __init__(self):
        """Create an empty graph."""
        # The set of all nodes (vertices) in the graph.
        self.nodes: Set[Node] = set()
        # Maps node name (concept) to node instance.
        self.node_index: Dict[str, Node] = dict()

        # Maps tail to head nodes
        self.adjacency_list: Dict[str, Set[Node]] = defaultdict(set)
        # Maps head to tail nodes
        self.adjacency_index: Dict[str: Set[Node]] = defaultdict(set)
        # The set of all edges in the graph
        self.edges: Set[Edge] = set()
        # Maps (tail, head) pairs to edge instance
        self.edge_index: Dict[Tuple[str, str], Optional[Edge]] = defaultdict(lambda: None)

        # Set of self-referential loops
        self.cycles: List[List[Node]] = list()
        # Set of disjointed_subgraphs
        self.subgraphs: List[Set[Node]] = list()

    @property
    def mean_outdegree(self):
        return sum([len(self.adjacency_list[node.name]) for node in self.nodes]) / len(self.nodes)

    @property
    def mean_weighted_outdegree(self):
        return sum([edge.log_weighted_frequency for edge in self.edges]) / len(self.nodes)

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

    def add_node(self, node: Node):
        """Add a node to the graph.

        :param node: The node to add.
        """
        if node in self.nodes:
            # Skip nodes that already exist
            return

        self.nodes.add(node)
        self.node_index[node.name] = node

    def add_edge(self, tail: Node, head: Node, edge_type: Type[Edge] = Edge) -> Optional[Edge]:
        """Add an edge between two nodes to the graph.

        :param tail: The node that the edge originates from.
        :param head: The node that the edge points to.
        :param edge_type: The type of edge to be created.
        :return: The edge instance, possibly None.
        """
        # Ignore spurious edges, a concept cannot be defined in terms of itself
        if tail == head:
            return None

        the_edge = edge_type(tail, head)

        if the_edge in self.edges:
            # Duplicate edges only increase a count so each edge is only
            # rendered once.
            the_edge = self.get_edge(the_edge.tail.name, the_edge.head.name)
            the_edge.frequency += 1

            return the_edge
        else:
            self.adjacency_list[tail.name].add(head)
            self.adjacency_index[head.name].add(tail)
            self.edges.add(the_edge)
            self.edge_index[(tail.name, head.name)] = the_edge

        return the_edge

    def remove_edge(self, edge: Edge):
        """Remove an edge from the graph.

        :param edge: The edge to remove
        """
        self.edges.discard(edge)

        try:
            del self.edge_index[(edge.tail.name, edge.head.name)]
        except KeyError:
            pass

        self.adjacency_list[edge.tail.name].discard(edge.head)
        self.adjacency_index[edge.head.name].discard(edge.tail)

    def get_edge(self, tail: str, head: str) -> Edge:
        """Get the edge that connects the nodes corresponding to `tail` and `head`.

        :param tail: The name of the node that the edge originates from.
        :param head: The name of the node that the edge points to.
        :return: The corresponding edge if it exists in the graph, None otherwise.
        """
        return self.edge_index[(tail, head)]

    def set_edge(self, edge: Edge) -> Edge:
        """Set (delete and add) an edge in the graph.

        The edge that is deleted will be the edge that has the same tail and
        head as the function argument `edge`.

        :param edge: The new edge that should replace the one in the graph.
        :return: The new edge.
        """
        the_edge = self.edge_index[(edge.tail.name, edge.head.name)]
        self.remove_edge(the_edge)

        return self.add_edge(edge.tail, edge.head, type(edge))

    def find_cycles(self) -> List[List[Node]]:
        """Find cycles in the graph."""
        G = nx.DiGraph()
        G.add_nodes_from([node.name for node in self.nodes])
        G.add_edges_from([(edge.tail.name, edge.head.name) for edge in self.edges])

        for cycle in nx.simple_cycles(G):
            self.cycles.append([self.node_index[node] for node in cycle])

        return self.cycles

    def find_subgraphs(self) -> List[Set[Node]]:
        """Find disjointed subgraphs in the graph.

        :return: A list of the subgraphs.
        """
        G = nx.Graph()
        G.add_nodes_from([node.name for node in self.nodes])
        G.add_edges_from([(edge.tail.name, edge.head.name) for edge in self.edges])

        self.subgraphs = [nodes for nodes in nx.connected_components(G)]

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
        print('Avg. Weighted Outdegree: %.2f' % self.mean_weighted_outdegree)

        if len(self.subgraphs) > 1:
            print(sep)
            print('Disjointed Subgraphs:', len(self.subgraphs))
            print('Avg. Disjointed Subgraph Size: %.2f' % (
                    sum(len(subgraph) for subgraph in self.subgraphs) / len(self.subgraphs)))

        if len(self.cycles) > 0:
            print(sep)
            print('Cycles:', len(self.cycles))
            print('Avg. Cycle Length: %.2f' % (sum([len(cycle) for cycle in self.cycles]) / len(self.cycles)))

        print(sep)

    def score(self) -> float:
        """Calculate a score of conceptual density for the given graph.

        :return: The score for the graph as a non-negative scalar.
        """
        n_cycles = len(self.cycles)
        avg_cycle_length = self.mean_cycle_length

        return self.mean_weighted_outdegree + n_cycles + avg_cycle_length

    def render(self):
        """Render the graph using GraphViz."""
        try:
            g = graphviz.Digraph(engine='neato')
            g.attr(overlap='false')

            for node in self.nodes:
                node.render(g)

            for edge in self.edges:
                edge.render(g)

            g.render(format='png', view=True)
        except graphviz.backend.ExecutableNotFound:
            print('Could not display graph -- GraphViz does not seem to be installed.')


class XMLSectionParser(Parser):
    @staticmethod
    def filter_spans(spans):
        # Filter a sequence of spans so they don't contain overlaps
        get_sort_key = lambda span: (span.end - span.start, span.start)
        sorted_spans = sorted(spans, key=get_sort_key, reverse=True)
        result = []
        seen_tokens = set()
        for span in sorted_spans:
            if span.start not in seen_tokens and span.end - 1 not in seen_tokens:
                result.append(span)
                seen_tokens.update(range(span.start, span.end))
        return result

    def parse(self, filename, graph, implicit_references=True):
        doc = self.load_doc(filename)
        self.chunk(doc)

        # TODO: Add sections back in? Also remove above Graph class and replace with ConceptGraph?
        for sent in doc.sents:
            tags = self.get_tagged(str(sent))
            tree = self.chunker.parse(tags)

            # Find the subject of the sentence
            subject = self.get_subject(sent)

            subject_node = Node(subject)
            graph.add_node(subject_node)

            # Add other noun phrases to the graph
            for np in tree.subtrees(lambda t: t.label() == 'NP'):
                tags = np.leaves()

                if tags[0][1] == 'DT':
                    tags = tags[1:]

                entity = ' '.join([token for token, tag in tags])

                try:
                    entity_node = graph.node_index[entity]
                except KeyError:
                    entity_node = Node(entity)
                    graph.add_node(entity_node)

                graph.add_edge(subject_node, entity_node)

                for implicit_entity, context in self.permutations(tags):
                    try:
                        implicit_entity_node = graph.node_index[implicit_entity]
                    except KeyError:
                        implicit_entity_node = Node(implicit_entity)
                        graph.add_node(implicit_entity_node)

                    graph.add_edge(graph.node_index[context], implicit_entity_node, ImplicitEdge)

    # TODO: Handle cases where no subject found (e.g. subordinate clauses).
    def get_subject(self, sent) -> str:
        subject = [w for w in sent.root.lefts if w.dep_.startswith('nsubj')]

        if subject:
            subject = subject[0]

            if subject.tag_ == 'DT':
                subject = ' '.join(str(subject).split()[1:])

            subject = ' '.join(map(self.lemmatizer.lemmatize, str(subject).split()))

        return str(subject)

    def load_doc(self, filename):
        if filename.endswith('xml'):
            tree = ET.parse(filename)
            root = tree.getroot()

            sentences = []

            for section in root.findall('section'):
                for text in section.findall('text'):
                    sentences += nltk.sent_tokenize(text.text)

            sentences = map(lambda s: s.strip(), sentences)
            sentences = filter(lambda s: len(s) > 0, sentences)
            sentences = list(sentences)
            text = ' '.join(sentences)
        else:
            with open(filename, 'r') as f:
                text = f.read()

        text = text.lower()
        nlp = spacy.load('en')
        neuralcoref.add_to_pipe(nlp)
        doc = nlp(text)
        # TODO: Fix loopy coreference resolution
        # doc = nlp(doc._.coref_resolved)

        return doc

    def chunk(self, doc):
        """Chunk the doc into noun chunks.

        :param doc: The document to chunk
        """
        spans = list(doc.noun_chunks)
        spans = XMLSectionParser.filter_spans(spans)
        with doc.retokenize() as retokenizer:
            for span in spans:
                retokenizer.merge(span)


# TODO: Fix bugs with single letter entities
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parse a file and group the entities by section.')
    parser.add_argument('-f', '--file', metavar='INPUT_FILE', type=str, required=True,
                        help='The file to parse. Can be a `.xml` file.')

    args = parser.parse_args()
    graph = Graph()
    xml_parser = XMLSectionParser()
    xml_parser.parse(args.file, graph)

    graph.find_cycles()
    graph.find_subgraphs()
    graph.print_summary()
    print('Score: %.2f' % graph.score())
    graph.render()
