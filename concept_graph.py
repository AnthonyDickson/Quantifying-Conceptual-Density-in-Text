from collections import defaultdict
from math import log2
from typing import Type, Set, Dict, Optional, List, Tuple
from xml.etree import ElementTree as ET

import graphviz
import networkx as nx
import nltk


class Parser:
    def __init__(self):
        """Create a parser."""
        self.grammar = r"""
            NBAR:
                # {<DT|NN.*><JJ|NN.*|IN|CC|DT>*<NN.*>}  # Nouns and Adjectives, terminated with Nouns
                # {<DT|JJ|NN.*|IN|CC>*<NN.*>}
                {<DT>?<NN.*|JJ>*<NN.*>}

            NP:
                {<NBAR>(<IN|CC><NBAR>)*}  # Above, connected with in/of/etc...
                {<NBAR>}
        """
        self.chunker = nltk.RegexpParser(self.grammar)
        self.lemmatizer = nltk.WordNetLemmatizer()

    def parse(self, filename: str, graph: 'ConceptGraph', implicit_references=True):
        """Parse a file and build up a graph structure.

        :param filename: The file to parse.
        :param graph: The graph instance to add the nodes and edges to.
        :param implicit_references: Whether or not to add implicit references to the graph.
        """
        raise NotImplementedError

    def get_tagged(self, phrase):
        """

        :param phrase:
        :return:
        """
        # TODO: Fill in docstring!
        phrase = phrase.lower()
        tags = nltk.pos_tag(nltk.word_tokenize(phrase))
        tags = [(self.lemmatizer.lemmatize(token), tag) for token, tag in tags]

        # Drop leading determiner
        if tags[0][1] == 'DT':
            return tags[1:]
        else:
            return tags

    def permutations(self, tagged_phrase: List[Tuple[str, str]]):
        """Generate variations of a POS (part of speech) tagged phrase.

        Variations generated are:
        - The entire phrase itself
        - nbar phrases (sequences of adjectives and/or nouns, terminated by a noun)
        - noun chunks (sequences of one or more nouns)

        Variations are yielded alongside a 'context', which represents the phrase that the variation was generated from.

        As an example, consider the sentence 'Zeus is the sky and thunder god in ancient Greek religion.' and the POS tagged
         phrase `[('Zeus', 'NNP'), ('is', 'VBZ'), ('the', 'DT'), ('sky', 'NN'), ('and', 'CC'), ('thunder', 'NN'),
         ('god', 'NN'), ('in', 'IN'), ('ancient', 'JJ'), ('Greek', 'JJ'), ('religion', 'NN'), ('.', '.')]`.
        The noun phrases we can expect are 'Zeus', 'sky and thunder god in ancient Greek religion'. For the second noun
        phrase we can expect the nbar phrases 'sky and thunder god' and 'ancient Greek religion'. These two nbar phrases
        would be yield with the noun phrase as the context.

        :param tagged_phrase: List of 2-tuples containing a POS tag and a token.
        :return: Yields 2-tuples containing a variation of `tagged_phrase` and the context it appears in.
        """
        context = ' '.join([token for token, tag in tagged_phrase])
        tree = self.chunker.parse(tagged_phrase)

        for st in tree.subtrees(filter=lambda t: t.label() == 'NBAR'):
            chunk = list(st)

            if chunk[0][1] == 'DT':
                chunk = chunk[1:]

            nbar = ' '.join([token for token, tag in chunk])
            yield nbar, context

            chunk = []

            for token, tag in st:
                if tag.startswith('NN') or tag in {'JJ', 'CC', 'IN'}:
                    chunk.append((token, tag))
                elif chunk:
                    yield from self._process_np_chunk(chunk, nbar)

                    chunk = []

            if chunk:
                yield from self._process_np_chunk(chunk, nbar)

    @staticmethod
    def _process_np_chunk(chunk, context):
        """Generate variations of a NP (noun phrase) chunk.

        :param chunk: List of 2-tuples containing a POS tag and a token.
        :param context: The parent phrase that the chunk originates from.
        """
        np = ' '.join([token for token, tag in chunk])
        yield np, context

        nbar_chunk = []

        for token, tag in chunk:
            if tag.startswith('NN') or tag in {'JJ'}:
                nbar_chunk.append((token, tag))
            elif nbar_chunk:
                yield from Parser.process_nbar_chunk(nbar_chunk, np)

                nbar_chunk = []

        if nbar_chunk:
            yield from Parser.process_nbar_chunk(nbar_chunk, np)

    @staticmethod
    def process_nbar_chunk(chunk, context):
        """Generate variations of a NBAR chunk.

        :param chunk: List of 2-tuples containing a POS tag and a token.
        :param context: The parent phrase that the chunk originates from.
        """
        nbar = ' '.join([token for token, tag in chunk])
        yield nbar, context

        noun_chunk = []

        for token, tag in chunk:
            if tag.startswith('NN'):
                noun_chunk.append((token, tag))
            elif noun_chunk:
                yield from Parser.process_noun_chunk(noun_chunk, nbar)

                noun_chunk = []

        if noun_chunk:
            yield from Parser.process_noun_chunk(noun_chunk, nbar)

    @staticmethod
    def process_noun_chunk(chunk, context):
        """Generate variations of a noun chunk.

        :param chunk: List of 2-tuples containing a POS tag and a token.
        :param context: The parent phrase that the chunk originates from.
        """
        if chunk[0][1] == 'DT':
            chunk = chunk[1:]

        noun_chunk = ' '.join([token for token, tag in chunk])
        yield noun_chunk, context

        for token, _ in chunk:
            yield token, noun_chunk

    def add_implicit_references(self, pos_tags, section, graph):
        """

        :param pos_tags:
        :param section:
        :param graph:
        """
        # TODO: Fill in above docstring!
        for implicit_entity, context in self.permutations(pos_tags):
            try:
                implicit_entity_node = graph.node_index[implicit_entity]
            except KeyError:
                implicit_entity_node = Node(implicit_entity, section)
                graph.add_node(implicit_entity_node)

            graph.add_edge(graph.node_index[context], implicit_entity_node, ImplicitEdge)


class XMLSectionParser(Parser):
    def parse(self, filename, graph, implicit_references=True):
        tree = ET.parse(filename)
        root = tree.getroot()

        for section in root.findall('section'):
            section_title = section.find('title').text
            section_title = section_title.lower()
            section_title = ' '.join(map(self.lemmatizer.lemmatize, nltk.word_tokenize(section_title)))

            section_node = Node(section_title, section_title)
            graph.add_node(section_node)

            for entity in section.findall('entity'):
                tags = self.get_tagged(entity.text)
                entity_name = ' '.join([token for token, _ in tags])

                try:
                    entity_node = graph.node_index[entity_name]
                except KeyError:
                    entity_node = Node(entity_name, section_title)
                    graph.add_node(entity_node)

                graph.add_edge(section_node, entity_node)

                if implicit_references:
                    self.add_implicit_references(tags, section_title, graph)


# TODO: Remove class and replace nodes with just strings
class Node:
    """A node/vertex in a graph."""

    def __eq__(self, other):
        return self.name == other.name

    def __init__(self, name: str, section_name: str):
        """Create a node representing a named entity that appears in a section
        of a given text.

        :param name: The name of the entity that the node represents.
        :param section_name: The name of the section that the entity appears in.
        """
        self.name = name
        self.section_name = section_name

    def __hash__(self):
        return hash(self.name)

    def __str__(self):
        return self.name

    def __repr__(self):
        return "Node('%s', '%s')" % (self.name, self.section_name)

    def render(self, g: graphviz.Digraph):
        """Render/draw the node using GraphViz.

        :param g: The graphviz graph instance.
        """
        if self.name == self.section_name:
            g.node(self.name, shape='doublecircle')
        else:
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


class ConceptGraph:
    def __init__(self, parser_type: Type[Parser] = XMLSectionParser, implicit_references=True, mark_references=True):
        """Create an empty graph.

        :param parser_type: The type of parser to use.
        :param implicit_references: Whether or not to include implicit references during parsing.
        :param mark_references: Whether or not to mark forward and backward references after parsing.
        """
        # The set of all nodes (vertices) in the graph.
        self.nodes: Set[Node] = set()
        # Maps node name (concept) to node instance.
        self.node_index: Dict[str, Node] = dict()

        # Maps section to nodes found in that section.
        self.section_listings: Dict[str, Set[Node]] = defaultdict(set)
        # Maps node names to sections:
        self.section_index: Dict[str, Optional[str]] = defaultdict(lambda: None)
        # Set of sections nodes (the main concept of a given section).
        self.section_nodes: Set[Node] = set()
        # List of sections used to record order that sections are introduced.
        self.sections: List[str] = list()

        # Maps tail to head nodes
        self.adjacency_list: Dict[str, Set[Node]] = defaultdict(set)
        # Maps head to tail nodes
        self.adjacency_index: Dict[str: Set[Node]] = defaultdict(set)
        # The set of all edges in the graph
        self.edges: Set[Edge] = set()
        # Maps (tail, head) pairs to edge instance
        self.edge_index: Dict[Tuple[str, str], Optional[Edge]] = defaultdict(lambda: None)

        # Set of forward references
        self.forward_references: Set[Edge] = set()
        # Set of backward references
        self.backward_references: Set[Edge] = set()
        # Set of self-contained references (edges)
        self.self_contained_references: Set[Edge] = set()
        # Set of shared entities (nodes)
        self.shared_entities: Set[Node] = set()
        # Set of self-referential loops
        self.cycles: List[List[Node]] = list()
        # Set of disjointed_subgraphs
        self.subgraphs: List[Set[Node]] = list()

        ## Parse Options ##
        self.parser: Parser = parser_type()
        self.implicit_references: bool = implicit_references
        self.mark_references: bool = mark_references

        ## Misc ##
        self.nx: Optional[nx.DiGraph] = None

    @property
    def mean_outdegree(self):
        return sum([len(self.adjacency_list[node.name]) for node in self.nodes]) / len(self.nodes)

    @property
    def mean_section_outdegree(self):
        avg_degree = 0

        for section in self.sections:
            avg_degree += sum(
                [len(self.adjacency_list[node.name]) for node in self.section_listings[section]]) / len(
                self.section_listings[section])

        avg_degree /= len(self.sections)

        return avg_degree

    @property
    def mean_weighted_outdegree(self):
        return sum([edge.log_weighted_frequency for edge in self.edges]) / len(self.nodes)

    @property
    def mean_weighted_section_outdegree(self):
        avg_degree = 0

        for section in self.sections:
            section_degree = 0

            for tail in self.section_listings[section]:
                for head in self.adjacency_list[tail.name]:
                    section_degree += self.get_edge(tail.name, head.name).log_weighted_frequency

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

    def add_node(self, node: Node):
        """Add a node to the graph.

        :param node: The node to add.
        """
        # Handle the case where the node for the main concept of a section
        # was already added for a previous section
        if node.name == node.section_name and node in self.nodes:
            # Remove or update the section details
            prev_section_name = self.node_index[node.name].section_name
            self.node_index[node.name].section_name = node.section_name

            try:
                self.section_listings[prev_section_name].remove(node)
            except ValueError:
                pass

            # Add new section details.
            self.section_nodes.add(node)
            self.section_listings[node.section_name].add(node)
            self.section_index[node.name] = node.section_name

            if node.section_name not in self.sections:
                self.sections.append(node.section_name)

            return
        elif node in self.nodes:
            # Skip nodes that already exist
            return

        self.nodes.add(node)
        self.node_index[node.name] = node
        self.section_listings[node.section_name].add(node)
        self.section_index[node.name] = node.section_name

        if node.section_name not in self.sections:
            self.sections.append(node.section_name)

        if node.name == node.section_name:
            self.section_nodes.add(node)

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

    def parse(self, filename):
        """Parse a XML document and build up a graph structure.

        :param filename: The XML file to parse.
        """
        self.parser.parse(filename, self, self.implicit_references)
        self.postprocessing()

    def postprocessing(self):
        """Perform tasks to fix/normalise the graph structure"""
        self._reassign_implicit_entities()
        self._reassign_sections()
        self._categorise_nodes()

        if self.mark_references:
            self.mark_edges()

        self.nx = self.to_nx()
        self.find_cycles()
        self.find_subgraphs()

    def to_nx(self):
        """Convert the graph to a NetworkX graph.

        :return: The graph as an NetworkX graph.
        """
        G = nx.DiGraph()
        G.add_nodes_from([node.name for node in self.nodes])
        G.add_edges_from([(edge.tail.name, edge.head.name) for edge in self.edges])

        return G

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
            for child in self.adjacency_list[node.name]:
                edge = self.get_edge(node.name, child.name)

                if isinstance(edge, ImplicitEdge):
                    self.section_listings[child.section_name].remove(child)
                    self.section_listings[node.section_name].add(child)
                    self.section_index[child.name] = node.section_name

                    child.section_name = node.section_name

    def _reassign_sections(self):
        """Reassign nodes to another section if the node appears more times in that section."""
        for node in self.section_nodes:
            for child in self.adjacency_list[node.name]:
                # Select child nodes from other sections
                if child.section_name != node.section_name and child not in self.section_nodes:
                    edge = self.get_edge(node.name, child.name)

                    for child_neighbour in self.adjacency_index[child.name]:
                        other_edge = self.get_edge(child_neighbour.name, child.name)

                        # Compare the frequency of edges coming from different sections.
                        if child_neighbour.section_name != node.section_name and edge.frequency > other_edge.frequency:
                            child.section_name = node.section_name
                            break

    def _categorise_nodes(self):
        """Categorise nodes in the graph into 'self-contained' and 'shared' entities.

        Nodes that are only referenced from one section represent 'self-contained references',
        all other nodes represent 'shared entities'.
        """
        # TODO: Change self_contained_references to external entities (edges to nodes)?
        for section in self.sections:
            for node in self.section_listings[section]:
                referencing_sections = set()

                for tail in self.adjacency_index[node.name]:
                    referencing_sections.add(tail.section_name)

                if len(referencing_sections) == 1 and len(referencing_sections.intersection([section])) == 1:
                    for tail in self.adjacency_index[node.name]:
                        the_edge = self.get_edge(tail.name, node.name)
                        the_edge.weight *= 0.5
                        self.self_contained_references.add(the_edge)
                elif node.name != section:
                    self.shared_entities.add(node)

    def mark_edges(self):
        """Colour edges as either forward or backward edges."""
        for node in self.section_nodes:
            self._mark_edges(node, None, set())

    def _mark_edges(self, curr: Node, prev: Optional[Node], visited: set):
        """Recursively mark edges.

        :param curr: The next node to vist.
        :param prev: The node that was previously visited.
        :param visited: The set of nodes that have already been visited.
        """
        # We have reached a 'leaf node' which is a node belonging to another section
        if prev and curr.section_name != prev.section_name:
            # Check if the path goes forward from section to a later section, or vice versa
            curr_i = self.sections.index(curr.section_name)
            prev_i = self.sections.index(prev.section_name)

            if curr_i < prev_i:
                self._mark_edge(prev, curr, BackwardEdge)
            elif curr_i > prev_i:
                self._mark_edge(prev, curr, ForwardEdge)
        elif curr not in visited:
            # Otherwise we continue the depth-first traversal
            visited.add(curr)

            for child in self.adjacency_list[curr.name]:
                self._mark_edges(child, curr, visited)

    def _mark_edge(self, tail: Node, head: Node, edge_type: Type[Edge] = Edge):
        """Mark an edge.

        :param tail: The node from which the edge originates from, or points
                     aways from.
        :param head: The node that the edge points towards.
        :param edge_type: The type to mark the edge as.
        """
        edge = self.get_edge(tail.name, head.name)

        new_edge = edge_type(tail, head)
        new_edge = self.set_edge(new_edge)
        # preserve edge style for cases where there are implicit
        # forward/backward edges
        new_edge.style = edge.style
        new_edge.frequency = edge.frequency

        if isinstance(new_edge, ForwardEdge):
            self.forward_references.add(new_edge)
        elif isinstance(new_edge, BackwardEdge):
            self.backward_references.add(new_edge)

    def find_cycles(self) -> List[List[Node]]:
        """Find cycles in the graph."""
        self.cycles = list()

        for cycle in nx.simple_cycles(self.nx):
            self.cycles.append([self.node_index[node] for node in cycle])

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
        print('Self-contained References:', len(self.self_contained_references))
        print('Shared Entities:', len(self.shared_entities))

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
