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
                {<DT>?<NN.*|JJ>*<NN.*>} # Nouns and Adjectives, terminated with Nouns

            NP:
                {<NBAR>(<IN|CC><NBAR>)*}  # Above, connected with in/of/etc...
        """
        self.chunker = nltk.RegexpParser(self.grammar)
        self.lemmatizer = nltk.WordNetLemmatizer()

    def parse(self, filename: str, graph: 'ConceptGraph', implicit_references: bool = True):
        """Parse a file and build up a graph structure.

        :param filename: The file to parse.
        :param graph: The graph instance to add the nodes and edges to.
        :param implicit_references: Whether or not to add implicit references to the graph.
        """
        raise NotImplementedError

    def get_tagged(self, phrase: str) -> List[Tuple[str, str]]:
        """Normalise and tag a string.

        :param phrase: The string to process.
        :return: List of token, tag pairs.
        """
        phrase = phrase.lower()
        tags = nltk.pos_tag(nltk.word_tokenize(phrase))
        tags = [(self.lemmatizer.lemmatize(token), tag) for token, tag in tags]

        # Drop leading determiner
        if tags[0][1] == 'DT':
            return tags[1:]
        else:
            return tags

    def permutations(self, tagged_phrase: List[Tuple[str, str]]) -> Tuple[str, str]:
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
    def _process_np_chunk(chunk: List[Tuple[str, str]], context: str) -> Tuple[str, str]:
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
    def process_nbar_chunk(chunk: List[Tuple[str, str]], context: str) -> Tuple[str, str]:
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
            elif tag == 'JJ':
                yield token, nbar
            elif noun_chunk:
                yield from Parser.process_noun_chunk(noun_chunk, nbar)

                noun_chunk = []

        if noun_chunk:
            yield from Parser.process_noun_chunk(noun_chunk, nbar)

    @staticmethod
    def process_noun_chunk(chunk: List[Tuple[str, str]], context: str) -> Tuple[str, str]:
        """Generate variations of a noun chunk.

        :param chunk: List of 2-tuples containing a POS tag and a token.
        :param context: The parent phrase that the chunk originates from.
        """
        noun_chunk = ' '.join([token for token, tag in chunk])
        yield noun_chunk, context

        for token, _ in chunk:
            yield token, noun_chunk

    def add_implicit_references(self, pos_tags: List[Tuple[str, str]], section: str, graph: 'ConceptGraph'):
        """Derive nodes and edges from a POS tagged phrase.

        See `permutations()` for details on what kind of nodes and edges are derived.

        :param pos_tags: A phrase as a list of token, tag pairs.
        :param section: The section that the phrase appears in.
        :param graph: The graph to add the derived nodes and edges to.
        """
        for implicit_entity, context in self.permutations(pos_tags):
            graph.add_node(implicit_entity, section)
            graph.add_edge(context, implicit_entity, ImplicitEdge)


class XMLSectionParser(Parser):
    """Parser for XML documents.

    Expects XML documents to have section tags containing a title tag and entity tags around the concepts.
    """

    def parse(self, filename: str, graph: 'ConceptGraph', implicit_references=True):
        tree = ET.parse(filename)
        root = tree.getroot()

        for section in root.findall('section'):
            section_title = section.find('title').text
            section_title = section_title.lower()
            section_title = ' '.join(map(self.lemmatizer.lemmatize, nltk.word_tokenize(section_title)))

            graph.add_node(section_title, section_title)

            for concept_element in section.findall('concept'):
                tags = self.get_tagged(concept_element.text)
                concept = ' '.join([token for token, _ in tags])

                graph.add_node(concept, section_title)
                graph.add_edge(section_title, concept)

                if implicit_references:
                    self.add_implicit_references(tags, section_title, graph)


Node = str


class Edge:
    """A connection between two nodes in a graph."""

    def __init__(self, tail: Node, head: Node, weight: float = 1):
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

        self.colour = 'black'
        self.style = 'solid'
        self.label = ''

    def __eq__(self, other):
        return self.tail == other.tail and self.head == other.head

    def __hash__(self):
        return hash(self.tail + self.head)

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
        g.edge(self.tail, self.head,
               penwidth=str(self.log_weighted_frequency),
               color=colour if colour else self.colour,
               style=self.style,
               label=self.label)


class ForwardEdge(Edge):
    """An edge that references a node in a section that comes after the section
    that the tail node is in."""

    def __init__(self, tail: Node, head: Node, weight: float = 2.0):
        super().__init__(tail, head, weight)

        self.colour = 'blue'


class BackwardEdge(Edge):
    """An edge that references a node in a section that comes before the section
    that the tail node is in."""

    def __init__(self, tail: Node, head: Node, weight: float = 1.5):
        super().__init__(tail, head, weight)

        self.colour = 'red'


class ImplicitEdge(Edge):
    def __init__(self, tail: Node, head: Node, weight: float = 0.5):
        super().__init__(tail, head, weight)

        self.style = 'dashed'


class ConceptGraph:
    # TODO: Add 'dirty' flag to indicate the graph has been changed since postprocessing() was last called.
    def __init__(self, parser: Parser = None, implicit_references=True, mark_references=True):
        """Create an empty graph.

        :param parser_type: The type of parser to use.
        :param implicit_references: Whether or not to include implicit references during parsing.
        :param mark_references: Whether or not to mark forward and backward references after parsing.
        """
        # The set of all nodes (vertices) in the graph.
        self.nodes: Set[Node] = set()

        # Maps section to nodes found in that section.
        self.section_listings: Dict[str, Set[Node]] = defaultdict(set)
        # Maps node names to sections:
        self.section_index: Dict[str, Optional[str]] = defaultdict(lambda: None)
        # Set of sections nodes (the main concept of a given section).
        self.section_nodes: Set[Node] = set()
        # List of sections used to record order that sections are introduced.
        self.sections: List[str] = list()
        # Maps nodes to the frequency they appear in each section.
        self.section_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

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
        # Set of external references (edges)
        self.external_entities: Set[Edge] = set()
        # Set of shared entities (nodes)
        self.shared_entities: Set[Node] = set()
        # Set of self-referential loops
        self.cycles: List[List[Node]] = list()
        # Set of disjointed_subgraphs
        self.subgraphs: List[Set[Node]] = list()

        ## Parse Options ##
        self.parser: Parser = parser if parser else XMLSectionParser()
        self.implicit_references: bool = implicit_references
        self.mark_references: bool = mark_references

        ## Misc ##
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
        return sum([edge.log_weighted_frequency for edge in self.edges]) / len(self.nodes)

    @property
    def mean_weighted_section_outdegree(self):
        avg_degree = 0

        for section in self.sections:
            section_degree = 0

            for tail in self.section_listings[section]:
                for head in self.adjacency_list[tail]:
                    section_degree += self.get_edge(tail, head).log_weighted_frequency

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

    def add_node(self, node: Node, section: str):
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

    def add_edge(self, tail: str, head: str, edge_type: Type[Edge] = Edge) -> Optional[Edge]:
        """Add an edge between two nodes to the graph.

        :param tail: The node that the edge originates from.
        :param head: The node that the edge points to.
        :param edge_type: The type of edge to be created.
        :return: The edge instance, possibly None.
        """
        # Ignore spurious edges, a concept cannot be defined in terms of itself
        if tail == head:
            return None

        assert tail in self.nodes and head in self.nodes, 'Both nodes in the edge must exist within the graph.'

        the_edge = edge_type(tail, head)

        if the_edge in self.edges:
            # Duplicate edges only increase a count so each edge is only
            # rendered once.
            the_edge = self.get_edge(the_edge.tail, the_edge.head)
            the_edge.frequency += 1

            return the_edge
        else:
            self.adjacency_list[tail].add(head)
            self.adjacency_index[head].add(tail)
            self.edges.add(the_edge)
            self.edge_index[(tail, head)] = the_edge

        return the_edge

    def remove_edge(self, tail: Node, head: Node):
        """Remove an edge from the graph.

        :param tail: The tail node of the edge to remove.
        :param head: The head node of the edge to remove.
        """
        try:
            edge = self.edge_index[(tail, head)]
            self.edges.discard(edge)
            del self.edge_index[(tail, head)]

            self.adjacency_list[tail].discard(head)
            self.adjacency_index[head].discard(tail)
        except KeyError:
            # non-existent edge, probably
            pass

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
        self.remove_edge(edge.tail, edge.head)

        return self.add_edge(edge.tail, edge.head, type(edge))

    def update_section_count(self, node: str, section: str):
        """Update the count of times a node appears in a given section by one.

        :param node: The node.
        :param section: The section the node was found in.
        """
        self.section_counts[node][section] += 1

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
        self.nx = self.to_nx()

        self.mark_edges()
        self.find_cycles()
        self.find_subgraphs()

    def to_nx(self):
        """Convert the graph to a NetworkX graph.

        :return: The graph as an NetworkX graph.
        """
        G = nx.DiGraph()
        G.add_nodes_from([node for node in self.nodes])
        G.add_edges_from([(edge.tail, edge.head) for edge in self.edges])

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
            for child in self.adjacency_list[node]:
                edge = self.get_edge(node, child)

                if isinstance(edge, ImplicitEdge):
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
        """Categorise nodes in the graph into 'self-contained' and 'shared' entities.

        Nodes that are only referenced from one section represent 'self-contained references',
        all other nodes represent 'shared entities'.
        """
        # TODO: Change self_contained_references to external entities (edges to nodes)?
        for section in self.sections:
            for node in self.section_listings[section]:
                referencing_sections = set()

                for tail in self.adjacency_index[node]:
                    tail_section = self.section_index[tail]
                    referencing_sections.add(tail_section)

                if len(referencing_sections) == 1 and len(referencing_sections.intersection([section])) == 1:
                    for tail in self.adjacency_index[node]:
                        the_edge = self.get_edge(tail, node)
                        the_edge.weight *= 0.5
                        self.external_entities.add(the_edge)
                elif node != section:
                    self.shared_entities.add(node)

    def mark_edges(self):
        """Colour edges as either forward or backward edges."""
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
                self._mark_edge(prev, curr, BackwardEdge)
            elif curr_i > prev_i:
                self._mark_edge(prev, curr, ForwardEdge)
        elif curr not in visited:
            # Otherwise we continue the depth-first traversal
            visited.add(curr)

            for child in self.adjacency_list[curr]:
                self._mark_edges(child, curr, visited)

    def _mark_edge(self, tail: Node, head: Node, edge_type: Type[Edge] = Edge):
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

        if isinstance(new_edge, ForwardEdge):
            self.forward_references.add(new_edge)
        elif isinstance(new_edge, BackwardEdge):
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
        print('Self-contained References:', len(self.external_entities))
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
