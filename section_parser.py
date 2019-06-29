import argparse
import xml.etree.ElementTree as ET
from collections import defaultdict
from math import log2

import graphviz
import nltk


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
        return 'Node(%s, %s)' % (self.name, self.section_name)

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
    def log_frequency(self):
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
               penwidth=str(self.log_frequency),
               color=self.color,
               style=self.style)


class ForwardEdge(Edge):
    """An edge that references a node in a section that comes after the section
    that the tail node is in."""

    def __init__(self, tail, head, weight=1.5):
        super().__init__(tail, head, weight)

        self.color = 'blue'


class BackwardEdge(Edge):
    """An edge that references a node in a section that comes before the section
    that the tail node is in."""

    def __init__(self, tail, head, weight=0.5):
        super().__init__(tail, head, weight)

        self.color = 'red'


class ImplicitEdge(Edge):
    def __init__(self, tail, head, weight=0.75):
        super().__init__(tail, head, weight)
        self.style = 'dashed'


class Graph:
    def __init__(self):
        """Create an empty graph."""
        # The set of all nodes (vertices) in the graph.
        self.nodes = set()
        # Maps node name (concept) to node instance.
        self.node_index = dict()
        # Maps section to nodes found in that section.
        self.section_listings = defaultdict(set)
        # Maps node names to sections:
        self.section_index = defaultdict(lambda: None)
        # Set of sections nodes (the main concept of a given section).
        self.section_nodes = set()
        # List of sections used to record order that sections are introduced.
        self.sections = list()
        # Maps tail to head nodes
        self.adjacency_list = defaultdict(set)
        # Maps head to tail nodes
        self.adjacency_index = defaultdict(set)
        # The set of all edges in the graph
        self.edges = set()
        # Maps (tail, head) pairs to edge instance
        self.edge_index = defaultdict(lambda: None)

        # Set of self-contained references (edges0
        self.self_contained_references = set()
        # Set of shared entities (nodes)
        self.shared_entities = set()

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

    def add_edge(self, tail: Node, head: Node, edge_type=Edge) -> Edge:
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

    def colour_edges(self):
        """Colour edges as either forward or backward edges."""
        visited = set()
        path = list()

        for section_name in self.sections:
            origin = self.node_index[section_name]
            self._colour_edges(origin, origin, visited, path)

    def _colour_edges(self, curr: Node, start: Node, visited: set, path: list):
        """Recursively colour paths starting from a given node.

        :param curr: The next node to evaluate. This should be the same node as
                     start when first called.
        :param start: Where the path originates from.
        :param visited: The set of nodes that have already been visited.
        :param path: The sequence of nodes denoting the path that has be
                     traversed so far.
        """
        path.append(curr)

        # We have reached a 'leaf node' which is either:
        # - the main node of a section
        # OR
        # - a child node belonging to another section
        if (curr.name in self.sections and curr != start) or curr.section_name != start.section_name:
            # Check if the path goes forward from section to a later section, or vice versa
            start_i = self.sections.index(self.section_index[start.name])
            curr_i = self.sections.index(self.section_index[curr.name])

            if start_i < curr_i:
                self._colour_path(path, ForwardEdge)
            elif start_i > curr_i:
                self._colour_path(path, BackwardEdge)
        elif curr not in visited:
            # Otherwise we continue the depth-first traversal
            visited.add(curr)

            for child in self.adjacency_list[curr.name]:
                self._colour_edges(child, start, visited, path)

        path.pop()

    def _colour_path(self, path, edge_type=Edge):
        """Colour a path

        :param path:
        :param edge_type:
        :return:
        """
        for i in range(1, len(path)):
            prev_node = path[i - 1]
            node = path[i]
            edge = self.get_edge(prev_node.name, node.name)

            new_edge = edge_type(edge.tail, edge.head)
            new_edge = self.set_edge(new_edge)
            # preserve edge style for cases where there are implicit
            # forward/backward edges
            new_edge.style = edge.style
            new_edge.frequency = edge.frequency

            # Override default edge colour to preserve original colours for
            # direct references within a section.
            if node.section_name == path[0].section_name:
                new_edge.color = edge.color

    def render(self):
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


# TODO: Phase out code that uses Chunks
class Chunk:
    """A chunk is a section of text on a certain topic.
    It keeps track of what things are related to it.
    """

    def __init__(self, name):
        self.name = name
        self.entity_counts = dict()

    def log_count(self, entity):
        return 1 + log2(self.entity_counts[entity])

    def add(self, item):
        if item == self.name:
            return

        if item in self.entity_counts:
            self.entity_counts[item] += 1
        else:
            self.entity_counts[item] = 1

    def __str__(self):
        return "'%s': %s" % (self.name, list(self.entity_counts.keys()))


# TODO: Mark adjacency_list that form links.
# TODO: Return paths that form cycles.
# TODO: Return a list of cycle lengths.
def find_cycles(chunk, chunks, visited, marked, length=0):
    """Check if a cycle is formed between chunks.

    :param chunk: The chunk to start the search from.
    :param chunks: A dictionary that maps entities to the chunks they appear in.
    :param visited: The set of chunks that have been visited so far.
    :param length: The length of the current path.
    :return: True if a cycle is found, False otherwise.
    """
    if chunk in visited:
        print('Found cycle of length %d' % length)
        marked.add(chunk.name)
        return 1

    visited.add(chunk)

    cycles = 0

    for entity in chunk.entity_counts:
        if entity in chunks:
            n_cycles = find_cycles(chunks[entity], chunks, visited, marked, length + 1)

            if n_cycles > 0:
                marked.add(chunk.name)

            cycles += n_cycles

    return cycles


def register_entity(entity, chunk, entities):
    """Register an entity with the given chunk and entity-chunk index.

    :param entity: The entity to register.
    :param chunk: The chunk to register the entity with.
    :param entities: The index that maps entities to the chunk they appeared in.
    """
    chunk.add(entity)

    if entity in entities:
        entities[entity].add(chunk.name)
    else:
        entities[entity] = {chunk.name}


# TODO: Fix bugs with single letter entities
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parse a file and group the entities by section.')
    parser.add_argument('-f', '--file', metavar='INPUT_FILE', type=str, required=True,
                        help='The file to parse. Can be a `.xml` file.')

    args = parser.parse_args()

    tree = ET.parse(args.file)
    root = tree.getroot()

    # Parse the text and find the chunks - the sections, what the section is about, the things that are mentioned in it -
    # and the entities/things that appear in the text and which chunks they appear in.
    chunks = []
    chunks_dict = dict()
    entities = dict()
    grammar = r"""
                NBAR:
                    {<NN.*|JJ>*<NN.*>}  # Nouns and Adjectives, terminated with Nouns

                NP:
                    {<NBAR><IN><NBAR>}  # Above, connected with in/of/etc...
                    {<NBAR>}
            """
    chunker = nltk.RegexpParser(grammar)
    graph = Graph()

    for section in root.findall('section'):
        section_title = section.find('title').text
        section_title = section_title.lower()

        chunk = Chunk(section_title)
        chunks_dict[section_title] = chunk

        graph.add_node(Node(section_title, section_title))

        if chunk.name in entities:
            entities[chunk.name].add(chunk.name)
        else:
            entities[chunk.name] = {chunk.name}

        for entity in section.findall('entity'):
            entity_name = entity.text.lower()

            register_entity(entity_name, chunk, entities)
            graph.add_node(Node(entity_name, section_title))
            graph.add_edge(graph.node_index[section_title], graph.node_index[entity_name])

            # register permutations of a phrase.
            # E.g. 'wheat flour' gives the entities 'wheat', 'flour', and 'wheat flour'
            parts = entity_name.split(' ')

            phrase = nltk.word_tokenize(entity_name)
            tags = nltk.pos_tag(phrase)
            tree = chunker.parse(tags)

            for st in tree.subtrees(filter=lambda t: t.label() == 'NBAR'):
                nbar = ' '.join([str(token) for token, tag in st])
                register_entity(nbar, chunk, entities)

                graph.add_node(Node(nbar, section_title))
                graph.add_edge(graph.node_index[entity_name], graph.node_index[nbar], edge_type=ImplicitEdge)

                nouns = [token for token, tag in st if tag.startswith('NN')]

                for i in range(len(nouns)):
                    register_entity(nouns[i], chunk, entities)
                    graph.add_node(Node(nouns[i], section_title))
                    graph.add_edge(graph.node_index[nbar], graph.node_index[nouns[i]], edge_type=ImplicitEdge)

                    substr = ' '.join(nouns[i:])
                    register_entity(substr, chunk, entities)
                    graph.add_node(Node(substr, section_title))
                    graph.add_edge(graph.node_index[nbar], graph.node_index[substr], edge_type=ImplicitEdge)

        chunks.append(chunk)

        print(chunk)

    # Analyse the chunks for forward links, backwards links, cyclic links, and self-contained entities.
    print('\nEntities and the sections they appear in:')
    print(entities)

    print('\nLink types...')

    nodes_in_cycles = set()
    n_cycles = find_cycles(chunks[0], chunks_dict, set(), nodes_in_cycles)

    if n_cycles > 0:
        print('Found %d cycle(s) in the graph.' % n_cycles)
        print('These nodes are part of at least one cycle:', nodes_in_cycles)

    # Correct for inferred edges derived from forward references to main entities:
    # E.g. section on bread references (the main entity) wheat flour, which creates three nodes:
    # 'wheat flour', 'wheat', and 'flour'. Initially these are all assigned the
    # section 'bread', which is incorrect since 'wheat flour' is its own section
    # and any nodes derived from this should also be of the same section.
    for node in graph.section_nodes:
        for child in graph.adjacency_list[node.name]:
            edge = graph.get_edge(node.name, child.name)

            if isinstance(edge, ImplicitEdge):
                graph.section_listings[child.section_name].remove(child)
                graph.section_listings[node.section_name].add(child)
                graph.section_index[child.name] = node.section_name

                child.section_name = node.section_name

    # Reassign nodes to another section if the node appears more times in that section.
    for node in graph.section_nodes:
        for child in graph.adjacency_list[node.name]:
            # Select child nodes from other sections
            if child.section_name != node.section_name and child not in graph.section_nodes:
                edge = graph.get_edge(node.name, child.name)

                for child_neighbour in graph.adjacency_index[child.name]:
                    other_edge = graph.get_edge(child_neighbour.name, child.name)

                    # Compare the frequency of edges coming from different sections.
                    if child_neighbour.section_name != node.section_name and edge.frequency > other_edge.frequency:
                        child.section_name = node.section_name
                        break

    # Nodes that are only referenced from one section are 'self-contained references'
    # TODO: If a node is only referenced from one section but references nodes
    #  in other sections, can the edges pointing to that node still be
    #  considered 'self-contained references'?
    for section in graph.sections:
        for node in graph.section_listings[section]:
            referencing_sections = set()

            for tail in graph.adjacency_index[node.name]:
                referencing_sections.add(tail.section_name)

            if len(referencing_sections) == 1 and len(referencing_sections.intersection([section])) == 1:
                for tail in graph.adjacency_index[node.name]:
                    the_edge = graph.get_edge(tail.name, node.name)
                    the_edge.weight *= 0.5
                    graph.self_contained_references.add(the_edge)
            elif node.name != section:
                graph.shared_entities.add(node)

    graph.colour_edges()
    graph.render()
