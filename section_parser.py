import argparse
import xml.etree.ElementTree as ET
from collections import defaultdict
from math import log2

import graphviz
import nltk


class Node:
    def __eq__(self, other):
        return self.name == other.name

    def __init__(self, name: str, section_name: str):
        self.name = name
        self.section_name = section_name

    def __hash__(self):
        return hash(self.name)

    def __str__(self):
        return self.name

    def __repr__(self):
        return 'Node(%s, %s)' % (self.name, self.section_name)

    def render(self, g: graphviz.Digraph):
        if self.name == self.section_name:
            g.node(self.name, shape='doublecircle')
        else:
            g.node(self.name)


class Edge:
    def __init__(self, tail: Node, head: Node, weight=1):
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
    def log_weight(self):
        """A logarithmically scaled version of `weight`.

        This is used when rendering an edge so that the edges do not get
        arbitrarily thick as the weight gets large.
        """
        # Log weight:
        # > Start at one so that edges are always at least one unit thick
        # > Add one to log argument so that the return value is strictly
        # non-negative, weight=0 gives a return value of 0 and avoids
        # log(0) which is undefined.
        return 1 + log2(1 + self.weight * self.frequency)

    def render(self, g: graphviz.Digraph):
        g.edge(self.tail.name, self.head.name,
               penwidth=str(self.log_weight),
               color=self.color,
               style=self.style)


class ForwardEdge(Edge):
    def __init__(self, tail, head, weight=1.5):
        super().__init__(tail, head, weight)

        self.color = 'blue'


class BackwardEdge(Edge):
    def __init__(self, tail, head, weight=0.5):
        super().__init__(tail, head, weight)

        self.color = 'red'


class PartialEdge(Edge):
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

    def add_node(self, node: Node):
        """Add a node to the graph.

        :param node: The node to add.
        """
        # Handle the case where the node for the main concept of a section
        # was already added for a previous section
        if node.name == node.section_name and node in self.nodes:
            prev_section_name = self.node_index[node.name].section_name
            self.node_index[node.name].section_name = node.section_name
            self.section_nodes.add(node)

            if prev_section_name in self.section_listings:
                self.section_listings[prev_section_name].remove(node)

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

    def add_edge(self, tail, head, edge_type=Edge) -> Edge:
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

        if (edge.tail.name, edge.head.name) in self.edge_index:
            del self.edge_index[(edge.tail.name, edge.head.name)]

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
        visited = set()
        path = list()

        for section_name in self.sections:
            origin = self.node_index[section_name]
            self._colour_edges(origin, origin, visited, path)

    def _colour_edges(self, curr: Node, start: Node, visited: set, path: list):
        path.append(curr)

        if (curr.name in self.sections and curr != start) or curr.section_name != start.section_name:
            start_i = self.sections.index(self.section_index[start.name])
            curr_i = self.sections.index(self.section_index[curr.name])

            if start_i < curr_i:
                self._colour_path(path, ForwardEdge)
            elif start_i > curr_i:
                self._colour_path(path, BackwardEdge)
        elif curr not in visited:
            visited.add(curr)

            for child in self.adjacency_list[curr.name]:
                self._colour_edges(child, start, visited, path)

        path.pop()

    def _colour_path(self, path, edge_type=Edge):
        for i in range(1, len(path)):
            prev_node = path[i - 1]
            node = path[i]
            edge = self.get_edge(prev_node.name, node.name)

            new_edge = edge_type(edge.tail, edge.head)
            new_edge = self.set_edge(new_edge)
            new_edge.style = edge.style  # preserve edge style
            new_edge.frequency = edge.frequency

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

    # Parse the text and find the chunks - the sections, what the secion is about, the things that are mentioned in it -
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
                graph.add_edge(graph.node_index[entity_name], graph.node_index[nbar], edge_type=PartialEdge)

                nouns = [token for token, tag in st if tag.startswith('NN')]

                for i in range(len(nouns)):
                    register_entity(nouns[i], chunk, entities)
                    graph.add_node(Node(nouns[i], section_title))
                    graph.add_edge(graph.node_index[nbar], graph.node_index[nouns[i]], edge_type=PartialEdge)

                    substr = ' '.join(nouns[i:])
                    register_entity(substr, chunk, entities)
                    graph.add_node(Node(substr, section_title))
                    graph.add_edge(graph.node_index[nbar], graph.node_index[substr], edge_type=PartialEdge)

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

    # TODO: do the below except with `graph`.
    # # Display the graph representing the relationships of the entities in the text.
    # try:
    #     g = graphviz.Digraph(engine='circo')
    #
    #     # Add the nodes for the sections first
    #     with g.subgraph(name='cluster_main_entities') as sg:
    #         sg.attr('node', shape='doublecircle')
    #
    #         for chunk in chunks:
    #             sg.node(chunk.name)
    #
    #     # Add the nodes for the entities and all of the adjacency_list
    #     forward_backward_links = set()
    #
    #     for i, chunk in enumerate(chunks):
    #         print('Chunk: %s' % chunk)
    #
    #         # Check for forward and backward links between chunk[i] and chunk[j]
    #         for j in range(i + 1, len(chunks)):
    #             # Check for forward links
    #             for entity in chunk.entity_counts:
    #                 if entity in chunks[j].name:
    #                     print("Forward link from '%s' to '%s'" % (chunk.name, chunks[j].name))
    #                     g.edge(chunk.name, entity,
    #                            penwidth=str(chunk.log_count(entity)),
    #                            color='blue')
    #                     forward_backward_links.add((chunk.name, entity))
    #
    #             # Check for backward links
    #             for entity in chunks[j].entity_counts:
    #                 if entity in chunk.name:
    #                     print("Backward link from '%s' to '%s'" % (chunks[j].name, chunk.name))
    #                     g.edge(chunks[j].name, entity,
    #                            penwidth=str(chunks[j].log_count(entity)),
    #                            color='red')
    #                     forward_backward_links.add((chunks[j].name, entity))
    #
    #         # Check for self-contained and shared entities, also add links
    #         # that are neither forward or backward.
    #         for entity in chunk.entity_counts:
    #             if chunk.name in entities[entity]:
    #                 if len(entities[entity]) == 1:
    #                     print("Self-contained entity '%s' found in: '%s'." % (entity, chunk.name))
    #                 else:
    #                     print("Shared entity '%s' found in %s." % (entity, ', '.join(entities[entity])))
    #
    #             # Do check to avoid adding duplicate links
    #             if (chunk.name, entity) not in forward_backward_links:
    #                 g.edge(chunk.name, entity,
    #                        penwidth=str(chunk.log_count(entity)))
    #
    #         print()
    #
    #     g.render(format='png', view=True)
    # except graphviz.backend.ExecutableNotFound:
    #     print('Could not display graph -- GraphViz does not seem to be installed.')

    # Correct for inferred edges derived from forward references to main entities:
    # E.g. section on bread references (the main entity) wheat flour, which creates three nodes:
    # 'wheat flour', 'wheat', and 'flour'. Initially these are all assigned the
    # section 'bread', which is incorrect since 'wheat flour' is its own section
    # and any nodes derived from this should also be of the same section.
    for node in graph.section_nodes:
        for child in graph.adjacency_list[node.name]:
            edge = graph.get_edge(node.name, child.name)

            if isinstance(edge, PartialEdge):
                child.section_name = node.section_name

    # TODO: Reassign nodes to another section if the node appears more times in that section.

    graph.colour_edges()
    graph.render()
