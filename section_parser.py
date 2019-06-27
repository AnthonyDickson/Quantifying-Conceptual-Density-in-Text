import argparse
import xml.etree.ElementTree as ET
from math import log2

import graphviz
import nltk


class Graph:
    def __init__(self):
        self.edges = set()
        self.nodes = set()
        self.node_index = dict()
        self.section_index = dict()
        self.section_nodes = set()
        self.adjacency_list = dict()
        self.adjacency_index = dict()

    def add_node(self, node):
        if node.name == node.section_name and node in self.nodes:
            prev_section_name = self.node_index[node.name].section_name
            self.node_index[node.name].section_name = node.section_name
            self.section_nodes.add(node)

            if prev_section_name in self.section_index:
                self.section_index[prev_section_name].remove(node)

            if node.section_name in self.section_index:
                self.section_index[node.section_name].add(node)
            else:
                self.section_index[node.section_name] = {node}

            return
        elif node in self.nodes:
            return

        self.nodes.add(node)
        self.node_index[node.name] = node

        if node.section_name in self.section_index:
            self.section_index[node.section_name].add(node)
        else:
            self.section_index[node.section_name] = {node}

        if node.name == node.section_name:
            self.section_nodes.add(node)

    def add_edge(self, tail, head):
        if tail == head:
            return

        if tail in self.adjacency_list:
            self.adjacency_list[tail].add(head)
            self.adjacency_index[head].add(tail)
        else:
            self.adjacency_list[tail] = {head}
            self.adjacency_index[head] = {tail}

        self.edges.add(Edge(tail, head))

    def render(self):
        try:
            g = graphviz.Digraph(engine='circo')

            with g.subgraph(name='cluster_main_entities') as sg:
                sg.attr('node', shape='doublecircle')

                for section_node in self.section_nodes:
                    sg.node(section_node.name)

            for node in self.nodes:
                g.node(node.name)

            for edge in self.edges:
                edge.render(g)

            g.render(format='png', view=True)
        except graphviz.backend.ExecutableNotFound:
            print('Could not display graph -- GraphViz does not seem to be installed.')


class Node:
    def __init__(self, name: str, section_name: str):
        self.name = name
        self.section_name = section_name

    def __eq__(self, other):
        return self.name == other.name

    def __hash__(self):
        return hash(self.name)

    def __str__(self):
        return self.name

    def __repr__(self):
        return 'Node(%s, %s)' % (self.name, self.section_name)


class Edge:
    def __init__(self, tail: Node, head: Node, weight=1):
        self.tail = tail
        self.head = head
        self.weight = weight
        self.color = 'black'
        self.style = 'solid'

    def __eq__(self, other):
        return self.tail == other.tail and self.head == other.head

    def __hash__(self):
        return hash(self.tail.name + self.head.name)

    def render(self, g: graphviz.Digraph):
        g.edge(self.tail.name, self.head.name,
               penwidth=str(1 + log2(self.weight)),
               color=self.color,
               style=self.style)


class ForwardEdge(Edge):
    def __init__(self, tail, head, weight=1):
        super().__init__(tail, head, weight)

        self.color = 'blue'


class BackwardEdge(Edge):
    def __init__(self, tail, head, weight=1):
        super().__init__(tail, head, weight)

        self.color = 'red'


class PartialEdge(Edge):
    def __init__(self, tail, head, weight=1):
        super().__init__(tail, head, weight)

        self.style = 'dashed'


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
                graph.add_edge(graph.node_index[entity_name], graph.node_index[nbar])

                nouns = [token for token, tag in st if tag.startswith('NN')]

                for i in range(len(nouns)):
                    register_entity(nouns[i], chunk, entities)
                    graph.add_node(Node(nouns[i], section_title))
                    graph.add_edge(graph.node_index[nbar], graph.node_index[nouns[i]])

                    substr = ' '.join(nouns[i:])
                    register_entity(substr, chunk, entities)
                    graph.add_node(Node(substr, section_title))
                    graph.add_edge(graph.node_index[nbar], graph.node_index[substr])

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

    graph.render()
