import argparse
import xml.etree.ElementTree as ET

import graphviz


class Chunk:
    """A chunk is a section of text on a certain topic.
    It keeps track of what things are related to it.
    """
    def __init__(self, name):
        self.name = name
        self.items = set()

    def add(self, item):
        if item == self.name:
            return

        self.items.add(item)

    def __str__(self):
        return "'%s': %s" % (self.name, self.items)


def forms_cycle(chunk, chunks, visited):
    """Check if a cycle is formed between chunks.

    :param chunk: The chunk to start the search from.
    :param chunks: A dictionary that maps entities to the chunks they appear in.
    :param visited: The set of chunks that have been visited so far.
    :return: True if a cycle is found, False otherwise.
    """
    if chunk in visited:
        return True

    visited.add(chunk)

    for item in chunk.items:
        if item in chunks and forms_cycle(chunks[item], chunks, visited):
            return True

    return False


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

    for section in root.findall('section'):
        section_title = section.find('title').text
        section_title = section_title.lower()

        chunk = Chunk(section_title)
        chunks_dict[section_title] = chunk

        if chunk.name in entities:
            entities[chunk.name].add(chunk.name)
        else:
            entities[chunk.name] = {chunk.name}

        for entity in section.findall('entity'):
            entity_name = entity.text.lower()

            register_entity(entity_name, chunk, entities)

            # register permutations of a phrase.
            # E.g. 'wheat flour' gives the entities 'wheat', 'flour', and 'wheat flour'
            parts = entity_name.split(' ')

            for i in range(len(parts)):
                register_entity(parts[i], chunk, entities)

                for j in range(i + 1, len(parts)):
                    register_entity(' '.join(parts[i:j]), chunk, entities)

        chunks.append(chunk)

        print(chunk)

    # Analyse the chunks for forward links, backwards links, cyclic links, and self-contained entities.
    print('\nEntities and the sections they appear in:')
    print(entities)

    print('\nLink types...')

    for i, chunk in enumerate(chunks):
        print('Chunk: %s' % chunk)

        for item in chunk.items:
            if chunk.name in entities[item] and len(entities[item]) == 1:
                print("Self-contained entity '%s' found in '%s'." % (item, chunk.name))

        for j in range(i + 1, len(chunks)):
            for item in chunk.items:
                if item in chunks[j].name:
                    print("Forward link from '%s' to '%s'" % (chunk.name, chunks[j].name))

            for item in chunks[j].items:
                if item in chunk.name:
                    print("Backward link from '%s' to '%s'" % (chunks[j].name, chunk.name))

        print()

    has_cycle = forms_cycle(chunks[0], chunks_dict, set())

    if has_cycle:
        print('Found cycle in graph.')

    # Display the graph representing the relationships of the entities in the text.
    try:
        g = graphviz.Digraph()

        for chunk in chunks:
            g.node(chunk.name)

            for item in chunk.items:
                g.node(item)
                g.edge(chunk.name, item)

        g.render(view=True)
    except graphviz.backend.ExecutableNotFound:
        print('Could not display graph -- GraphViz does not seem to be installed.')
