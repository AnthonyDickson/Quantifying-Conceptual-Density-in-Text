import argparse
import xml.etree.ElementTree as ET

import graphviz


class Chunk:
    def __init__(self, name):
        self.name = name
        self.items = []

    def add(self, item):
        self.items.append(item)

    def __str__(self):
        return "'%s': %s" % (self.name, self.items)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parse a file and group the entities by section.')
    parser.add_argument('-f', '--file', metavar='INPUT_FILE', type=str, required=True,
                        help='The file to parse. Can be a `.xml` file.')

    args = parser.parse_args()

    tree = ET.parse(args.file)
    root = tree.getroot()

    chunks = []
    entities = dict()

    for section in root.findall('section'):
        section_title = section.find('title').text
        section_title = section_title.lower()

        chunk = Chunk(section_title)

        if chunk.name in entities:
            entities[chunk.name].add(chunk.name)
        else:
            entities[chunk.name] = {chunk.name}

        for entity in section.findall('entity'):
            entity_name = entity.text.lower()

            chunk.add(entity_name)

            if entity in entities:
                entities[entity_name].add(chunk.name)
            else:
                entities[entity_name] = {chunk.name}

        chunks.append(chunk)

        print(chunk)

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

    g = graphviz.Digraph()

    for chunk in chunks:
        g.node(chunk.name)

        for item in chunk.items:
            g.node(item)
            g.edge(chunk.name, item)

    g.render(view=True)
