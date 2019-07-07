import argparse
import xml.etree.ElementTree as ET

import spacy

from concept_graph import Parser, ConceptGraph


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
        nlp = spacy.load('en')

        tree = ET.parse(filename)
        root = tree.getroot()

        for section in root.findall('section'):
            section_title = section.find('title').text
            section_title = section_title.lower()

            section_text = section.find('text').text
            section_text = section_text.lower()

            span = nlp(section_text)
            self.chunk(span)

            for sent in span.sents:
                tags = self.get_tagged(str(sent))
                tree = self.chunker.parse(tags)

                # Find the subject of the sentence
                subject = self.get_subject(sent)

                graph.add_node(subject, section_title)

                # Add other noun phrases to the graph
                for np in tree.subtrees(lambda t: t.label() == 'NP'):
                    tags = np.leaves()

                    if tags[0][1] == 'DT':
                        tags = tags[1:]

                    entity = ' '.join([token for token, tag in tags])

                    if entity not in graph.nodes:
                        graph.add_node(entity, section_title)
                    else:
                        graph.update_section_count(entity, section_title)

                    graph.add_edge(subject, entity)

                    self.add_implicit_references(tags, section_title, graph)

    # TODO: Handle cases where no subject found (e.g. subordinate clauses).
    # TODO: Handle subjects that have more than one actor (e.g. two things joined by 'and').
    def get_subject(self, sent) -> str:
        subject = [w for w in sent.root.lefts if w.dep_.startswith('nsubj')]

        if subject:
            subject = subject[0]

            if subject.tag_ == 'DT':
                subject = ' '.join(str(subject).split()[1:])

            subject = ' '.join(map(self.lemmatizer.lemmatize, str(subject).split()))

        return str(subject)

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
    graph = ConceptGraph(parser=XMLSectionParser())
    graph.parse(args.file)

    graph.print_summary()
    print('Score: %.2f' % graph.score())
    graph.render()
