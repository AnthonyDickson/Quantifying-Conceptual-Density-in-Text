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

                self.add_gerund_phrase(subject, section_title, sent, graph)

                # Add other noun phrases to the graph
                for np in tree.subtrees(lambda t: t.label() == 'NP'):
                    tags = np.leaves()

                    if tags[0][1] == 'DT':
                        tags = tags[1:]

                    entity = ' '.join([token for token, tag in tags])

                    graph.add_node(entity, section_title)
                    graph.add_edge(subject, entity)

                    if implicit_references:
                        self.add_implicit_references(tags, section_title, graph)

    def add_gerund_phrase(self, subject: str, section: str, sentence: spacy.tokens.span.Span, graph: ConceptGraph):
        """Add gerund (verb) phrases to the graph.

        For gerunds without an object, just the verb is added to the graph.
        For all other gerunds, the object is added to the graph and the edge between the subject and object is annotated
        with the S-form of the gerund. Form example, 'Tom likes cake.' yields the nodes 'Tom' and 'cake' connected by
        an edge annotated with 'likes'.

        :param subject: The subject of the sentence the gerund was found in.
        :param section: The section the gerund (and its sentence) was found in.
        :param sentence: The sentence the gerund was found in.
        :param graph: The graph to add the gerund phrase to.
        """
        for gerund in filter(lambda token: token.tag_ == 'VBG', sentence):
            verb = str(gerund)

            for right in gerund.rights:
                if 'obj' in right.dep_:
                    break
            else:
                graph.add_node(verb, section)
                graph.add_edge(subject, verb)

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parse a file and create a graph structure, '
                                                 'grouping concepts by section.')
    args = parser.parse_args()

    graph = ConceptGraph(parser=XMLSectionParser(),
                         implicit_references=False,
                         mark_references=False)
    graph.parse('bread-sections_only.xml')
    graph.render('bread_graph-sections_only-simple', view=False)

    graph = ConceptGraph(parser=XMLSectionParser(),
                         implicit_references=False,
                         mark_references=True)
    graph.parse('bread-sections_only.xml')
    graph.render('bread_graph-sections_only-reference_marking', view=False)

    graph = ConceptGraph(parser=XMLSectionParser(),
                         implicit_references=True,
                         mark_references=False)
    graph.parse('bread-sections_only.xml')
    graph.render('bread_graph-sections_only-implicit_references', view=False)

    graph = ConceptGraph(parser=XMLSectionParser(),
                         implicit_references=True,
                         mark_references=True)
    graph.parse('bread-sections_only.xml')
    graph.render('bread_graph-sections_only', view=False)
