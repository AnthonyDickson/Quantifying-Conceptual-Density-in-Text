import sys
import xml.etree.ElementTree as ET

import neuralcoref
import plac
import spacy

from qcd.concept_graph import Parser, ConceptGraph


class XMLSectionParser(Parser):
    """Parser for XML documents.

    Expects XML documents to have section tags containing a 'title' tag and a 'text' tag around the text.
    """

    def __init__(self, annotate_edges=True, resolve_coreferences=False):
        """Create a parser for XML documents.

        :param annotate_edges: Whether or not to annotate edges with a relationship type.
        :param resolve_coreferences: Whether or not to resolve coreferences.
        """

        super().__init__()

        self.resolve_coreferences = resolve_coreferences
        self.annotate_edges = annotate_edges

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
        try:
            tree = ET.parse(filename)
        except ET.ParseError as e:
            print('Could not parse the file. \n%s.' % e.msg.capitalize(), file=sys.stderr)
            exit(1)
        except FileNotFoundError as e:
            print('Could not open the file. \n%s' % e)
            exit(2)

        root = tree.getroot()

        if self.resolve_coreferences:
            nlp_ = spacy.load('en')
            neuralcoref.add_to_pipe(nlp_)
            nlp = lambda text: nlp_(nlp_(text)._.coref_resolved)
        else:
            nlp_ = spacy.load('en')
            nlp = lambda text: nlp_(text)

        for section in root.findall('section'):
            section_title = section.find('title').text
            section_title = section_title.lower()

            section_text = section.find('text').text
            section_text = section_text.lower()

            span = nlp(section_text)
            self.chunk(span)

            for sent in span.sents:
                # TODO: Use spacy tags instead, more accurate.
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

            # TODO: Add edge between gerund and object
            # TODO: Remove redundant edge between subject and object since that relation is represented
            #  through the path subject -> verb -> object.
            # TODO: Refactor verbal phrase stuff such that we instead have
            #  subject -- verb (S-form) --> object
            #  including is_a and has_a relations.
            for right in gerund.rights:
                if 'obj' in right.dep_:
                    if self.annotate_edges:
                        object_ = str(right)
                        graph.add_node(object_, section)

                        the_edge = graph.add_edge(subject, object_)
                        the_edge.label = gerund.lemma_

                        if the_edge.label.endswith(('s', 'sh', 'ch')):
                            the_edge.label += 'es'
                        elif the_edge.label.endswith('y'):
                            the_edge.label = the_edge.label[:-1] + 'ies'
                        else:
                            the_edge.label += 's'

                    break
            else:
                graph.add_node(verb, section)
                graph.add_edge(subject, verb)

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


@plac.annotations(
    file=
    plac.Annotation("The file to parse. Must be a XML formatted file.", type=str),

    disable_coreference_resolution=
    plac.Annotation('Flag indicating to not use coreference resolution.', kind='flag', abbrev='c'),

    disable_implicit_references=
    plac.Annotation('Flag indicating to not add implicit references.', kind='flag', abbrev='i'),

    disable_edge_annotation=
    plac.Annotation('Flag indicating to not annotate edges with relation types.', kind='flag', abbrev='a'),

    disable_reference_marking=
    plac.Annotation('Flag indicating to not mark reference types.', kind='flag', abbrev='m'),

    disable_summary=
    plac.Annotation('Flag indicating to not print the graph summary.', kind='flag', abbrev='s'),

    disable_graph_rendering=
    plac.Annotation('Flag indicating to not render (visualise) the graph structure.', kind='flag', abbrev='r'),

    debug_mode=
    plac.Annotation('Flag indicating to enable debug mode.', kind='flag', abbrev='d')
)
def main(file, disable_coreference_resolution=False, disable_implicit_references=False, disable_edge_annotation=False,
         disable_reference_marking=False, disable_summary=False, disable_graph_rendering=False, debug_mode=False):
    """Parse a text document and produce a score relating to conceptual density."""
    graph = ConceptGraph(parser=XMLSectionParser(not disable_edge_annotation, not disable_coreference_resolution),
                         implicit_references=not disable_implicit_references,
                         mark_references=not disable_reference_marking)
    graph.parse(file)

    if not disable_summary:
        graph.print_summary()

    print('Score: %.2f' % graph.score())

    if not disable_graph_rendering:
        graph.render()

    if debug_mode:
        sep = '#' + '-' * 78 + '#'
        print(sep, file=sys.stderr)
        print('DEBUG OUTPUT', file=sys.stderr)
        print(sep, file=sys.stderr)
        print('Forward References:', graph.forward_references, file=sys.stderr)
        print('Backward References:', graph.backward_references, file=sys.stderr)
        print('A priori Concepts:', graph.a_priori_concepts, file=sys.stderr)
        print('Emerging Concepts:', graph.emerging_concepts, file=sys.stderr)
        print(sep, file=sys.stderr)


if __name__ == '__main__':
    plac.call(main)
