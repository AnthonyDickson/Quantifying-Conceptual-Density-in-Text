from typing import List, Tuple
from xml.etree import ElementTree

import neuralcoref
import nltk
import spacy
from spacy.tokens.span import Span

from qcd.concept_graph import ImplicitReference, ConceptGraph
from qcd.parser import ParserI


class XMLParser(ParserI):
    """Parser for XML documents.

    Expects XML documents to have 'section' tags containing a 'title' tag and a 'text' tag around the text.
    """

    def __init__(self, annotate_edges: bool = True, implicit_references: bool = True,
                 resolve_coreferences: bool = False):
        """Create a parser for XML documents.

        :param annotate_edges: Whether or not to annotate edges with a relationship type.
        :param implicit_references: Whether or not to add implicit references to the graph during parsing.
        :param resolve_coreferences: Whether or not to resolve coreferences.
        """
        self.chunker: nltk.RegexpParser = nltk.RegexpParser(self.get_grammar())
        self.lemmatizer: nltk.WordNetLemmatizer = nltk.WordNetLemmatizer()

        self.resolve_coreferences: bool = resolve_coreferences
        self.implicit_references: bool = implicit_references
        self.annotate_edges: bool = annotate_edges

    def get_grammar(self) -> str:
        return r"""
           NBAR:
               {<DT>?<NN.*|JJ>*<NN.*>} # Nouns and Adjectives, terminated with Nouns

           NP:
               {<NBAR>(<IN|CC><NBAR>)*}  # Above, connected with in/of/etc...
       """

    def parse(self, filename: str, graph: ConceptGraph):
        """Parse a file and build up a graph structure.

        :param filename: The file to parse.
        :param graph: The graph instance to add the nodes and edges to.
        """
        tree = ElementTree.parse(filename)
        root = tree.getroot()

        if self.resolve_coreferences:
            nlp_ = spacy.load('en')
            neuralcoref.add_to_pipe(nlp_)

            def nlp(text: str):
                # noinspection PyProtectedMember
                return nlp_(nlp_(text)._.coref_resolved)
        else:
            nlp_ = spacy.load('en')

            def nlp(text: str):
                return nlp_(text)

        for section in root.findall('section'):
            section_title = section.find('title').text
            section_title = section_title.lower()

            section_text = section.find('text').text
            section_text = section_text.lower()

            span = nlp(section_text)
            self.chunk(span)

            for sent in span.sents:
                # TODO: Use spacy tags instead, more accurate.
                # TODO: Use CoreNLP parse tree
                tags = self.get_tagged(str(sent))
                parse_tree = self.chunker.parse(nltk.Tree('S', children=tags))

                # Find the subject of the sentence
                subject = self.get_subject(sent)

                graph.add_node(subject, section_title)

                self.add_gerund_phrase(subject, section_title, sent, graph)

                # Add other noun phrases to the graph
                for np in parse_tree.subtrees(lambda t: t.label() == 'NP'):
                    tags = np.leaves()

                    if tags[0][1] == 'DT':
                        tags = tags[1:]

                    entity = ' '.join([token for token, tag in tags])

                    graph.add_node(entity, section_title)
                    graph.add_edge(subject, entity)

                    if self.implicit_references:
                        self.add_implicit_references(tags, section_title, graph)

    # noinspection PyProtectedMember
    @staticmethod
    def filter_spans(spans):
        # Filter a sequence of spans so they don't contain overlaps
        sorted_spans = sorted(spans, key=lambda span: (span.end - span.start, span.start), reverse=True)
        result = []
        seen_tokens = set()
        for span in sorted_spans:
            if span.start not in seen_tokens and span.end - 1 not in seen_tokens:
                result.append(span)
                seen_tokens.update(range(span.start, span.end))
        return result

    @staticmethod
    def chunk(doc):
        """Chunk the doc into noun chunks.

        :param doc: The document to chunk
        """
        spans = list(doc.noun_chunks)
        spans = XMLParser.filter_spans(spans)

        with doc.retokenize() as retokenizer:
            for span in spans:
                retokenizer.merge(span)

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

    def get_subject(self, sent) -> str:
        # TODO: Handle cases where no subject found (e.g. subordinate clauses).
        # TODO: Handle subjects that have more than one actor (e.g. two things joined by 'and').
        subject = [w for w in sent.root.lefts if w.dep_.startswith('nsubj')]

        if subject:
            subject = subject[0]

            if subject.tag_ == 'DT':
                subject = ' '.join(str(subject).split()[1:])

            subject = ' '.join(map(self.lemmatizer.lemmatize, str(subject).split()))

        return str(subject)

    def add_gerund_phrase(self, subject: str, section: str, sentence: Span, graph: ConceptGraph):
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

    def add_implicit_references(self, pos_tags: List[Tuple[str, str]], section: str, graph: ConceptGraph):
        """Derive nodes and edges from a POS tagged phrase.

        See `permutations()` for details on what kind of nodes and edges are derived.

        :param pos_tags: A phrase as a list of token, tag pairs.
        :param section: The section that the phrase appears in.
        :param graph: The graph to add the derived nodes and edges to.
        """
        for implicit_entity, context in self.permutations(pos_tags):
            graph.add_node(implicit_entity, section)
            graph.add_edge(context, implicit_entity, ImplicitReference)

    def permutations(self, tagged_phrase: List[Tuple[str, str]]) -> Tuple[str, str]:
        """Generate variations of a POS (part of speech) tagged phrase.

        Variations generated are:
        - The entire phrase itself
        - nbar phrases (sequences of adjectives and/or nouns, terminated by a noun)
        - noun chunks (sequences of one or more nouns)

        Variations are yielded alongside a 'context', which represents the phrase that the variation was generated from.

        As an example, consider the sentence 'Zeus is the sky and thunder god in ancient Greek religion.' and the POS
        tagged phrase `[('Zeus', 'NNP'), ('is', 'VBZ'), ('the', 'DT'), ('sky', 'NN'), ('and', 'CC'), ('thunder', 'NN'),
         ('god', 'NN'), ('in', 'IN'), ('ancient', 'JJ'), ('Greek', 'JJ'), ('religion', 'NN'), ('.', '.')]`.
        The noun phrases we can expect are 'Zeus', 'sky and thunder god in ancient Greek religion'. For the second noun
        phrase we can expect the nbar phrases 'sky and thunder god' and 'ancient Greek religion'. These two nbar phrases
        would be yield with the noun phrase as the context.

        :param tagged_phrase: List of 2-tuples containing a POS tag and a token.
        :return: Yields 2-tuples containing a variation of `tagged_phrase` and the context it appears in.
        """
        context = ' '.join([token for token, tag in tagged_phrase])
        tree = self.chunker.parse(nltk.Tree('S', tagged_phrase))

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
                yield from XMLParser.process_nbar_chunk(nbar_chunk, np)

                nbar_chunk = []

        if nbar_chunk:
            yield from XMLParser.process_nbar_chunk(nbar_chunk, np)

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
                yield from XMLParser.process_noun_chunk(noun_chunk, nbar)

                noun_chunk = []

        if noun_chunk:
            yield from XMLParser.process_noun_chunk(noun_chunk, nbar)

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
