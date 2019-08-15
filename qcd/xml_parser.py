import warnings
from abc import ABC
from typing import List, Tuple
from xml.etree import ElementTree

import neuralcoref
import nltk
import spacy
from spacy.tokens.span import Span

from qcd.concept_graph import ImplicitReference, ConceptGraph, Relation
from qcd.corenlp import CustomCoreNLPClient
from qcd.graph import Node, Section
from qcd.parser import ParserI


class ParserABC(ParserI, ABC):
    def __init__(self, annotate_edges: bool = True, implicit_references: bool = True,
                 resolve_coreferences: bool = False):
        """Create a parser for XML documents.

        :param annotate_edges: Whether or not to annotate edges with a relationship type.
        :param implicit_references: Whether or not to add implicit references to the graph during parsing.
        :param resolve_coreferences: Whether or not to resolve coreferences.
        """
        self.resolve_coreferences: bool = resolve_coreferences
        self.implicit_references: bool = implicit_references
        self.annotate_edges: bool = annotate_edges

    def add_implicit_references(self, pos_tags: List[Tuple[str, str]], section: Section, graph: ConceptGraph):
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


class XMLParser(ParserABC):
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
        super().__init__(annotate_edges, implicit_references, resolve_coreferences)

        self.chunker: nltk.RegexpParser = nltk.RegexpParser(self.get_grammar())
        self.lemmatizer: nltk.WordNetLemmatizer = nltk.WordNetLemmatizer()

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

            if section_title == 'references':
                continue

            section_text = section.find('text').text
            section_text = section_text.lower()

            span = nlp(section_text)

            for sent in span.sents:
                # TODO: Use CoreNLP parse tree
                tags = self.get_tagged(sent)
                parse_tree = self.chunker.parse(nltk.Tree('S', children=tags))

                # TODO: Does the sentence need to be noun chunked?
                # Find the subject of the sentence
                subject = Node(self.get_subject(sent))

                graph.add_node(subject, section_title)

                self.add_gerund_phrase(subject, section_title, sent, graph)

                # Add other noun phrases to the graph
                for np in parse_tree.subtrees(lambda t: t.label() == 'NP'):
                    tags = np.leaves()

                    if tags[0][1] == 'DT':
                        tags = tags[1:]

                    entity = Node(' '.join([token for token, tag in tags]))

                    graph.add_node(entity, section_title)
                    graph.add_edge(subject, entity)

                    if self.implicit_references:
                        self.add_implicit_references(tags, section_title, graph)

    def get_tagged(self, phrase: Span) -> List[Tuple[str, str]]:
        """Normalise and tag a string.

        :param phrase: The string to process.
        :return: List of token, tag pairs.
        """
        tags = [(token.lemma_, token.tag_) for token in phrase]

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

    def add_gerund_phrase(self, subject: Node, section: Section, sentence: Span, graph: ConceptGraph):
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
            verb = Node(gerund.text)

            # TODO: Add edge between gerund and object
            # TODO: Remove redundant edge between subject and object since that relation is represented
            #  through the path subject -> verb -> object.
            # TODO: Refactor verbal phrase stuff such that we instead have
            #  subject -- verb (S-form) --> object
            #  including is_a and has_a relations.
            for right in gerund.rights:
                if 'obj' in right.dep_:
                    if self.annotate_edges:
                        object_ = Node(right.text)
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


class CoreNLPParserABC(ParserABC, ABC):
    """A parser for XML documents that uses a CoreNLP server for NLP."""

    def __init__(self, annotate_edges: bool = True, implicit_references: bool = False,
                 resolve_coreferences: bool = False, server_url: str = 'http://localhost:9000'):
        """Create a parser for XML documents.

        :param annotate_edges: Whether or not to annotate edges with a relationship type.
        :param implicit_references: Whether or not to add implicit references to the graph during parsing.
        :param resolve_coreferences: Whether or not to resolve coreferences.
        :param server_url: The URL to the CoreNLP server to use for NLP queries.
        """
        super().__init__(annotate_edges, implicit_references, resolve_coreferences)

        if implicit_references:
            warnings.warn('\'%s\' does not support implicit references. '
                          'Set the paramater \'implicit_references\' to False to hide this warning.'
                          % self.__class__.__name__)

        self.annotations = "tokenize,ssplit,pos,lemma,parse,natlog,depparse,openie".split(',')
        self.client = CustomCoreNLPClient(server=server_url,
                                          default_annotators=self.annotations)


class OpenIEParser(CoreNLPParserABC):
    def get_grammar(self) -> str:
        raise NotImplementedError('This parser does not define its own grammar.')

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

            if section_title == 'references':
                continue

            section_text = section.find('text').text
            section_text = section_text.lower()

            span = nlp(section_text)
            # self.chunk(span)

            for sent in span.sents:
                s = nlp(' '.join([tok.text for tok in filter(lambda tok: tok.tag_ not in {'RB'}, sent)]))

                annotation = self.client.annotate(s.text)

                for sentence in annotation['sentences']:
                    for triple in sentence['openie']:
                        subject, relation, object_ = triple['subject'], triple['relation'], triple['object']

                        if self.filter_triple(subject, relation, object_):
                            graph.add_relation(subject, relation, object_, section_title)

    def filter_triple(self, subject: str, relation: str, object_: str) -> bool:
        # annotation = self.client.annotate(object_)
        # parse_tree = nltk.Tree.fromstring(annotation['sentences'][0]['parse'])
        # child_nodes = [child for child in parse_tree[0]]
        #
        # if len(child_nodes) > 2:
        #     return False
        #
        # for node in child_nodes:
        #     if node.height() > 5:
        #         return False
        #
        #     if node.label() == 'VP':
        #         for child in node:
        #             if child.label() in {'NP', 'PP', 'VBG'}:
        #                 break
        #         else:
        #             return False
        #
        #     if node.height() > 2 and node.label() not in {'NP', 'VP', 'PP'}:
        #         return False

        return True


class CoreNLPParser(CoreNLPParserABC):
    def __init__(self, annotate_edges: bool = True, implicit_references: bool = False,
                 resolve_coreferences: bool = False, server_url: str = 'http://localhost:9000'):
        super().__init__(annotate_edges, implicit_references, resolve_coreferences, server_url)

        self.chunker: nltk.RegexpParser = nltk.RegexpParser(self.get_grammar())
        self.lemmatizer: nltk.WordNetLemmatizer = nltk.WordNetLemmatizer()

    def get_grammar(self) -> str:
        return r"""
           NBAR:
               {<DT>?<NN.*|JJ>*<NN.*>} # Nouns and Adjectives, terminated with Nouns

           NP:
               {<NBAR>(<IN|CC><NBAR>)*}  # Above, connected with in/of/etc...
       """

    def parse(self, filename: str, graph: ConceptGraph):
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

            if section_title == 'references':
                continue

            section_text = section.find('text').text
            section_text = section_text.lower()

            for sent in nltk.sent_tokenize(section_text):
                sent = sent.strip()
                # sent = nlp(sent)
                # sent = nlp(' '.join([tok.text for tok in filter(lambda tok: tok.tag_ not in {'RB'}, sent)]))

                annotation = self.client.annotate(sent)

                for sentence in annotation['sentences']:
                    parse_tree = nltk.Tree.fromstring(sentence['parse'])
                    # parse_tree.pretty_print()

                    for subject, verb, object_ in self.parse_the_parse_tree(parse_tree):
                        subject_tags = list(
                            filter(lambda token_tag: token_tag[1] not in {'DET', 'DT'}, nltk.pos_tag(subject)))
                        object_tags = list(
                            filter(lambda token_tag: token_tag[1] not in {'DET', 'DT'}, nltk.pos_tag(object_)))

                        subject = ' '.join([token for token, tag in subject_tags])
                        object_ = ' '.join([token for token, tag in object_tags])

                        graph.add_relation(Node(subject), Relation(' '.join(verb)), Node(object_),
                                           Section(section_title))

                        self.add_implicit_references(subject_tags, Section(section_title), graph)
                        self.add_implicit_references(object_tags, Section(section_title), graph)

    def parse_the_parse_tree(self, parse_tree):
        s = parse_tree[0]
        subject = None
        verb = None
        object_ = None
        modifying_phrase = None

        # TODO: Fix the case where the parse tree is shallow, i.e. no phrases were identified. Preprocess input into
        #  the CoreNLP server? Try work with what is given?
        if parse_tree.height() == 3:
            return

        for i in range(len(s)):
            if not subject and s[i].label() == 'NP':
                subject = s[i]
            # Look for main verb phrase
            elif not verb and s[i].label().startswith('VB'):
                # If the verb is not in its own VP then get the verb, will also have to get the object too.
                verb = s[i]
            elif not verb and s[i].label() == 'VP':
                # Verb phrases contain the verb plus the object.

                verb = s[i]

                # vp[0] is the pos tag and the verb of the VP
                main_verb = verb[0]

                # vp[1] is the phrase that directly follows the verb of the VP, i.e. the object
                if len(verb) > 1 and verb[1].label() in {'NP', 'VP', 'ADJP'}:
                    # Verb phrases may have more stuff nested in them
                    if verb[1].label() == 'VP':
                        has_appositive_phrase = False
                        phrase_verb = None
                        previous_object = None

                        # check constituent parts of the VP for multiple objects (appositions I think)
                        for phrase in verb[1]:
                            if phrase.label().startswith('VB'):
                                phrase_verb = phrase
                            elif phrase.label() in {'NP', 'PP'}:
                                if has_appositive_phrase:
                                    # appositives tend to be related to the main verb, not the appositive VP, so use
                                    # main verb here

                                    if phrase.label() == 'PP':
                                        # move the preposition to the relation for PPs
                                        yield subject.leaves(), main_verb.leaves() + \
                                              [phrase.leaves()[0]], phrase.leaves()[1:]
                                    else:
                                        # NP Appositives should be related to the object of the previous VP
                                        subject_ = next(previous_object.subtrees(lambda st: st.label() == 'NP'))
                                        yield subject_.leaves(), main_verb.leaves(), phrase.leaves()
                                else:
                                    if phrase.label() == 'PP':
                                        # move the preposition to the relation for PPs
                                        yield subject.leaves(), phrase_verb.leaves() + \
                                              [phrase.leaves()[0]], phrase.leaves()[1:]
                                    else:
                                        yield subject.leaves(), phrase_verb.leaves(), phrase.leaves()

                                    previous_object = phrase
                            elif phrase.label() in {',', '.'}:
                                # punctuation in the middle of a VP probably indicates that the following
                                # parts are appositive phrases
                                has_appositive_phrase = True
                    else:
                        object_ = verb[1]
                        verb = verb[0]
                else:
                    break

            elif verb and s[i].label() in {'NP', 'PP', 'ADJP'}:
                object_ = s[i]
            elif verb and s[i].label() == 'VP':
                # modifying verb phrase
                modifying_phrase = s[i]

                for phrase in modifying_phrase:
                    if phrase.label() in {'NP', 'VP'}:
                        yield subject.leaves(), modifying_phrase[0].leaves(), phrase.leaves()
                    elif phrase.label() == 'PP':
                        # move the preposition to the relation for PPs
                        yield subject.leaves(), modifying_phrase[0].leaves() + [phrase.leaves()[0]], phrase.leaves()[1:]

                break

        if subject and verb and object_:
            yield subject.leaves(), verb.leaves(), object_.leaves()
