from xml.etree import ElementTree

import plac
import spacy

from qcd.corenlp import CustomCoreNLPClient


def corenlp_strategy(emerging_concepts, text):
    # set up the client
    print('---')
    print('connecting to CoreNLP Server...')

    client = CustomCoreNLPClient(server='http://localhost:9000',
                                 default_annotators="tokenize,ssplit,pos,lemma,parse,natlog,depparse,openie".split(','))
    annotation = client.annotate(text)

    for sentence in annotation['sentences']:
        subject_head = None

        for dep in sentence['basicDependencies']:
            if dep['dep'].endswith('subj'):
                subject_head = dep['dependentGloss']

            if dep['dep'] == 'cop':
                emerging_concepts.append(subject_head)


def spacy_strategy(emerging_concepts, text):
    nlp = spacy.load('en')
    doc = nlp(text)

    concept_tokens = []

    for sent in doc.sents:
        for token in filter(lambda token: token.dep_ == 'ROOT', sent):
            if token.lemma_ == 'be':
                concept_tokens = list(filter(lambda left: left.dep_.endswith('subj'), token.lefts))
            elif token.lemma_ == 'define':
                concept_tokens = list(filter(lambda right: right.dep_ == 'dobj', token.rights))

            add_concept(concept_tokens, emerging_concepts)


def add_concept(concept_tokens, emerging_concepts):
    tokens = []

    for token in concept_tokens:
        tokens += token.subtree

    if len(tokens) > 0:
        if tokens[0].tag_ == 'DT':
            tokens = tokens[1:]

        tokens = filter(lambda token: len(token.text.strip()) > 0, tokens)
        emerging_concepts.add(' '.join(map(lambda token: token.text, tokens)))


@plac.annotations(
    file=plac.Annotation('The XML document to parse.'),
    strategy=plac.Annotation('How to parse the document', choices=['corenlp', 'spacy'], kind='option'),
)
def main(file, strategy='spacy'):
    tree = ElementTree.parse(file)
    root = tree.getroot()

    emerging_concepts = set()

    for section in root.iterfind('section'):
        text = section.findtext('text')

        if strategy == 'spacy':
            spacy_strategy(emerging_concepts, text)
        elif strategy == 'corenlp':
            corenlp_strategy(emerging_concepts, text)

    print(f'Emerging Concepts: {emerging_concepts}')


if __name__ == '__main__':
    plac.call(main)
