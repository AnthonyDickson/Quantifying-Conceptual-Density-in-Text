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

    for sent in doc.sents:
        for token in sent:
            if token.lemma_ == 'be' and token.dep_ == 'ROOT':
                subj = list(filter(lambda left: left.dep_.endswith('subj'), token.lefts))
                subj_tokens = []

                for tok in subj:
                    subj_tokens += tok.subtree

                if len(subj_tokens) > 0:
                    if subj_tokens[0].tag_ == 'DT':
                        subj_tokens = subj_tokens[1:]

                    subj_tokens = filter(lambda tok: len(tok.text.strip()) > 0, subj_tokens)
                    emerging_concepts.append(' '.join(map(lambda tok: tok.text, subj_tokens)))


@plac.annotations(
    file=plac.Annotation('The XML document to parse.'),
    strategy=plac.Annotation('How to parse the document', choices=['corenlp', 'spacy'], kind='option'),
)
def main(file, strategy='spacy'):
    tree = ElementTree.parse(file)
    root = tree.getroot()

    emerging_concepts = []

    for section in root.iterfind('section'):
        text = section.findtext('text')

        if strategy == 'spacy':
            spacy_strategy(emerging_concepts, text)
        elif strategy == 'corenlp':
            corenlp_strategy(emerging_concepts, text)

    print(f'Emerging Concepts: {emerging_concepts}')


if __name__ == '__main__':
    plac.call(main)
