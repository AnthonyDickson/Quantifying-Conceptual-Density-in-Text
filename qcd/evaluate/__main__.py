import os
from typing import Optional
from xml.etree import ElementTree as ET

import pandas as pd
import plac
import spacy

from qcd.evaluate.evaluation import evaluate_parser
from qcd.xml_parser import XMLParser, CoreNLPParser, OpenIEParser


@plac.annotations(
    filename=plac.Annotation('The annotated file to evaluate the model with.'),
    output_dir=plac.Annotation('The directory in which the calculated metrics should be saved. '
                               'By default metrics are not saved.')
)
def main(filename: str, output_dir: Optional[str] = None):
    pd.set_option('precision', 2)

    basename = os.path.splitext(os.path.basename(filename))

    if output_dir:
        if output_dir[-1] != '/':
            output_dir += '/'

        os.makedirs(output_dir, exist_ok=True)

    with open(filename, 'r') as f:
        tree = ET.parse(f)

    root = tree.getroot()

    a_priori_concepts = set()
    emerging_concepts = set()
    forward_references = set()
    backward_references = set()
    nlp = spacy.load('en')

    for section in root.findall('section'):
        annotations = section.find('annotations')

        if annotations:
            for annotation in annotations:
                tag = annotation.get('tag')
                tag = tag.lower()

                concept = annotation.text.lower()
                concept = nlp(concept)
                concept = ' '.join([token.lemma_ for token in concept])

                if tag == 'a priori':
                    a_priori_concepts.add(concept)
                elif tag == 'emerging':
                    emerging_concepts.add(concept)
                elif tag == 'forward':
                    forward_references.add(concept)
                elif tag == 'backward':
                    backward_references.add(concept)

    for parser in [XMLParser(), CoreNLPParser(), OpenIEParser()]:
        df = evaluate_parser(filename, parser, a_priori_concepts, emerging_concepts, forward_references,
                             backward_references)

        if output_dir:
            path = f'{output_dir}{parser.__class__.__name__}-{basename[0]}.csv'

            with open(path, 'w') as f:
                df.to_csv(f)


if __name__ == '__main__':
    plac.call(main)
