import os
import random
from typing import Optional
from typing import Tuple
from xml.etree import ElementTree as ET

import pandas as pd
import plac
import spacy

from qcd.concept_graph import ConceptGraph, DirectedEdge
from qcd.xml_parser import XMLParser, CoreNLPParser, OpenIEParser, EnsembleParser


# noinspection PyStringFormat


def evaluate(graph, a_priori_concepts, emerging_concepts, forward_references, backward_references):
    """Evaluate the labels derived from a given concept graph and a set of ground truth labels.

    :param graph: A concept graph contained predicted labels for concepts and references.
    :param a_priori_concepts: The ground truth set of a priori concepts in the document.
    :param emerging_concepts: The ground truth set of emerging concepts in the document.
    :param forward_references: The ground truth set of forward references in the document.
    :param backward_references: The ground truth set of a backward references in the document.

    :return: A Pandas DataFrame object containing the metrics calculated for the given parser and ground truth labels.
    """
    # Get head of edges since they represent the thing that is being referenced
    graph_forward_references = {edge.head for edge in graph.forward_references}
    graph_backward_references = {edge.head for edge in graph.backward_references}

    concepts_precision, concepts_recall, concepts_f1 = \
        precision_recall_f1(a_priori_concepts.union(emerging_concepts),
                            graph.a_priori_concepts.union(graph.emerging_concepts))

    # TODO: Fix NaNs in precision for forward references in some documents
    #  (e.g. bread_annotations.xml, closures_annotations.xml)
    references_precision, references_recall, references_f1 = \
        precision_recall_f1(forward_references.union(backward_references),
                            graph_forward_references.union(graph_backward_references))
    results = {
        'A Priori Concepts': [*precision_recall_f1(a_priori_concepts, graph.a_priori_concepts)],
        'Emerging Concepts': [*precision_recall_f1(emerging_concepts, graph.emerging_concepts)],
        'Concepts Overall': [concepts_precision, concepts_recall, concepts_f1],
        'Forward References': [*precision_recall_f1(forward_references, graph_forward_references)],
        'Backward References': [*precision_recall_f1(backward_references, graph_backward_references)],
        'References Overall': [references_precision, references_recall, references_f1],
        'Overall Average': [0.5 * (concepts_precision + references_precision),
                            0.5 * (concepts_recall + references_recall),
                            0.5 * (concepts_f1 + references_f1)],
    }

    metrics_df = pd.DataFrame.from_dict(results, orient='index', columns=['precision', 'recall', 'f1'])

    return metrics_df


def precision_recall_f1(target: set, prediction: set) -> Tuple[float, float, float]:
    """Calculate the precision, recall and f1 metrics for two sets.
    :param target: The ground truth set.
    :param prediction: The predicted set.
    :return: A 3-tuple containing the precision, recall and f1-score.
    """
    try:
        precision = len(target.intersection(prediction)) / len(prediction)
    except ZeroDivisionError:
        precision = float('nan')

    try:
        recall = len(target.intersection(prediction)) / len(target)
    except ZeroDivisionError:
        recall = float('nan')

    try:
        f1 = 2 * ((precision * recall) / (precision + recall))
    except ZeroDivisionError:
        f1 = float('nan')

    return precision, recall, f1


def extract_annotations_from_file(filename):
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

    return a_priori_concepts, backward_references, emerging_concepts, forward_references


@plac.annotations(
    filename=plac.Annotation('The annotated file to evaluate the model with.'),
    output_dir=plac.Annotation('The directory in which the calculated metrics should be saved. '
                               'By default metrics are not saved.', type=str, kind='option', abbrev='o'),
    random_trials=plac.Annotation('The number of trials for randomly labelling concepts that should be performed. '
                                  'If set to zero then evaluation is run without random labelling.',
                                  type=int, kind='option', abbrev='n')
)
def main(filename: str, output_dir: Optional[str] = None, random_trials: int = 0):
    pd.set_option('precision', 2)

    basename = os.path.splitext(os.path.basename(filename))

    if output_dir:
        if output_dir[-1] != '/':
            output_dir += '/'

        os.makedirs(output_dir, exist_ok=True)

    a_priori_concepts, backward_references, emerging_concepts, forward_references = extract_annotations_from_file(
        filename)
    parsers = [XMLParser(), CoreNLPParser(), OpenIEParser(), EnsembleParser()]

    if random_trials < 1:
        evaluate_deterministic(a_priori_concepts, backward_references, basename, emerging_concepts, filename,
                               forward_references, output_dir, parsers)
    else:
        evaluate_random(a_priori_concepts, backward_references, basename, emerging_concepts, filename,
                        forward_references, output_dir, parsers, random_trials)


def evaluate_deterministic(a_priori_concepts, backward_references, basename, emerging_concepts, filename,
                           forward_references, output_dir, parsers):
    for parser in parsers:
        graph = ConceptGraph(parser)
        graph.parse(filename)

        df = evaluate(graph, a_priori_concepts, emerging_concepts, forward_references,
                      backward_references)

        if output_dir:
            path = f'{output_dir}{parser.__class__.__name__}-{basename[0]}.csv'

            with open(path, 'w') as f:
                df.to_csv(f)

            print(f'Saved results for {parser.__class__.__name__} to {path}')

        print(f'Results for: {parser.__class__.__name__}')
        print(df)
        print()


def evaluate_random(a_priori_concepts, backward_references, basename, emerging_concepts, filename,
                    forward_references, output_dir, parsers, random_trials):
    for parser in parsers:
        graph = ConceptGraph(parser)
        graph.parse(filename)

        for trial in range(random_trials):
            # Mark concepts
            graph.a_priori_concepts = set()
            graph.emerging_concepts = set()

            for node in graph.nodes:
                if random.uniform(0, 1) < 0.5:
                    graph.a_priori_concepts.add(node)
                else:
                    graph.emerging_concepts.add(node)

            # Redo edges
            edges = graph.edges.copy()

            for edge in edges:
                new_edge = DirectedEdge(edge.tail, edge.head)
                new_edge.style = edge.style
                new_edge.frequency = edge.frequency

                graph.set_edge(new_edge)

            graph.mark_edges()

            # Evaluate graph
            df = evaluate(graph, a_priori_concepts, emerging_concepts, forward_references,
                          backward_references)

            if output_dir:
                path = f'{output_dir}{parser.__class__.__name__}-{basename[0]}-random_{trial + 1}.csv'

                with open(path, 'w') as f:
                    df.to_csv(f)

            print(f'\rTrial {trial + 1} of {random_trials} for {parser.__class__.__name__}', end='')

        print()


if __name__ == '__main__':
    plac.call(main)
