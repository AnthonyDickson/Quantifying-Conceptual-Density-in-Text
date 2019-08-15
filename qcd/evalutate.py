import xml.etree.ElementTree as ET
from typing import Tuple

import plac
import spacy

from qcd.concept_graph import ConceptGraph
from qcd.xml_parser import XMLParser


# noinspection PyStringFormat
@plac.annotations(
    filename=plac.Annotation('The annotated file to evaluate the model with.')
)
def main(filename: str):
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
                concept_type = annotation.get('type')
                reference_type = annotation.get('reference')
                concept = annotation.text.lower()
                concept = nlp(concept)
                concept = ' '.join([token.lemma_ for token in concept])

                if concept_type == 'a priori':
                    a_priori_concepts.add(concept)
                elif concept_type == 'emerging':
                    emerging_concepts.add(concept)

                if reference_type == 'forward':
                    forward_references.add(concept)
                elif reference_type == 'backward':
                    backward_references.add(concept)

    parser = XMLParser()
    graph = ConceptGraph(parser)
    graph.parse(filename)

    # Get tail of edges since they represent the thing that is making a forward/backward reference
    graph_forward_references = {edge.tail for edge in graph.forward_references}
    graph_backward_references = {edge.tail for edge in graph.backward_references}

    concepts_precision, concepts_recall, concepts_f1 = \
        precision_recall_f1(a_priori_concepts.union(emerging_concepts),
                            graph.a_priori_concepts.union(graph.emerging_concepts))
    references_precision, references_recall, references_f1 = \
        precision_recall_f1(forward_references.union(backward_references),
                            graph_forward_references.union(graph_backward_references))

    print('#' + '=' * 42 + '#')
    print('|%-20s |    p |    r |   f1 |' % 'Variable')
    print('|' + '-' * 21 + '|' + '-' * 6 + '|' + '-' * 6 + '|' + '-' * 6 + '|')
    print('|%-20s | %4.2f | %4.2f | %4.2f |'
          % ('A Priori Concepts', *precision_recall_f1(a_priori_concepts, graph.a_priori_concepts)))
    print('|%-20s | %4.2f | %4.2f | %4.2f |'
          % ('Emerging Concepts', *precision_recall_f1(emerging_concepts, graph.emerging_concepts)))
    print('|' + ' - ' * 7 + '|' + ' - ' * 2 + '|' + ' - ' * 2 + '|' + ' - ' * 2 + '|')
    print('|%-20s | %4.2f | %4.2f | %4.2f |'
          % ('Concepts Overall', concepts_precision, concepts_recall, concepts_f1))

    print('|' + '-' * 21 + '|' + '-' * 6 + '|' + '-' * 6 + '|' + '-' * 6 + '|')
    print('|%-20s | %4.2f | %4.2f | %4.2f |'
          % ('Forward References', *precision_recall_f1(forward_references, graph_forward_references)))
    print('|%-20s | %4.2f | %4.2f | %4.2f |'
          % ('Backward References', *precision_recall_f1(backward_references, graph_backward_references)))
    print('|' + ' - ' * 7 + '|' + ' - ' * 2 + '|' + ' - ' * 2 + '|' + ' - ' * 2 + '|')
    print('|%-20s | %4.2f | %4.2f | %4.2f |'
          % ('References Overall', references_precision, references_recall, references_f1))
    print('|' + '-' * 21 + '|' + '-' * 6 + '|' + '-' * 6 + '|' + '-' * 6 + '|')
    print('|%-20s | %4.2f | %4.2f | %4.2f |'
          % ('Overall Average',
             0.5 * (concepts_precision + references_precision),
             0.5 * (concepts_recall + references_recall),
             0.5 * (concepts_f1 + references_f1)))
    print('#' + '=' * 42 + '#')


def precision_recall_f1(target: set, prediction: set) -> Tuple[float, float, float]:
    """Calculate the precision, recall and f1 metrics for two sets.
    :param target: The ground truth set.
    :param prediction: The predicted set.
    :return: A 3-tuple containing the precision, recall and f1-score.
    """
    # Use a small value to avoid zero division, zero division is treated as if it produces zero for the sake of
    # numerical stability and to prevent the whole program from crashing and burning.
    eps = 1e-128

    true_positive_rate = len(target.intersection(prediction)) / len(target) if len(target) > 0 else float('nan')
    false_negative_rate = 1 - true_positive_rate
    false_positive_rate = len(prediction.difference(target)) / len(prediction) if len(target) > 0 else float('nan')

    precision = true_positive_rate / (true_positive_rate + false_positive_rate + eps)
    recall = true_positive_rate / (true_positive_rate + false_negative_rate + eps)
    f1 = 2 * ((precision * recall) / (precision + recall + eps)) - 2 * eps if precision != 0 and recall != 0 else 0

    return precision, recall, f1


if __name__ == '__main__':
    plac.call(main)
