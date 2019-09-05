from typing import Tuple

import pandas as pd

from qcd.concept_graph import ConceptGraph


# noinspection PyStringFormat


def evaluate_parser(filename, parser, a_priori_concepts, emerging_concepts, forward_references, backward_references):
    """Evaluate a parsing algorithm on a file given a set of ground truth labels.

    :param filename: The path to an annotated XML file.
    :param parser: The parser instance to use and evaluate.
    :param a_priori_concepts: The ground truth set of a priori concepts in the document.
    :param emerging_concepts: The ground truth set of emerging concepts in the document.
    :param forward_references: The ground truth set of forward references in the document.
    :param backward_references: The ground truth set of a backward references in the document.

    :return: A Pandas DataFrame object containing the metrics calculated for the given parser and ground truth labels.
    """
    graph = ConceptGraph(parser)
    graph.parse(filename)

    # Get tail of edges since they represent the thing that is making a forward/backward reference
    graph_forward_references = {edge.tail for edge in graph.forward_references}
    graph_backward_references = {edge.tail for edge in graph.backward_references}

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

    print('Results for: %s' % parser.__class__.__name__)
    print(metrics_df)
    print()

    return metrics_df


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
    false_positive_rate = len(prediction.difference(target)) / len(prediction) if len(prediction) > 0 else float('nan')

    precision = true_positive_rate / (true_positive_rate + false_positive_rate + eps)
    recall = true_positive_rate / (true_positive_rate + false_negative_rate + eps)
    f1 = 2 * ((precision * recall) / (precision + recall + eps)) - 2 * eps if precision != 0 and recall != 0 else 0

    return precision, recall, f1

