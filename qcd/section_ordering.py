import itertools
import statistics

import numpy as np
import plac

from qcd.concept_graph import ConceptGraph
from qcd.xml_parser import XMLParser


def evaluate_permutation(graph, i, permutation, permutations, scores):
    print(f'\rProcessed {i + 1} section ordering permutations...', end='')
    permutations.append(permutation)
    graph.sections = permutation
    graph.mark_edges()
    scores.append(graph.score())


@plac.annotations(
    file=plac.Annotation("The file to parse. Must be a XML formatted file.", type=str),
    n_samples=plac.Annotation("The number of section ordering permutations to sample. "
                              "Set to -1 to sample each unique permutation.", type=int, kind='option'),
    disable_implicit_references=plac.Annotation('Flag indicating to not add implicit references.', kind='flag',
                                                abbrev='i'),
    disable_reference_marking=plac.Annotation('Flag indicating to not mark reference types.', kind='flag', abbrev='m'),
    disable_edge_annotation=plac.Annotation('Flag indicating to not annotate edges with relation types.', kind='flag',
                                            abbrev='a'),
    disable_summary=plac.Annotation('Flag indicating to not print the graph summary.', kind='flag', abbrev='s'),
    disable_graph_rendering=plac.Annotation('Flag indicating to not render (visualise) the graph structure.',
                                            kind='flag',
                                            abbrev='r')

)
def main(file, n_samples=-1, disable_implicit_references=False, disable_reference_marking=False,
         disable_edge_annotation=False,
         disable_summary=False, disable_graph_rendering=False):
    """Run an experiment testing how ordering of sections affects the scoring of conceptual density for a given
    document.

    NOTE: This very slow for documents with any more than 7 sections due to the O(n!) time complexity of checking each
    permutation of section ordering.
    """

    graph = ConceptGraph(parser=XMLParser(not disable_edge_annotation, not disable_implicit_references),
                         mark_references=not disable_reference_marking)
    graph.parse(file)

    if not disable_summary:
        graph.print_summary()

    print('Original Section Ordering: %s' % graph.sections)
    print('Score on Original Ordering: %.2f' % graph.score())

    if not disable_graph_rendering:
        graph.render()

    scores = []
    permutations = []

    assert n_samples == -1 or n_samples > 0, 'Parameter `n-samples` must be -1 or a positive integer.'

    if n_samples > 0:
        for i in range(n_samples):
            permutation = np.random.permutation(graph.sections)

            evaluate_permutation(graph, i, permutation.tolist(), permutations, scores)
    else:
        for i, permutation in enumerate(itertools.permutations(graph.sections)):
            evaluate_permutation(graph, i, permutation, permutations, scores)

    print('\nDone.')

    min_score = min(scores)
    max_score = max(scores)

    if len(scores) > 10:
        print('scores: %s...' % ['%.2f' % score for score in scores[:10]])
    else:
        print('scores: %s' % ['%.2f' % score for score in scores])

    print('min: %.2f - ordering: %s' % (min_score, permutations[scores.index(min_score)]))
    print('max: %.2f - ordering: %s' % (max_score, permutations[scores.index(max_score)]))
    print('Mean: %.2f - Std. Dev.: %.2f' % (sum(scores) / len(scores), statistics.stdev(scores)))
    print('Max Absolute Difference: %.2f - Max Diff. Ratio: %.2f' % (max_score - min_score,
                                                                     (max_score - min_score) / max_score))


if __name__ == '__main__':
    plac.call(main)
