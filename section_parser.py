import argparse

from concept_graph import ConceptGraph

# TODO: Fix bugs with single letter entities
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parse a file and group the entities by section.')
    parser.add_argument('-f', '--file', metavar='INPUT_FILE', type=str, required=True,
                        help='The file to parse. Can be a `.xml` file.')

    parser.add_argument('-i', '--no-implicit-references', action='store_true',
                        help='Flag indicating to not add implicit references.')

    parser.add_argument('-m', '--no-reference-marking', action='store_true',
                        help='Flag indicating to not mark reference types.')

    parser.add_argument('-s', '--no-summary', action='store_true',
                        help='Flag indicating to not print the graph summary.')

    parser.add_argument('-r', '--no-render', action='store_true',
                        help='Flag indicating to not render (visualise) the graph structure.')

    args = parser.parse_args()

    graph = ConceptGraph(implicit_references=not args.no_implicit_references,
                         mark_references=not args.no_reference_marking)
    graph.parse(args.file)

    if not args.no_reference_marking:
        graph.mark_edges()

    if not args.no_summary:
        graph.print_summary()

    print('Score: %.2f' % graph.score())

    if not args.no_render:
        graph.render()
