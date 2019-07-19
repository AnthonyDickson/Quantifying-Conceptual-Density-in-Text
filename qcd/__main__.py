import sys
from xml.etree import ElementTree as ET

import plac

from qcd.concept_graph import ConceptGraph
from qcd.xml_parser import XMLParser


@plac.annotations(
    file=
    plac.Annotation("The file to parse. Must be a XML formatted file.", type=str),

    disable_coreference_resolution=
    plac.Annotation('Flag indicating to not use coreference resolution.', kind='flag', abbrev='c'),

    disable_implicit_references=
    plac.Annotation('Flag indicating to not add implicit references.', kind='flag', abbrev='i'),

    disable_edge_annotation=
    plac.Annotation('Flag indicating to not annotate edges with relation types.', kind='flag', abbrev='a'),

    disable_reference_marking=
    plac.Annotation('Flag indicating to not mark reference types.', kind='flag', abbrev='m'),

    disable_summary=
    plac.Annotation('Flag indicating to not print the graph summary.', kind='flag', abbrev='s'),

    disable_graph_rendering=
    plac.Annotation('Flag indicating to not render (visualise) the graph structure.', kind='flag', abbrev='r'),

    debug_mode=
    plac.Annotation('Flag indicating to enable debug mode.', kind='flag', abbrev='d')
)
def main(file, disable_coreference_resolution=False, disable_implicit_references=False, disable_edge_annotation=False,
         disable_reference_marking=False, disable_summary=False, disable_graph_rendering=False, debug_mode=False):
    """Parse a text document and produce a score relating to conceptual density."""
    parser = XMLParser(not disable_edge_annotation, not disable_implicit_references,
                       not disable_coreference_resolution)
    graph = ConceptGraph(parser=parser, mark_references=not disable_reference_marking)

    try:
        graph.parse(file)
    except ET.ParseError as e:
        print('Could not parse the file. \n%s.' % e.msg.capitalize(), file=sys.stderr)
        exit(1)
    except FileNotFoundError as e:
        print('Could not open the file. \n%s' % e)
        exit(2)

    if not disable_summary:
        graph.print_summary()

    print('Score: %.2f' % graph.score())

    if not disable_graph_rendering:
        graph.render()

    if debug_mode:
        sep = '#' + '-' * 78 + '#'
        print(sep, file=sys.stderr)
        print('DEBUG OUTPUT', file=sys.stderr)
        print(sep, file=sys.stderr)
        print('Forward References:', graph.forward_references, file=sys.stderr)
        print('Backward References:', graph.backward_references, file=sys.stderr)
        print('A priori Concepts:', graph.a_priori_concepts, file=sys.stderr)
        print('Emerging Concepts:', graph.emerging_concepts, file=sys.stderr)
        print(sep, file=sys.stderr)


if __name__ == '__main__':
    plac.call(main)
