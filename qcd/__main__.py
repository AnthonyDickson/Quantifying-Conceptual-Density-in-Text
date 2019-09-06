import sys
import warnings
from datetime import datetime
from xml.etree import ElementTree

import plac

from qcd.concept_graph import ConceptGraph
from qcd.xml_parser import XMLParser, OpenIEParser, CoreNLPParser, EnsembleParser


@plac.annotations(
    file=
    plac.Annotation("The file to parse. Must be a XML formatted file.", kind='positional', type=str),

    parser_type=
    plac.Annotation('The type of parser to use.', kind='positional', type=str,
                    choices=['default', 'openie', 'corenlp', 'ensemble']),

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
def main(file, parser_type='default', disable_coreference_resolution=False, disable_implicit_references=False,
         disable_edge_annotation=False, disable_reference_marking=False, disable_summary=False,
         disable_graph_rendering=False, debug_mode=False):
    """Parse a text document and produce a score relating to conceptual density."""

    if parser_type == 'openie':
        parser_type = OpenIEParser
    elif parser_type == 'corenlp':
        parser_type = CoreNLPParser
    elif parser_type == 'ensemble':
        parser_type = EnsembleParser
    else:
        if parser_type != 'default':
            warnings.warn('Unrecognised parser type \'%s\' - using default parser.' % parser_type)

        parser_type = XMLParser

    parser = parser_type(not disable_edge_annotation, not disable_implicit_references,
                         not disable_coreference_resolution)
    graph = ConceptGraph(parser=parser, mark_references=not disable_reference_marking)

    try:
        start = datetime.now()

        graph.parse(file)

        delta = datetime.now() - start

        print('Document parsed in: %s' % delta)
    except ElementTree.ParseError as e:
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
