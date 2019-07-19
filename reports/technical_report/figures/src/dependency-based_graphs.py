from qcd.concept_graph import ConceptGraph
from qcd.xml_parser import XMLParser

if __name__ == '__main__':
    parser = XMLParser(annotate_edges=False, implicit_references=False)
    filename = 'bread.xml'

    graph = ConceptGraph(parser=parser,
                         mark_references=False)
    graph.parse(filename)
    graph.render('bread_graph-sections_only-simple', view=False)

    graph = ConceptGraph(parser=parser,
                         mark_references=True)
    graph.parse(filename)
    graph.render('bread_graph-sections_only-reference_marking', view=False)

    parser.implicit_references = True

    graph = ConceptGraph(parser=parser,
                         mark_references=False)
    graph.parse(filename)
    graph.render('bread_graph-sections_only-implicit_references', view=False)

    graph = ConceptGraph(parser=parser,
                         mark_references=True)
    graph.parse(filename)
    graph.render('bread_graph-sections_only', view=False)
