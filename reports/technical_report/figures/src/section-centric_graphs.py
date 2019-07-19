from qcd.concept_graph import ConceptGraph
from qcd.xml_parser import XMLParser

if __name__ == '__main__':
    filename = 'bread-concepts_marked.xml'

    parser = XMLParser(implicit_references=False)

    graph = ConceptGraph(parser, mark_references=False)
    graph.parse(filename)
    graph.render(filename='bread_graph-simple', view=False)

    graph = ConceptGraph(parser, mark_references=True)
    graph.parse(filename)
    graph.render(filename='bread_graph-reference_marking', view=False)

    parser.implicit_references = True

    graph = ConceptGraph(parser, mark_references=False)
    graph.parse(filename)
    graph.render(filename='bread_graph-implicit_references', view=False)

    graph = ConceptGraph(parser, mark_references=True)
    graph.parse(filename)
    graph.render(filename='bread_graph', view=False)
