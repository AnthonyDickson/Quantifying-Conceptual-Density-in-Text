from conceptual_density.concept_graph import ConceptGraph
from conceptual_density.parser import XMLSectionParser

if __name__ == '__main__':
    parser = XMLSectionParser(annotate_edges=False)
    filename = 'bread-sections_only.xml'

    graph = ConceptGraph(parser=parser,
                         implicit_references=False,
                         mark_references=False)
    graph.parse(filename)
    graph.render('bread_graph-sections_only-simple', view=False)

    graph = ConceptGraph(parser=parser,
                         implicit_references=False,
                         mark_references=True)
    graph.parse(filename)
    graph.render('bread_graph-sections_only-reference_marking', view=False)

    graph = ConceptGraph(parser=parser,
                         implicit_references=True,
                         mark_references=False)
    graph.parse(filename)
    graph.render('bread_graph-sections_only-implicit_references', view=False)

    graph = ConceptGraph(parser=parser,
                         implicit_references=True,
                         mark_references=True)
    graph.parse(filename)
    graph.render('bread_graph-sections_only', view=False)
