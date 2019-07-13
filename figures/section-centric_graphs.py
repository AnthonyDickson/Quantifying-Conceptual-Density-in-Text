from conceptual_density.concept_graph import ConceptGraph

if __name__ == '__main__':
    filename = 'bread-concepts_marked.xml'

    graph = ConceptGraph(implicit_references=False,
                         mark_references=False)
    graph.parse(filename)
    graph.render(filename='bread_graph-simple', view=False)

    graph = ConceptGraph(implicit_references=False,
                         mark_references=True)
    graph.parse(filename)
    graph.render(filename='bread_graph-reference_marking', view=False)

    graph = ConceptGraph(implicit_references=True,
                         mark_references=False)
    graph.parse(filename)
    graph.render(filename='bread_graph-implicit_references', view=False)

    graph = ConceptGraph(implicit_references=True,
                         mark_references=True)
    graph.parse(filename)
    graph.render(filename='bread_graph', view=False)
