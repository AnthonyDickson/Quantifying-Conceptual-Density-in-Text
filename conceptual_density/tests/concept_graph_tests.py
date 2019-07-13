def all_nodes_in_section(graph):
    """Test if all nodes in the graph belong to a section.

    :param graph: The graph to test.
    :return: True if all nodes belong to a section, False otherwise.
    """
    for node in graph.nodes:
        for section in graph.sections:
            if node in graph.section_listings[section]:
                assert graph.section_index[node] == section
                break
        else:
            raise AssertionError('Node %s not in a section.' % node)

    return True
