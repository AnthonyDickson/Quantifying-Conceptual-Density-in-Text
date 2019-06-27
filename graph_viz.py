import argparse
import pickle

import graphviz

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create a visualisation of a graph structure.')
    parser.add_argument('-f', '--file', type=str, help='The file containing a graph.')
    parser.add_argument('-d', '--directed', action='store_true',
                        help='Flag indicating that the graph should be a directed one.')
    args = parser.parse_args()

    with open(args.file, 'rb') as f:
        _, graph = pickle.load(f)

    g = graphviz.Digraph(engine='circo') if args.directed else graphviz.Graph(engine='circo')
    g.attr(mindist='1.5')
    processed_edges = []

    for node in graph:
        for other in graph[node]:
            if {node, other} not in processed_edges:
                g.edge(node, other)
                processed_edges.append({node, other})

    g.render('graph', format='png', view=True)
