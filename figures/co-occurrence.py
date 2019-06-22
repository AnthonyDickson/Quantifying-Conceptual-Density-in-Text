import graphviz

if __name__ == '__main__':
	X = 'Japanese Car Brands'
	A = 'Mitsubishi'
	B = 'Subaru'
	C = 'Toyota'

	g = graphviz.Digraph(engine='circo')
	
	g.node(X)
	g.edge(X, A)
	g.edge(X, B)
	g.edge(X, C)

	g.render('co-occurrence', format='png')
