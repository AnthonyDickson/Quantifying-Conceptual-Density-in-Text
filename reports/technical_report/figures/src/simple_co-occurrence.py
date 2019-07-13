import graphviz

if __name__ == '__main__':
	X = 'Japanese Car Brands'
	A = 'Mitsubishi'
	B = 'Subaru'
	C = 'Toyota'

	g = graphviz.Graph(engine='circo')
	
	g.edge(A, B)
	g.edge(A, C)
	g.edge(A, X)
	g.edge(B, C)
	g.edge(B, X)
	g.edge(C, X)

	g.render('simple_co-occurrence', format='png')
