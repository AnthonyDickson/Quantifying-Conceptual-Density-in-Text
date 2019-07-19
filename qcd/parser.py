from qcd.graph import GraphI


class ParserI:
    """An interface describing the guaranteed functionality of a parser object."""

    def get_grammar(self) -> str:
        """
        :return: The grammar used by this parser.
        """
        raise NotImplementedError

    def parse(self, filename: str, graph: GraphI):
        """Parse a file and build up a graph structure.

        :param filename: The file to parse.
        :param graph: The graph instance to add the nodes and edges to.
        """
        raise NotImplementedError
