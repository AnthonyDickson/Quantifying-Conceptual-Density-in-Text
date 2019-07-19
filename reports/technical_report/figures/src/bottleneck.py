import pickle
import re

import graphviz
import nltk


class Text2Graph:
    def __init__(self, text):
        self.stopwords = set(nltk.corpus.stopwords.words('english'))
        self.hyphenated_word = re.compile(r'^[a-z]+([/-][a-z]+)+$')
        # Grammar from:
        # S. N. Kim, T. Baldwin, and M.-Y. Kan. Evaluating n-gram based evaluation metrics for automatic keyphrase
        # extraction.
        # Technical report, University of Melbourne, Melbourne 2010.
        self.grammar = r"""
            NBAR:
                {<NN.*|JJ>*<NN.*>}  # Nouns and Adjectives, terminated with Nouns
                
            NP:
                {<NBAR><IN><NBAR>}  # Above, connected with in/of/etc...
                {<NBAR>}
        """
        self.chunker = nltk.RegexpParser(self.grammar)
        self.lemmatiser = nltk.stem.WordNetLemmatizer()
        self.graph = dict()
        self.text = text
        self.doc_length = 0

    def is_valid_token(self, t):
        return t.isalnum() or re.match(self.hyphenated_word, t)

    def add_token(self, token, context):
        for other in context:
            if other != token:
                try:
                    self.graph[token][other] += 1
                except KeyError:
                    if token not in self.graph:
                        self.graph[token] = {}

                if other not in self.graph[token]:
                    self.graph[token][other] = 1

    @staticmethod
    def remove_hyphenation(text):
        """Some documents, in particular LaTeX documents, like to break words
        over lines with hyphens, this method stitches these words back
        together.
        """
        skip_whitespace = False
        prev_char = ''
        processed = ''

        for char in text:
            if skip_whitespace:
                if char in ' \n':
                    continue
                else:
                    skip_whitespace = False

            if char == '\n' and prev_char == '-':
                processed = processed[:-1]
                skip_whitespace = True
            elif char == '\xad':  # remove soft hyphens
                continue
            else:
                processed += char

            prev_char = char

        return processed

    def split_words(self, tokens):
        """Split hyphenated words (two words joined by a hyphen, not a single
        word that was hyphenated as a visual line break) into two separate words.

        :param tokens: The set of tokens to process
        :return: The new set of tokens where hyphenated words have been split
        into two separate tokens.
        """
        res = []

        for token in tokens:
            res.append(token)

            if re.match(self.hyphenated_word, token):
                for t in re.split(r'[-/]', token):
                    res.append(t)

        return res

    def parse(self):
        text = self.text.lower()
        text = Text2Graph.remove_hyphenation(text)
        sentences = nltk.sent_tokenize(text)

        for i, sent in enumerate(sentences):
            tokens = nltk.word_tokenize(sent)
            tokens = filter(self.is_valid_token, tokens)
            pos = nltk.pos_tag(list(tokens))

            if pos:
                parse_tree = self.chunker.parse(pos)
                noun_phrases = list(parse_tree.subtrees(filter=lambda st: st.label() == 'NP'))
                nbar_phrases = [nbar.leaves() for nbar in noun_phrases]
                nbars = [' '.join([str(token) for token, pos in nbar]) for nbar in nbar_phrases]

                for nbar in nbars:
                    self.add_token(nbar, nbars)

    def save(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump((text, self.graph), f)

    @staticmethod
    def load(filepath):
        with open(filepath, 'rb') as f:
            text, edges = pickle.load(f)

        t2g = Text2Graph(text)
        t2g.graph = edges

        return t2g


if __name__ == '__main__':
    with open('SLAM.txt', 'r') as f:
        text = f.read()

    t2g = Text2Graph(text)
    t2g.parse()

    g = graphviz.Graph(engine='circo')
    g.attr(mindist='1.5')
    processed_edges = []

    for node in t2g.graph:
        for other in t2g.graph[node]:
            if {node, other} not in processed_edges:
                g.edge(node, other)
                processed_edges.append({node, other})

    g.render('bottleneck', format='png')
