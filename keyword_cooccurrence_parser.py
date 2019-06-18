import argparse
import os
import pickle
import re

import nltk


class Text2Graph:
    def __init__(self, text):
        self.stopwords = set(nltk.corpus.stopwords.words('english'))
        self.hyphenated_word = re.compile(r'^[a-z]+([/-][a-z]+)+$')
        # Grammar from:
        # S. N. Kim, T. Baldwin, and M.-Y. Kan. Evaluating n-gram based evaluation metrics for automatic keyphrase extraction.
        # Technical report, University of Melbourne, Melbourne 2010.
        self.grammar = r"""
            NBAR:
                {<NN.*|JJ>*<NN.*>}  # Nouns and Adjectives, terminated with Nouns
                
            NP:
                {<NBAR>}
                {<NBAR><IN><NBAR>}  # Above, connected with in/of/etc...
        """
        self.chunker = nltk.RegexpParser(self.grammar)
        self.lemmatiser = nltk.stem.WordNetLemmatizer()
        self.graph = dict()
        self.positions = dict()
        self.text = text
        self.doc_length = 0

    def is_valid_token(self, t):
        return (t.isalnum() or re.match(self.hyphenated_word, t)) \
            # and t not in self.stopwords

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
        positions = dict()

        for i, sent in enumerate(sentences):
            tokens = nltk.word_tokenize(sent)
            tokens = filter(self.is_valid_token, tokens)
            # tokens = map(self.lemmatiser.lemmatize, tokens)
            pos = nltk.pos_tag(list(tokens))

            if pos:
                parse_tree = self.chunker.parse(pos)
                noun_phrases = list(parse_tree.subtrees(filter=lambda st: st.label() == 'NP'))
                nbar_phrases = [nbar.leaves() for nbar in noun_phrases]
                nbars = [' '.join([str(token) for token, pos in nbar]) for nbar in nbar_phrases]

                for nbar in nbars:
                    if nbar in positions:
                        positions[nbar].append(i)
                    else:
                        positions[nbar] = [i]

                for nbar in nbars:
                    self.add_token(nbar, nbars)

        # print(positions)
        self.positions = positions

    def keyword_distances(self):
        for ref in self.graph:
            ref, others = ref, self.graph[ref]

            for other in others:
                print('distance between %s and %s: %f' % (ref, other,
                                                          sum(self.positions[ref]) / len(self.positions[ref]) - sum(
                                                              self.positions[other]) / len(self.positions[other])))

    def density_score(self):
        """A fairly arbitrary scoring metric that attempts to measure the 'density' of a given document based on it's
        term co-occurrence graph.
        """
        N = len(self.graph)
        res = N

        for node in self.graph:
            res += sum(self.graph[node].values()) / N

        return res

    def ranked_terms(self):
        """Rank the terms that appear in this document by frequency."""
        word_conn_pairs = [(k, sum(self.graph[k].values())) for k in self.graph]

        ranked = sorted(word_conn_pairs, key=lambda pair: pair[1])

        return list(reversed(ranked))

    def top_n(self, term, n=5):
        """Get the top n related words, ranked by term frequency.
        :param term: The term to get the related words for.
        :param n: How many results to show.
        :return:
        """
        term_count_pairs = [(k, self.graph[term][k]) for k in self.graph[term]]

        ranked = list(reversed(sorted(term_count_pairs, key=lambda pair: pair[1])))

        return ranked[:min(n, len(ranked))]

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
    parser = argparse.ArgumentParser(description='Parse text documents in the given directory and assign a rough score '
                                                 'for each document\'s conceptual density.')
    parser.add_argument('-c', '--corpora', type=str, default='corpora/',
                        help='The directory where the text corpora are located.')
    args = parser.parse_args()

    for path, dirs, files in os.walk(args.corpora):
        for file in files:
            if file.endswith('.txt'):
                title = file[:-4]

                if len(title) > 32:
                    title = title[:29] + '...'

                print('Processing \'%s\'...' % title)

                with open(path + '/' + file, 'r') as f:
                    text = f.read()

                t2g = Text2Graph(text)
                t2g.parse()
                t2g.save(path + '/' + file[:-4] + '.graph')

                print('Density Score: %.4f' % t2g.density_score())
