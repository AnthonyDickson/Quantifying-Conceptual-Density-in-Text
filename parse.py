import argparse
import os
import pickle
import re

import nltk


class Text2Graph:
    def __init__(self, text):
        self.stopwords = set(nltk.corpus.stopwords.words('english'))
        self.hyphenated_word = re.compile(r'^[a-z]+([/-][a-z]+)+$')
        self.wnl = nltk.stem.WordNetLemmatizer()
        self.edges = {}
        self.text = text

    def is_valid_token(self, t):
        return (t.isalnum() or re.match(self.hyphenated_word, t)) and \
               t not in self.stopwords

    def add_token(self, token, context):
        for other in context:
            if other != token:
                if token not in self.edges:
                    self.edges[token] = {}

                if other not in self.edges[token]:
                    self.edges[token][other] = 0

                self.edges[token][other] += 1

    def remove_hyphenation(self, text):
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

    def parse(self):
        text = self.text.lower()
        text = self.remove_hyphenation(text)
        sentences = nltk.sent_tokenize(text)

        for sent in sentences:
            tokens = nltk.word_tokenize(sent)
            tokens = filter(self.is_valid_token, tokens)
            tokens = list(map(self.wnl.lemmatize, tokens))

            for token in tokens:
                self.add_token(token, tokens)

                if re.match(self.hyphenated_word, token):
                    for t in re.split(r'[-/]', token):
                        self.add_token(self.wnl.lemmatize(t), tokens)

    @property
    def mean_connectivity(self):
        N = len(self.edges)
        res = N

        for node in self.edges:
            res += sum(self.edges[node].values()) / N

        return res

    def save(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump((text, self.edges), f)

    @staticmethod
    def load(filepath):
        with open(filepath, 'rb') as f:
            text, edges = pickle.load(f)

        t2g = Text2Graph(text)
        t2g.edges = edges

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

                if len(title) > 16:
                    title = title[:13] + '...'

                print('Processing file \'%s\'...' % title)

                with open(path + '/' + file, 'r') as f:
                    text = f.read()

                t2g = Text2Graph(text)
                t2g.parse()
                t2g.save(path + '/' + file[:-4] + '.graph')

                print('Mean node connectivity for \'%s\': %.4f'
                      % (title, t2g.mean_connectivity))
