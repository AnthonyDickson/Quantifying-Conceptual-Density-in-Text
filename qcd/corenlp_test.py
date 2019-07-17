from nltk.tree import Tree

from stanza.nlp.corenlp import CoreNLPClient as CoreNLPClientBase


class CoreNLPClient(CoreNLPClientBase):
    def annotate(self, text, properties=None):
        # Fix bug where AnnotatedDocument objects built from a ProtoBuf buffer would be missing the `parse` attribute.
        properties = {
            'annotators': ','.join(properties or client.default_annotators),
            'outputFormat': 'json'
        }

        return client._request(text, properties).json(strict=False)


if __name__ == '__main__':
    # example text
    print('---')
    print('input text')
    print('')

    # text = "Bread is a staple food prepared from a dough of wheat flour and water, usually by baking."
    with open('docs/bread.txt', 'r') as f:
        text = f.read()

    print(text)

    # set up the client
    print('---')
    print('connecting to CoreNLP Server...')
    client = CoreNLPClient(server='http://localhost:9000',
                           default_annotators="tokenize,ssplit,pos,lemma,parse,natlog,depparse,openie".split(','))

    annotation = client.annotate(text)

    for sentence in annotation['sentences']:
        Tree.fromstring(sentence['parse']).pretty_print()

        for triple in sentence['openie']:
            print((triple['subject'], triple['relation'], triple['object']))
