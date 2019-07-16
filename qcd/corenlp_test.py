import os

from stanfordnlp.server import CoreNLPClient

if __name__ == '__main__':
    # This should be set elsewhere, probably
    os.environ["CORENLP_HOME"] = "stanford-corenlp"

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
    print('starting up Java Stanford CoreNLP Server...')

    # set up the client
    with CoreNLPClient(annotators=['tokenize', 'ssplit', 'pos', 'parse', 'depparse', 'openie'],
                       timeout=30000, memory='2G') as client:
        # submit the request to the server
        ann = client.annotate(text)

        for sentence in ann.sentence:
            for triple in sentence.openieTriple:
                print((triple.subject, triple.relation, triple.object))
