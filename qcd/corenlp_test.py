from stanza.nlp.corenlp import CoreNLPClient

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
                           default_annotators=['ssplit', 'tokenize', 'pos', 'parse', 'depparse', 'openie'])

    ann = client.annotate(text)

    for sentence in ann.sentence:
        for triple in sentence.openieTriple:
            print((triple.subject, triple.relation, triple.object))
