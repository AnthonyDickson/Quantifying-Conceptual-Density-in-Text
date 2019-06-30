import argparse
import xml.etree.ElementTree as ET

import neuralcoref
import spacy
from spacy import displacy


def filter_spans(spans):
    # Filter a sequence of spans so they don't contain overlaps
    get_sort_key = lambda span: (span.end - span.start, span.start)
    sorted_spans = sorted(spans, key=get_sort_key, reverse=True)
    result = []
    seen_tokens = set()
    for span in sorted_spans:
        if span.start not in seen_tokens and span.end - 1 not in seen_tokens:
            result.append(span)
            seen_tokens.update(range(span.start, span.end))
    return result


def extract_currency_relations(doc):
    # Merge entities and noun chunks into one token
    seen_tokens = set()
    spans = list(doc.noun_chunks)
    spans = filter_spans(spans)
    with doc.retokenize() as retokenizer:
        for span in spans:
            retokenizer.merge(span)

    relations = []
    for token in doc:
        if token.dep_ in ("attr", "dobj", "conj"):
            subject = [w for w in token.head.lefts if w.dep_.startswith("nsubj")]
            if subject:
                subject = subject[0]
                relations.append((subject, token))
        elif token.dep_ == "pobj" and token.head.dep_ == "prep":
            relations.append((token.head.head, token))
    return relations


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parse a file and group the entities by section.')
    parser.add_argument('-f', '--file', metavar='INPUT_FILE', type=str, required=True,
                        help='The file to parse. Can be a `.xml` file.')
    args = parser.parse_args()

    with open(args.file, 'r') as f:
        text = f.read()

    if args.file.endswith('xml'):
        sentences = ET.fromstring(text).itertext()
        sentences = map(lambda s: s.strip(), sentences)
        sentences = filter(lambda s: len(s) > 0, sentences)
        sentences = list(sentences)
        print(sentences)
        text = ' '.join(sentences)

    text = text.lower()

    nlp = spacy.load('en')
    neuralcoref.add_to_pipe(nlp)
    doc = nlp(text)
    text = doc._.coref_resolved
    del doc
    del nlp

    print(text)

    nlp = spacy.load('en')
    doc = nlp(text)

    # print(doc)
    # for token in doc:
    #     print(token.text, token.dep_, token.head.text, token.head.pos_,
    #           [child for child in token.children])
    #
    # print(displacy.render(doc, style='dep'))
    # displacy.serve(doc, style='dep')

    relations = extract_currency_relations(doc)
    for r1, r2 in relations:
        print("{:<10}\t{}".format(r1.text, r2.text))

    displacy.serve(list(doc.sents))
