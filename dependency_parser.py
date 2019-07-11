import argparse
import xml.etree.ElementTree as ET

import neuralcoref
import nltk
import spacy


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

    if args.file.endswith('xml'):
        tree = ET.parse(args.file)
        root = tree.getroot()

        sentences = []

        for section in root.findall('section'):
            for text in section.findall('text'):
                sentences += nltk.sent_tokenize(text.text)

        sentences = map(lambda s: s.strip(), sentences)
        sentences = filter(lambda s: len(s) > 0, sentences)
        sentences = list(sentences)
        text = ' '.join(sentences)
    else:
        with open(args.file, 'r') as f:
            text = f.read()

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

    spans = list(doc.noun_chunks)
    spans = filter_spans(spans)

    with doc.retokenize() as retokenizer:
        for span in spans:
            retokenizer.merge(span)

    for token in doc:
        print('TOKEN:', token,
              '\n\tDEP:', token.dep_,
              '\n\tHEAD:', token.head,
              '\n\tLEFTS:', list(token.lefts),
              '\n\tRIGHTS:', list(token.rights))


    def append_rights(token, a, stop_at=None):
        for right in token.rights:
            if right.dep_ == stop_at:
                break

            a.append(right)

            append_rights(right, a)

    relations = []

    for sent in doc.sents:
        for token in sent:
            if token.dep_ == 'ROOT':
                subject = [w for w in token.lefts if w.dep_.startswith('nsubj')]
                subject = subject[0] if subject else ''

                verb = token
                found = False

                for attr in [w for w in token.rights if w.dep_ == 'attr']:
                    obj = [attr]
                    append_rights(attr, obj, stop_at='acl')

                    relations.append((subject, verb, obj))
                    found = True

                    for acl in [w for w in attr.rights if w.dep_ == 'acl']:
                        acl_verb = [acl]
                        acl_obj = []

                        for adp in [w for w in acl.rights if w.dep_ == 'prep']:
                            acl_verb.append(adp)

                            acl_obj = []
                            append_rights(adp, acl_obj)

                            relations.append((subject, acl_verb, acl_obj))
                            found = True

                if not found:
                    relations.append((subject, verb))

            elif token.dep_ == 'appos':
                dobj = None

                for token_ in sent:
                    if token_.dep_.endswith('obj'):
                        dobj = token_
                        break
                else:
                    continue

                rights = []
                append_rights(token, rights)
                relations.append((dobj, [token] + rights))

    # return relations
    print(relations)

    # relations = extract_currency_relations(doc)
    # for r1, r2 in relations:
    #     print("{:<10}\t{}".format(r1.text, r2.text))

    # displacy.serve(list(doc.sents))
