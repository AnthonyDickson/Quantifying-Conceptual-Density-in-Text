from statistics import mean
from xml.etree import ElementTree

import plac
import spacy


@plac.annotations(
    filepath=plac.Annotation('The XML file to calculate the average sentence length for.', type=str)
)
def main(filepath):
    tree = ElementTree.parse(filepath)
    root = tree.getroot()

    nlp = spacy.load('en')

    sentence_lengths = []

    for section in root.findall('section'):
        section_text = section.find('text').text
        span = nlp(section_text)

        for sent in span.sents:
            sentence_lengths.append(len(sent))

    print('Mean Sentence Length: %.2f' % mean(sentence_lengths))


if __name__ == '__main__':
    plac.call(main)
