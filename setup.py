import nltk
import spacy

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

spacy.cli.download('en')
spacy.cli.validate()
