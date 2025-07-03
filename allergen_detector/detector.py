import spacy
from rapidfuzz import fuzz
from nltk.corpus import wordnet
import nltk

# Download the WordNet for symnonym lookup
nltk.download('wordnet')

class AllergenDetector:
    def __init__(self):
        # Load the small English spaCy model for tokenization and NLP
        self.nlp = spacy.load("en_core_web_sm")

    def get_wordnet_synsets(self, word):
        """
        Get all WordNet synsets (meanings) for a given word.
        """
        return wordnet.synsets(word)
    
    def extract_lemmas_from_synsets(self, synsets):
        """
        Extract lemma names from the given list of synsets.
        """
        lemmas = set()
        for syn in synsets:
            for lemma in syn.lemmas():
                name = lemma.name().lower().replace("_", " ")
                lemmas.add(name)
        return lemmas