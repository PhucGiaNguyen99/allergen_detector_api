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
    
    def get_synonyms(self, word):
        """
        Full synonym pipeline: get WordNet synsets and extract all synonyms.
        """
        synsets = self.get_wordnet_synsets(word)
        return self.extract_lemmas_from_synsets(synsets)
    
    def match_allergens(self, ingredients, allergens):
        """
        Main logic for matching allergens in ingredient strings using:
        - Exact match
        - Fuzzy string match (e.g., 'milkk' matches 'milk')
        - Token-based fuzzy match
        - Synonym expansion
        """
        detected = set()


        # Build a map: allergen -> set of variants (synonyms + itself)
        allergen_map = {}
        for allergen in allergens:
            allergen_lower = allergen.lower()
            synonym_set = self.get_synonyms(allergen_lower)
            synonym_set.add(allergen_lower)  # include the original word
            allergen_map[allergen_lower] = synonym_set


        # Loop through each ingredient string
        for ingredient in ingredients:
            ingredient = ingredient.lower()
            doc = self.nlp(ingredient)  # Convert to a Doc object


            # Compare each allergen variant against the ingredient string
            for allergen, variants in allergen_map.items():
                for variant in variants:
                    # Exact substring match
                    if variant in ingredient:
                        detected.add(allergen)
                    # Fuzzy match on full string
                    elif fuzz.partial_ratio(variant, ingredient) > 85:
                        detected.add(allergen)
                    else:
                        # Fuzzy match at token (word) level
                        for token in doc:
                            if fuzz.ratio(variant, token.text) > 85:
                                detected.add(allergen)


        return list(detected)
