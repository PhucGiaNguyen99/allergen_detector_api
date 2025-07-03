import unittest
from allergen_detector.detector import AllergenDetector
from nltk.corpus import wordnet

class TestAllergenDetector(unittest.TestCase):
    def setUp(self):
        self.detector = AllergenDetector()

    def test_get_wordnet_synsets(self):
        synsets = self.detector.get_wordnet_synsets("peanut")
        self.assertIsInstance(synsets, list)
        self.assertGreater(len(synsets), 0)
        self.assertTrue(any("peanut" in s.name() for s in synsets))

    def test_extract_lemmas_from_synsets(self):
        synsets = wordnet.synsets("milk")
        lemmas = self.detector.extract_lemmas_from_synsets(synsets)
        self.assertIn("milk", lemmas)
        self.assertIsInstance(lemmas, set)
        self.assertGreater(len(lemmas), 0)

    def test_get_synonyms(self):
        synonyms = self.detector.get_synonyms("egg")
        self.assertIn("egg", synonyms)
        self.assertIsInstance(synonyms, set)

    def test_match_allergens_exact(self):
        ingredients = ["This product contains peanuts and eggs."]
        allergens = ["peanut", "egg", "milk"]
        result = self.detector.match_allergens(ingredients, allergens)
        self.assertIn("peanut", result)
        self.assertIn("egg", result)
        self.assertNotIn("milk", result)

    def test_match_allergens_fuzzy(self):
        ingredients = ["Contains milkk and egs."]
        allergens = ["milk", "egg"]
        result = self.detector.match_allergens(ingredients, allergens)
        self.assertIn("milk", result)
        self.assertIn("egg", result)

    def test_match_allergens_synonyms(self):
        ingredients = ["This has groundnuts in it."]
        allergens = ["peanut"]
        result = self.detector.match_allergens(ingredients, allergens)
        self.assertIn("peanut", result)

if __name__ == '__main__':
    unittest.main()
