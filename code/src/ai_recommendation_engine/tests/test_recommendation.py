import unittest
from src.recommendation_engine import setup_llm, build_recommendation_chain

class TestRecommendationEngine(unittest.TestCase):
    def test_llm_setup(self):
        llm = setup_llm("dummy_key")
        self.assertIsNotNone(llm)

    def test_chain_creation(self):
        llm = setup_llm("dummy_key")
        chain = build_recommendation_chain(llm)
        self.assertIsNotNone(chain)

if __name__ == "__main__":
    unittest.main()
