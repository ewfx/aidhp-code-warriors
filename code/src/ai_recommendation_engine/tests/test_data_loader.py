from src.data_loader import load_customer_data
import unittest

class TestDataLoader(unittest.TestCase):
    def test_load_customer_data(self):
        profiles, transactions, social_media, demographics = load_customer_data()
        self.assertFalse(profiles.empty)
        self.assertFalse(transactions.empty)
        self.assertFalse(social_media.empty)
        self.assertFalse(demographics.empty)

if __name__ == "__main__":
    unittest.main()
