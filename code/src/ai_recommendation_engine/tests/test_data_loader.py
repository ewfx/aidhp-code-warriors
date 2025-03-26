import unittest
import os
import pandas as pd
from code.src.ai_recommendation_engine.src.dataloader.data_loader import load_customer_data

class TestDataLoader(unittest.TestCase):

    def setUp(self):
        # Create sample data for testing
        self.test_dir = os.path.dirname(__file__)
        self.data_dir = os.path.join(self.test_dir, 'test_data')
        os.makedirs(self.data_dir, exist_ok=True)

        self.customer_profiles_path = os.path.join(self.data_dir, 'customer_profiles.csv')
        self.transactions_path = os.path.join(self.data_dir, 'transaction_history.csv')
        self.social_media_path = os.path.join(self.data_dir, 'social_media_activity.csv')
        self.demographics_path = os.path.join(self.data_dir, 'demographic_details.csv')

        # Create sample CSV files
        pd.DataFrame({'Customer_Id': [1, 2], 'Name': ['Alice', 'Bob']}).to_csv(self.customer_profiles_path, index=False)
        pd.DataFrame({'Customer_Id': [1, 2], 'Amount': [100, 200]}).to_csv(self.transactions_path, index=False)
        pd.DataFrame({'Customer_Id': [1, 2], 'Activity': ['Post', 'Comment']}).to_csv(self.social_media_path, index=False)
        pd.DataFrame({'Customer_Id': [1, 2], 'Age': [30, 40]}).to_csv(self.demographics_path, index=False)

    def tearDown(self):
        # Clean up the test data
        os.remove(self.customer_profiles_path)
        os.remove(self.transactions_path)
        os.remove(self.social_media_path)
        os.remove(self.demographics_path)
        os.rmdir(self.data_dir)

    def test_load_customer_data(self):
        # Test the load_customer_data function
        customer_profiles, transactions, social_media, demographics = load_customer_data()

        # Check if the data is loaded correctly
        self.assertIsNotNone(customer_profiles)
        self.assertIsNotNone(transactions)
        self.assertIsNotNone(social_media)
        self.assertIsNotNone(demographics)

if __name__ == '__main__':
    unittest.main()