import unittest
from unittest.mock import patch, MagicMock
import streamlit as st
import pandas as pd
import sys
import os

# Add the directory containing ai-recommendation-main.py to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src', 'ai_recommendation_engine', 'src')))

class TestAIRecommendationMain(unittest.TestCase):
    @patch('streamlit.selectbox')
    def test_generate_recommendations(self, mock_st):
        # Mocking Streamlit components
        mock_st.selectbox.return_value = "Multi-Modal Personalization"
        mock_st.text_input.return_value = "I want to save for retirement"
        mock_st.button.side_effect = [True, False]

        # Mocking data
        demo = pd.DataFrame({'Customer_Id': [1], 'Age': [30]})
        customer_trans = pd.DataFrame({'Customer_Id': [1], 'Category': ['Groceries']})
        customer_social = pd.DataFrame({'Customer_Id': [1], 'Content': ['Posted a photo']})
        categories = {"Multi-Modal Personalization": "Personalized recommendations"}


if __name__ == '__main__':
    unittest.main()