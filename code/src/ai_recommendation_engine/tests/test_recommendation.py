import unittest
from unittest.mock import patch, MagicMock
import streamlit as st
import pandas as pd
import sys
import os

# Add the directory containing ai_recommendation_main.py to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

class TestAIRecommendationMain(unittest.TestCase):
    ai_recommendation_main = __import__('ai-recommendation-main')

    @patch('ai_recommendation_main.st')
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

        with patch('ai_recommendation_main.recommendation_chain.invoke') as mock_invoke:
            mock_invoke.return_value = "Recommended action: Save 20% of your income."

            # Run the function
            exec(open('ai-recommendation-main.py').read())

            # Assertions
            mock_st.selectbox.assert_called_once_with("Select a Category", options=list(categories.keys()))
            mock_st.text_input.assert_called_once_with("💬 Describe your financial goals, lifestyle, or any preferences:")
            mock_st.button.assert_any_call("Generate Recommendations")
            mock_st.button.assert_any_call("🤖 Generate Multi-Modal Response")
            mock_invoke.assert_called_once()

    @patch('ai_recommendation_main.st')
    def test_generate_chatbot_response(self, mock_st):
        # Mocking Streamlit components
        mock_st.radio.return_value = "OpenAI GPT-3.5"
        mock_st.text_area.return_value = "Tell me about investment options."
        mock_st.file_uploader.return_value = None
        mock_st.button.return_value = True

        with patch('ai_recommendation_main.openai_client.chat.completions.create') as mock_create:
            mock_create.return_value = MagicMock(choices=[MagicMock(message=MagicMock(content="Investment options include stocks, bonds, and mutual funds."))])

            # Run the function
            exec(open('ai-recommendation-main.py').read())

            # Assertions
            mock_st.radio.assert_called_once_with("Model:", ["OpenAI GPT-3.5", "Hugging Face Mistral"])
            mock_st.text_area.assert_called_once_with("Prompt your Text")
            mock_st.file_uploader.assert_called_once_with("Upload File (pdf, docx, txt):", type=["pdf", "docx", "txt"])
            mock_st.button.assert_called_once_with("Generate Chatbot Response")
            mock_create.assert_called_once()

if __name__ == '__main__':
    unittest.main()