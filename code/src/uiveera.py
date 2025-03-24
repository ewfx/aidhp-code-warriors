import streamlit as st
st.set_page_config(page_title="✨ AI Personalized Recommendations ✨", layout="wide")

import pandas as pd
import configparser
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import json
from huggingface_hub import login
from langchain_community.llms import HuggingFaceHub
import os

# Apply CSS for styling
st.markdown("""
    <style>
    .main {
        background: linear-gradient(to right, #f4f7f9, #e0eafc);
        color: #2c3e50;
        font-family: 'Segoe UI', sans-serif;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-size: 16px;
        padding: 0.6em 1.5em;
        border-radius: 10px;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #388E3C;
        transform: scale(1.02);
    }
    .stTextArea textarea, .stTextInput input {
        background-color: #f9f9f9;
        border-radius: 8px;
        padding: 10px;
        font-size: 16px;
    }
    .stRadio > div {
        background-color: #f0f4f7;
        padding: 10px;
        border-radius: 10px;
    }
    .stTable, .stDataFrame {
        background-color: #ffffff;
        border-radius: 10px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.05);
    }
    </style>
""", unsafe_allow_html=True)

# Read Hugging Face access token from config.ini
config = configparser.ConfigParser()
config.read('config.ini')
hf_api_key = config['huggingface']['access_token']

# Login to Hugging Face
login(hf_api_key)
os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_api_key

# Initialize Hugging Face LLM
llm = HuggingFaceHub(
    repo_id="mistralai/Mistral-7B-Instruct-v0.1",
    model_kwargs={"temperature": 0.7, "max_new_tokens": 512}
)

# Load datasets
customer_profiles = pd.read_csv("customer_profiles.csv")
transactions = pd.read_csv("transaction_history.csv")
social_media = pd.read_csv("social_media_activity.csv")
demographics = pd.read_csv("demographic_details.csv")

# Enhanced Prompt Template with formatting and links
base_template = """
Given the following detailed customer profile:

Demographic Information:
{demographics}

Recently Purchased Products:
{transactions}

Social Media Interests and Activities:
{social_media}

Please provide the following:
{category_prompt}

**Output Requirements:**
- Start the response with a bold heading: **Recommendation:** followed by **Reason for Recommendation:** (these headings should appear only once).
- Use bullet points to list items under each section.
- Highlight product names in **bold**.
- Provide specific, real-world **product or service names**.
- Include at least one **online source link (http URL)** for each product or content.
- Use **Markdown** formatting throughout.

PROMPT ENDED:
"""

prompt = PromptTemplate.from_template(base_template)

# LangChain Runnable Chain
recommendation_chain = (
        {
            "demographics": RunnablePassthrough(),
            "transactions": RunnablePassthrough(),
            "social_media": RunnablePassthrough(),
            "category_prompt": RunnablePassthrough()
        } | prompt | llm | StrOutputParser()
)

# Streamlit UI
st.title("✨ AI-Powered Personalized Recommendation Engine")

# (Rest of the logic continues — keeping your tabs, user interaction, and LLM invocation structure as is)
