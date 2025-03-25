import streamlit as st
import pandas as pd
import configparser
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import json
from huggingface_hub import login
from langchain_community.llms import HuggingFaceHub
import os
import PyPDF2, docx, tempfile, soundfile as sf, av, numpy as np
from transformers import pipeline
from openai import OpenAI
from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase

# --- Configuration ---
config = configparser.ConfigParser()
config.read('config.ini')
hf_api_key = config['huggingface']['access_token']
openai_api_key = config['openai']['api_key']

# HuggingFace login
login(hf_api_key)
os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_api_key

# HuggingFace setup
llm = HuggingFaceHub(
    repo_id="mistralai/Mistral-7B-Instruct-v0.1",
    model_kwargs={"temperature": 0.7, "max_new_tokens": 512}
)

# OpenAI setup
openai_client = OpenAI(api_key=openai_api_key)

# Data Load
customer_profiles = pd.read_csv("customer_profiles.csv")
transactions = pd.read_csv("transaction_history.csv")
social_media = pd.read_csv("social_media_activity.csv")
demographics = pd.read_csv("demographic_details.csv")

# Template for recommendations
base_template = """
Demographics:
{demographics}

Recent Transactions:
{transactions}

Social Media Activities:
{social_media}

{category_prompt}

Output format:
- **Recommendation:** 
- **Reason for Recommendation:**
- Provide real-world examples and HTTP URLs.
"""

prompt = PromptTemplate.from_template(base_template)

recommendation_chain = (
        {
            "demographics": RunnablePassthrough(),
            "transactions": RunnablePassthrough(),
            "social_media": RunnablePassthrough(),
            "category_prompt": RunnablePassthrough()
        } | prompt | llm | StrOutputParser()
)

# Streamlit App Setup
st.set_page_config(page_title="✨ AI Recommendation & Chatbot ✨", layout="wide")
st.title("✨ AI-Powered Recommendations & Chatbot ✨")

tab1, tab2, tab3 = st.tabs(["🔎 Existing Customer", "🆕 New Customer", "🤖 Smart Chatbot"])

# Category dictionary
categories = {
    "Adaptive Recommendation Engine": "Suggest adaptive products based on behavior shift.",
    "AI-Generated Personalized Suggestions": "Suggest 2 personalized products/services.",
    "Sentiment-Driven Content": "Content suggestions based on social media sentiment.",
    "Predictive Insights & Retention": "Predict churn and retention strategies.",
    "Multi-Modal Personalization": "Personalization using multi-modal input.",
    "Financial Recommendations": "Suggest financial products."
}

# --- Existing Customer Tab ---
with tab1:
    st.header("🔍 Existing Customer Recommendations")
    customer_id = st.selectbox("Select Customer ID:", customer_profiles['Customer_Id'].unique())

    demo = demographics[demographics['Customer_Id'] == customer_id].iloc[0]
    customer_trans = transactions[transactions['Customer_Id'] == customer_id]['Category'].tolist()
    customer_social = social_media[social_media['Customer_Id'] == customer_id]['Content'].tolist()

    st.subheader("Customer Information")
    st.write("**Demographics:**", demo.drop("Customer_Id").to_dict())
    st.write("**Transactions:**", customer_trans)
    st.write("**Social Media:**", customer_social)

    category_choice = st.selectbox("Choose Category:", list(categories.keys()), key='existing')
    if st.button("Generate Existing Customer Recommendation"):
        with st.spinner("Generating..."):
            response = recommendation_chain.invoke({
                "demographics": demo.drop("Customer_Id").to_dict(),
                "transactions": ", ".join(customer_trans),
                "social_media": ", ".join(customer_social),
                "category_prompt": categories[category_choice]
            })
            st.markdown(response)

# --- New Customer Tab ---
with tab2:
    st.header("🆕 New Customer Recommendations")

    demo_input = st.text_area("Demographics (JSON format)", '{"Location":"Urban","Marital Status":"Single","Dependents":0}')
    transactions_input = st.text_area("Recent Transactions (comma-separated)", "Electronics, Fitness")
    social_input = st.text_area("Social Media Activities (comma-separated)", "Interest in healthy lifestyle")

    try:
        demo_dict = json.loads(demo_input)
    except json.JSONDecodeError:
        st.error("Invalid JSON format for demographics!")
        st.stop()

    category_choice_new = st.selectbox("Choose Category:", list(categories.keys()), key='new')
    if st.button("Generate New Customer Recommendation"):
        with st.spinner("Generating..."):
            response_new = recommendation_chain.invoke({
                "demographics": demo_dict,
                "transactions": transactions_input,
                "social_media": social_input,
                "category_prompt": categories[category_choice_new]
            })
            st.markdown(response_new)

# --- Smart Chatbot Tab ---
with tab3:
    st.header("🤖 Smart Chatbot")

    class MyAudioProcessor(AudioProcessorBase):
        def __init__(self):
            self.frames = []

        def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
            self.frames.append(frame.to_ndarray())
            return frame

    def extract_text(uploaded_file):
        if uploaded_file.name.endswith('.pdf'):
            reader = PyPDF2.PdfReader(uploaded_file)
            return "\n".join([page.extract_text() for page in reader.pages])
        elif uploaded_file.name.endswith('.docx'):
            doc = docx.Document(uploaded_file)
            return "\n".join([para.text for para in doc.paragraphs])
        else:
            return uploaded_file.read().decode()

    text_prompt = st.text_area("Text Prompt:")
    file_upload = st.file_uploader("Upload File (pdf, docx, txt):", type=["pdf", "docx", "txt"])

    st.markdown("**🎙️ Record Audio:**")
    ctx = webrtc_streamer(key="audio-chatbot", mode=WebRtcMode.SENDONLY,
                          audio_processor_factory=MyAudioProcessor,
                          media_stream_constraints={"audio": True})

    chosen_model = st.radio("Model:", ["OpenAI GPT-3.5", "Hugging Face Mistral"])

    if st.button("Generate Chatbot Response"):
        context = text_prompt
        if file_upload:
            context += "\n\n" + extract_text(file_upload)


        st.subheader("🤖 Response:")
        if chosen_model == "OpenAI GPT-3.5":
            response = openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": context}],
                temperature=0.7
            )
            st.write(response.choices[0].message.content)
        else:
            hf_pipe = pipeline("text-generation", model="mistralai/Mistral-7B-Instruct-v0.1")
            result = hf_pipe(context, max_new_tokens=500)[0]['generated_text']
            st.write(result)

