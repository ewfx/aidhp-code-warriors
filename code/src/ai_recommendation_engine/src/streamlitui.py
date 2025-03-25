import streamlit as st
import json

import pandas as pd
from langchain_openai import ChatOpenAI
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
from config import load_config
from data_loader import load_customer_data
from prompts.templates import base_template
from prompts.templates import categories
from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase

# Load configuration
config = load_config()
hf_api_key = config['huggingface']['access_token']
openai_api_key = config['openai']['api_key']

# OpenAI setup
openai_client = OpenAI(api_key=openai_api_key)
# Login to Hugging Face
login(hf_api_key)
os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_api_key

# Initialize Hugging Face LLM
#llm = HuggingFaceHub(
#    repo_id="mistralai/Mistral-7B-Instruct-v0.1",
#    model_kwargs={"temperature": 0.7, "max_new_tokens": 512}
#)


# Initialize ChatOpenAI Model
llm = ChatOpenAI(
    openai_api_key=openai_api_key,
    model="gpt-3.5-turbo",
    temperature=0.7
)

# Load data
profiles, transactions, social_media, demographics = load_customer_data()




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


# Set page configuration
st.set_page_config(page_title="Customer Management Page", layout="wide")

# Apply custom CSS styling for the page, sidebar, and title ribbon
st.markdown("""
    <style>
    /* General Page Styling */
    .main {
        background-color: #E0E0E0;
        border: 3px solid #4CAF50;
    }
    .title {
        font-size: 25px;
        text-align: left;
        font-weight: bold;
        margin-bottom: 15px;
    }
    .custombutton {
        background-color: #4CAF50; /* Green background */
        color: white; /* White text color */
        font-size: 18px; /* Increase font size */
        font-weight: bold;
        padding: 10px 20px; /* Add padding for better size */
        border: none; /* Remove border */
        border-radius: 5px; /* Rounded corners */
        cursor: pointer; /* Pointer cursor on hover */
        transition: background-color 0.3s ease; /* Smooth hover effect */
    }

    .custombutton:hover {
        background-color: #45A049; /* Darker green when hovered */
    }

    .custombutton:active {
        background-color: #388E3C; /* Even darker green on click */
    }
    /* Title ribbon styling - Full Width */
    .title-ribbon {
        background-color: #4CAF50; /* Green background */
        color: white; /* White text */
        font-weight: bold; /* Bold font */
        font-size: 25px; /* Larger font size */
        text-align: center; /* Center alignment */
        padding: 15px; /* Padding for the title */
        border-radius: 10px 10px 0 0; /* Rounded top corners */
        width: 100%; /* Full width */
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); /* Add a shadow for effect */
    }
    /* Section title styling */
    .section-title {
        color: #4CAF50;
        font-weight: bold;
        font-size: 20px;
        width: 100%;
        margin-bottom: 10px;
        margin-top: 15px;
    }
    /* Sidebar button styling */
    .sidebar-button {
        display: block;
        text-align: left;
        font-size: 16px;
        padding: 10px 20px;
        width: 100%; /* Full width */
        margin-bottom: 5px;
        border: none;
        border-radius: 5px;
        cursor: pointer;
    }
    .sidebar-button.active {
        background-color: #4CAF50; /* Green for active */
        color: white;
    }
    .sidebar-button.inactive {
        background-color: #F0F0F0; /* Light gray for inactive */
        color: black;
    }
    </style>
""", unsafe_allow_html=True)
st.markdown('<div class="title">✨ AI-Powered Personalized Recommendation Engine</div>', unsafe_allow_html=True)

# Sidebar State Management
if "selected_page" not in st.session_state:
    st.session_state["selected_page"] = "New Customer"


# Sidebar Buttons Helper Function
def render_sidebar_buttons():
    # Sidebar buttons logic
    pages = ["New Customer", "Existing Customer", "ChatBot"]
    for page in pages:
        is_active = st.session_state["selected_page"] == page
        button_style = f"""
        display: block;
        text-align: left;
        font-size: 25px;
        font-weight: bold;
        padding: 10px 20px;
        width: 100%; /* Full width */
        margin-bottom: 5px;
        border: none;
        border-radius: 5px;
        cursor: pointer;
        background-color: {'#4CAF50' if is_active else '#F0F0F0'}; /* Green for active, gray for inactive */
        color: {'white' if is_active else 'black'};
        """
        if st.sidebar.button(page):  # If button clicked
            st.session_state["selected_page"] = page
        st.sidebar.markdown(
            f"""<div style="{button_style}"></div>""",
            unsafe_allow_html=True
        )


# Render Sidebar Buttons
st.sidebar.title("Navigation Menu")
render_sidebar_buttons()

# Handle Navigation Logic
selected_page = st.session_state["selected_page"]

if selected_page == "New Customer":
    # Render the title ribbon after the Navigation Bar
    st.markdown('<div class="title-ribbon">Recommendations for New Customer</div>', unsafe_allow_html=True)

    # Layout for Customer Information, Transaction Info, and Social Media Info in a Single Row
    col1, col3, col4, col5 = st.columns([3, 1, 3, 1])  # Equal column distribution

    # Customer Information Section
    with col1:
        st.markdown('<div class="section-title">Customer Information</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)

        with col1:
            name = st.text_input("Name", placeholder="Enter name")
            age = st.number_input("Age", min_value=0, step=1, format="%d")
            gender = st.selectbox("Gender", options=["Male", "Female", "Other"])
            country = st.selectbox("Country", options=[
                "India", "Germany", "Canada", "Australia", "United Kingdom", "Japan", "United States"
            ])

        with col2:
            income = st.number_input("Income", min_value=0, step=1, format="%d")
            interest = st.text_input("Interest", placeholder="Enter interest")

            preferences = st.text_input("Preferences", placeholder="Enter preferences")

    # Collapsible Transaction Info Section
    with col4:
        st.markdown('<div class="section-title">Transaction Info</div>', unsafe_allow_html=True)
        with st.expander("Expand to Enter Transaction Details"):
            trans_input = st.text_area("Transaction Details", placeholder="Enter transaction details")
        st.markdown('<div class="section-title">Social Media Info</div>', unsafe_allow_html=True)
        with st.expander("Expand to Enter Social Media Activity"):
            social_input = st.text_area("Social Media Activity", placeholder="Enter social media activity")

    # Buttons and JSON Output
    col_btn, col_output = st.columns([1, 3])

    with col_btn:
        # generate_json_btn = st.button("Generate JSON", use_container_width=True)

        demo_dict = {
            "Name": name,
            "Age": age,
            "Country": country,
            "Gender": gender,
            "Income": income,
            "Interest": interest,
            "Preferences": preferences
        }
        st.markdown('<div class="section-title">Choose the Recommendation Type</div>', unsafe_allow_html=True)
        # Display the keys in a selectbox
        selected_category = st.selectbox("Select a Category:", options=list(categories.keys()))

        if st.button("Generate Recommendations"):
            st.success("Generating ...")
            if selected_category == "Multi-Modal Personalization":
                st.subheader("🧠 Chatbot Interaction Mode for Multi-Modal Personalization")
                user_text = st.text_input("💬 Describe your financial goals, lifestyle, or any preferences:",
                                          key="multi_new_input")
                if st.button("🤖 Generate Multi-Modal Response (New)"):
                    with st.spinner("Processing multi-modal input..."):
                        response_new = recommendation_chain.invoke({
                            "demographics": demo_dict,
                            "transactions": trans_input,
                            "social_media": social_input,
                            "category_prompt": f"Use the following user input: '{user_text}' and provide a chatbot-style response as per the category: {categories[selected_category]}"
                        })
                    st.markdown(response_new, unsafe_allow_html=True)
                    #st.markdown(response_new.split("PROMPT ENDED:")[1].strip(), unsafe_allow_html=True)
            else:

                with col_output:
                    with st.spinner("Generating recommendation..."):
                        recommendations_new = recommendation_chain.invoke({
                            "demographics": demo_dict,
                            "transactions": trans_input,
                            "social_media": social_input,
                            "category_prompt": categories[selected_category]
                        })
                    st.success("✅ Recommendation Generated!")
                    st.subheader(f"📌 {selected_category}")
                    #st.markdown(recommendations_new.split("PROMPT ENDED:")[1].strip(), unsafe_allow_html=True)
                    st.markdown(recommendations_new, unsafe_allow_html=True)

    # with col_output:
    #     if generate_json_btn:
    #         # Create a JSON object from the input fields
    #         data = {
    #             "Name": name,
    #             "Age": age,
    #             "Country": country,
    #             "Gender": gender,
    #             "Income": income,
    #             "Interest": interest,
    #             "Preferences": preferences
    #         }
    #         # Display the JSON on the screen
    #         st.markdown('<div class="section-title">Generated JSON</div>', unsafe_allow_html=True)
    #         st.json(data)



elif selected_page == "Existing Customer":

    st.markdown('<div class="title-ribbon">Recommendations for Existing  Customer</div>', unsafe_allow_html=True)
    customer_id = st.selectbox("Select Customer ID", customer_profiles['Customer_Id'].unique())

    # Fetch customer data
    demo = demographics[demographics['Customer_Id'] == customer_id].iloc[0]
    customer_trans = transactions[transactions['Customer_Id'] == customer_id][
        ['Purchase_Date', 'Category', 'Amount (In Dollars)']]
    customer_social = social_media[social_media['Customer_Id'] == customer_id][['Timestamp', 'Content']]

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("📋 Demographics")
        st.table(demo.drop("Customer_Id"))

    with col2:
        st.subheader("🛒 Recent Transactions")
        st.dataframe(customer_trans.reset_index(drop=True), height=200)

    with col3:
        st.subheader("📲 Social Media Activities")
        st.dataframe(customer_social.reset_index(drop=True), height=200)


    with col1:
        st.markdown('<div class="section-title" >Choose the Recommendation Type</div>', unsafe_allow_html=True)
        selected_category_new = st.selectbox("Select a Category", options=list(categories.keys()))
        if st.button("Generate Recommendations"):
            if selected_category_new == "Multi-Modal Personalization":
                st.markdown("---")
                st.subheader("🧠 Chatbot Interaction Mode for Multi-Modal Personalization")

                user_text = st.text_input("💬 Describe your financial goals, lifestyle, or any preferences:")
                if st.button("🤖 Generate Multi-Modal Response"):
                    with st.spinner("Processing multi-modal input..."):
                        response = recommendation_chain.invoke({
                            "demographics": demo.drop("Customer_Id").to_dict(),
                            "transactions": ", ".join(customer_trans['Category'].tolist()),
                            "social_media": ", ".join(customer_social['Content'].tolist()),
                            "category_prompt": f"Use the following user input: '{user_text}' and provide a chatbot-style response as per the category: {categories[selected_category_new]}"
                        })
                    st.markdown(response, unsafe_allow_html=True)
                    #st.markdown(response.split("PROMPT ENDED:")[1].strip(), unsafe_allow_html=True)
            else:
                with st.spinner("Generating recommendation..."):
                    recommendations = recommendation_chain.invoke({
                        "demographics": demo.drop("Customer_Id").to_dict(),
                        "transactions": ", ".join(customer_trans['Category'].tolist()),
                        "social_media": ", ".join(customer_social['Content'].tolist()),
                        "category_prompt": categories[selected_category_new]
                    })
                st.success("✅ Recommendation Generated!")
                st.subheader(f"📌 {selected_category_new}")
                st.markdown(recommendations, unsafe_allow_html=True)
                #st.markdown(recommendations.split("PROMPT ENDED:")[1].strip(), unsafe_allow_html=True)



elif selected_page == "ChatBot":

    st.markdown('<div class="section-title">Chat with a Bot</div>', unsafe_allow_html=True)


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


    text_prompt = st.text_area("", placeholder="Text Prompt")
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
