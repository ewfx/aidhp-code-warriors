import streamlit as st
import json
from config import load_config
from data_loader import load_customer_data
from recommendation.recommendation_engine import setup_llm, build_recommendation_chain
from utils import parse_json

# Load configuration
config = load_config()
hf_api_key = config['huggingface']['access_token']

# Load data
profiles, transactions, social_media, demographics = load_customer_data()

# Setup LLM and recommendation chain
llm = setup_llm(hf_api_key)
recommendation_chain = build_recommendation_chain(llm)

# Streamlit UI
st.set_page_config(page_title="✨ AI Personalized Recommendations ✨", layout="wide")
st.title("✨ AI-Powered Personalized Recommendation Engine")

tabs = st.tabs(["🔎 Existing Customer", "🆕 New Customer"])

categories = {
    "Adaptive Recommendation Engine": "Suggest a specific product or service that adapts to recent shifts in customer behavior with real-world examples.",
    "AI-Generated Personalized Product/Service Suggestions": "Recommend two highly relevant products/services with examples.",
    "Sentiment-Driven Content Recommendations": "Recommend content (videos, blogs, etc.) based on sentiment analysis of social media posts.",
    "Predictive Customer Insights & Business Strategies": "Predict churn risks and provide retention strategies.",
    "Multi-Modal Personalization": "Personalize suggestions interactively based on user input.",
    "Hyper-Personalized Financial Product Recommendations": "Recommend financial products based on transactions and risk profile."
}

# Existing Customer Tab
with tabs[0]:
    st.header("🔍 Existing Customer Profile")
    customer_id = st.selectbox("Select Customer ID:", profiles['Customer_Id'].unique())

    demo = demographics[demographics['Customer_Id'] == customer_id].iloc[0]
    customer_trans = transactions[transactions['Customer_Id'] == customer_id]
    customer_social = social_media[social_media['Customer_Id'] == customer_id]

    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("📋 Demographics")
        st.table(demo.drop("Customer_Id"))

    with col2:
        st.subheader("🛒 Recent Transactions")
        st.dataframe(customer_trans[['Purchase_Date', 'Category', 'Amount (In Dollars)']].reset_index(drop=True), height=200)

    with col3:
        st.subheader("📲 Social Media Activities")
        st.dataframe(customer_social[['Timestamp', 'Content']].reset_index(drop=True), height=200)

    option = st.selectbox("🚀 Choose Recommendation:", list(categories.keys()), key="existing_category")
    user_text = ""
    if option == "Multi-Modal Personalization":
        user_text = st.text_area("💬 Describe your financial goals or lifestyle:")

    if st.button("Generate Existing Customer Recommendation"):
        with st.spinner("Generating recommendation..."):
            response = recommendation_chain.invoke({
                "demographics": demo.drop("Customer_Id").to_dict(),
                "transactions": ", ".join(customer_trans['Category'].tolist()),
                "social_media": ", ".join(customer_social['Content'].tolist()),
                "category_prompt": f"{categories[option]} {'User input: ' + user_text if user_text else ''}"
            })
        st.success("✅ Recommendation Generated!")
        st.markdown(response.split("PROMPT ENDED:")[-1].strip(), unsafe_allow_html=True)

# New Customer Tab
with tabs[1]:
    st.header("✨ New Customer Information")
    col1, col2 = st.columns(2)

    with col1:
        demo_input = st.text_area("📋 Demographics (JSON format)", json.dumps({
            "Location": "Suburban",
            "Marital Status": "Married",
            "Dependents": 2,
            "Home Ownership": "With Family",
            "Nationality": "Indian"
        }, indent=4), height=200)

    with col2:
        trans_input = st.text_area("🛒 Transactions (comma-separated)", "Gucci, Mutual Funds, Supermarket")
        social_input = st.text_area("📲 Social Media Activities (comma-separated)", "Financial management concerns.")

    demo_dict, error = parse_json(demo_input)
    if error:
        st.error(f"Demographic JSON error: {error}")
        st.stop()

    option_new = st.selectbox("🚀 Choose Recommendation:", list(categories.keys()), key="new_category")
    user_text_new = ""
    if option_new == "Multi-Modal Personalization":
        user_text_new = st.text_area("💬 Describe your financial goals or lifestyle (New):")

    if st.button("Generate New Customer Recommendation"):
        with st.spinner("Generating recommendation..."):
            response_new = recommendation_chain.invoke({
                "demographics": demo_dict,
                "transactions": trans_input,
                "social_media": social_input,
                "category_prompt": f"{categories[option_new]} {'User input: ' + user_text_new if user_text_new else ''}"
            })
        st.success("✅ Recommendation Generated!")
        st.markdown(response_new.split("PROMPT ENDED:")[-1].strip(), unsafe_allow_html=True)
