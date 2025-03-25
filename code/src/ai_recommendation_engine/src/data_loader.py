import pandas as pd

def load_customer_data():
    customer_profiles = pd.read_csv("code/src/ai_recommendation_engine/data/customer_profiles.csv")
    transactions = pd.read_csv("code/src/ai_recommendation_engine/data/transaction_history.csv")
    social_media = pd.read_csv("code/src/ai_recommendation_engine/data/social_media_activity.csv")
    demographics = pd.read_csv("code/src/ai_recommendation_engine/data/demographic_details.csv")
    return customer_profiles, transactions, social_media, demographics


