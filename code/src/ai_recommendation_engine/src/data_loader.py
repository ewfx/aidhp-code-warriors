import pandas as pd

def load_customer_data():
    profiles = pd.read_csv("./../data/customer_profiles.csv")
    transactions = pd.read_csv("./../data/transaction_history.csv")
    social_media = pd.read_csv("./../data/social_media_activity.csv")
    demographics = pd.read_csv("./../data/demographic_details.csv")
    return profiles, transactions, social_media, demographics


