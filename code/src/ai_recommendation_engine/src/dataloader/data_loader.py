import pandas as pd
import os

def load_customer_data():
    # Get the directory of the current file
    current_dir = os.path.dirname(__file__)
    # Construct the relative paths to the CSV files
    customer_profiles_path = os.path.join(current_dir, '..','..', 'data', 'customer_profiles.csv')
    transactions_path = os.path.join(current_dir, '..','..', 'data', 'transaction_history.csv')
    social_media_path = os.path.join(current_dir, '..', '..', 'data', 'social_media_activity.csv')
    demographics_path = os.path.join(current_dir, '..', '..','data', 'demographic_details.csv')

    customer_profiles = pd.read_csv(customer_profiles_path)
    transactions = pd.read_csv(transactions_path)
    social_media = pd.read_csv(social_media_path)
    demographics = pd.read_csv(demographics_path)
    return customer_profiles, transactions, social_media, demographics