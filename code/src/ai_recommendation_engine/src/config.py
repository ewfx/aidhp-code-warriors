import configparser

def load_config(filepath="code/src/ai_recommendation_engine/config.ini"):
    config = configparser.ConfigParser()
    config.read(filepath)
    return config
# Logic to load configuration from config.ini