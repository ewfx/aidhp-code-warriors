import configparser

def load_config(filepath="./../config.ini"):
    config = configparser.ConfigParser()
    config.read(filepath)
    return config
# Logic to load configuration from config.ini