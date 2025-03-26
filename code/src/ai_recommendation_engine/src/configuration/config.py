import configparser
import os

def load_config():
    # Get the directory of the current file
    current_dir = os.path.dirname(__file__)
    # Construct the relative path to config.ini
    config_path = os.path.join(current_dir, '..','..', 'config.ini')

    config = configparser.ConfigParser()
    config.read(config_path)
    return config