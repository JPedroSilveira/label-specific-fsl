import pandas as pd
from src.config.general_config import DATASETS_PATH


def load_dataset(filename):
    df = pd.read_csv(DATASETS_PATH + filename)
    return df