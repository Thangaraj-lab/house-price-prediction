import pandas as pd

URL = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"

def load_data():
    return pd.read_csv(URL)
