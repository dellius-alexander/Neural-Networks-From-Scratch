import pandas as pd


# Normalize the dataset
def normalize_data(data: pd.DataFrame):
    return (data - data.mean()) / data.std()
