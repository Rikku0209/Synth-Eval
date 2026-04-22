import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_data(path):
    return pd.read_csv(path)

def handle_missing(df):
    df = df.replace(" ?", pd.NA)
    return df.dropna()

def encode_data(df):
    le = LabelEncoder()
    for col in df.select_dtypes(include='object'):
        df[col] = le.fit_transform(df[col])
    return df

def preprocess(path):
    df = load_data(path)
    df = handle_missing(df)
    df = encode_data(df)
    return df

def save_processed(df, path):
    df.to_csv(path, index=False)