def generate_synthetic(df):
    # simple bootstrap method
    synthetic_data = df.sample(frac=1, replace=True, random_state=42)
    return synthetic_data