import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np

def load_and_preprocess(path):
    df = pd.read_csv(path)

    # Normalize column names
    df.columns = df.columns.str.strip().str.lower()

    if 'label' not in df.columns:
        raise ValueError("❌ Error: The dataset must contain a 'label' column.")

    # Clean Label column
    df['label'] = df['label'].astype(str).str.strip().str.lower()
    df['label'] = df['label'].apply(lambda x: 1 if x in ['benign', 'normal'] else -1)

    # Drop unused columns
    df.drop(['source ip', 'destination ip'], axis=1, errors='ignore', inplace=True)

    # Encode categorical columns
    for col in df.select_dtypes(include='object').columns:
        df[col] = LabelEncoder().fit_transform(df[col])

    # Separate features and labels
    y = df['label']
    X = df.drop('label', axis=1)

    # Remove rows with inf, -inf, or very large values
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X = X.apply(pd.to_numeric, errors='coerce')  # Convert problematic strings to NaN
    X.dropna(inplace=True)

    # Reset y to match cleaned X
    y = y.loc[X.index]

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler
