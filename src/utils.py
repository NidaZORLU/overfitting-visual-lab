import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.shape[0] < 5:
        raise ValueError("CSV çok küçük görünüyor. En az birkaç satır olmalı.")
    return df

def build_preprocessor(df: pd.DataFrame, target_col: str, scaling: str, one_hot: bool):
    if target_col not in df.columns:
        raise ValueError(f"Target column bulunamadı: {target_col}")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    num_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    transformers = []

    # Numeric scaler
    scaler = None
    if scaling == "standard":
        scaler = StandardScaler()
    elif scaling == "minmax":
        scaler = MinMaxScaler()
    elif scaling == "none":
        scaler = "passthrough"
    else:
        raise ValueError("scaling: none | standard | minmax olmalı")

    transformers.append(("num", scaler, num_cols))

    # Categorical encoder
    if one_hot and len(cat_cols) > 0:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        transformers.append(("cat", ohe, cat_cols))
    else:
        # Kategorikler varsa ve one_hot kapalıysa hata verelim (bilerek)
        if len(cat_cols) > 0:
            raise ValueError("CSV'de kategorik kolonlar var ama One-Hot kapalı. Aç veya kategorikleri sayısallaştır.")
        # yoksa bir şey ekleme

    preprocessor = ColumnTransformer(transformers, remainder="drop")
    return X, y, preprocessor

def split_data(X, y, test_size: float, random_state: int = 42):
    if not (0.05 <= test_size <= 0.95):
        raise ValueError("Test split 0.05 ile 0.95 arasında olmalı.")
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y if y.nunique() <= 20 else None)
