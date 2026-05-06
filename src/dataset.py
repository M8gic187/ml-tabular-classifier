"""Data loading, preprocessing, and PyTorch Dataset wrapper for tabular classification."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import torch
from torch.utils.data import Dataset, DataLoader


class TabularDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


def _normalize_tabular_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Apply consistent cleaning for tabular datasets across sources."""
    df = df.copy()

    # Normalize missing markers
    df.replace("?", np.nan, inplace=True)

    # Normalize object columns: trim whitespace and map empty strings to NaN
    obj_cols = df.select_dtypes(
        include=["object", "category"]).columns.tolist()
    for col in obj_cols:
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].replace("", np.nan)

    # Auto-convert mostly numeric object columns (e.g., Telco TotalCharges)
    for col in obj_cols:
        converted = pd.to_numeric(df[col], errors="coerce")
        non_nan_ratio = converted.notna().mean()
        if non_nan_ratio >= 0.90:
            df[col] = converted

    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def load_adult_income():
    """Fetch UCI Adult Income dataset from OpenML and return a cleaned DataFrame."""
    dataset = fetch_openml(name="adult", version=2,
                           as_frame=True, parser="auto")
    df = dataset.frame.copy()

    # Drop rows with missing values (encoded as '?')
    df.replace("?", np.nan, inplace=True)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df, dataset.target_names[0] if hasattr(dataset, "target_names") else "class"


def load_telco_churn(version: int = 1):
    """Fetch Telco Customer Churn dataset from OpenML and return (df, target_col)."""
    dataset = fetch_openml(
        name="Telco-Customer-Churn",
        version=version,
        as_frame=True,
        parser="auto",
    )
    df = dataset.frame.copy()

    df = _normalize_tabular_dataframe(df)

    target_col = dataset.target_names[0] if hasattr(
        dataset, "target_names") else "Churn"
    return df, target_col


def load_csv_dataset(
    csv_path: str,
    target_col: str,
    drop_columns: list[str] | None = None,
):
    """Load a tabular dataset from CSV and return (df, target_col)."""
    path = Path(csv_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    df = pd.read_csv(path)

    if drop_columns:
        missing = [col for col in drop_columns if col not in df.columns]
        if missing:
            raise ValueError(
                f"drop_columns contains unknown columns: {missing}. Available columns: {list(df.columns)}"
            )
        df = df.drop(columns=drop_columns)

    if target_col not in df.columns:
        raise ValueError(
            f"target_col '{target_col}' not found in CSV columns: {list(df.columns)}"
        )

    df = _normalize_tabular_dataframe(df)
    return df, target_col


def load_dataset(data_config: dict):
    """
    Load dataset based on config.

    Supported sources:
    - adult_income (default)
    - telco_churn (OpenML)
    - csv (requires csv_path and target_col)
    """
    source = data_config.get("source")
    dataset_name = data_config.get("dataset", "adult_income")

    if source is None:
        source = "adult_income" if dataset_name == "adult_income" else dataset_name

    if source == "adult_income":
        df, target_col = load_adult_income()
        return df, target_col, source

    if source == "telco_churn":
        version = int(data_config.get("openml_version", 1))
        df, target_col = load_telco_churn(version=version)
        return df, target_col, source

    if source == "csv":
        csv_path = data_config.get("csv_path")
        target_col = data_config.get("target_col")
        drop_columns = data_config.get("drop_columns", [])
        if not csv_path:
            raise ValueError(
                "For data.source='csv', please set data.csv_path in config")
        if not target_col:
            raise ValueError(
                "For data.source='csv', please set data.target_col in config")
        df, target_col = load_csv_dataset(
            csv_path=csv_path,
            target_col=target_col,
            drop_columns=drop_columns,
        )
        return df, target_col, source

    raise ValueError(
        f"Unsupported data source '{source}'. Use 'adult_income', 'telco_churn' or 'csv'."
    )


def preprocess(
    df: pd.DataFrame,
    target_col: str,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_seed: int = 42,
):
    """
    Encode categoricals, scale numerics, and split into train/val/test.

    Returns (train_loader, val_loader, test_loader, input_dim, num_classes).
    """
    le = LabelEncoder()
    df[target_col] = le.fit_transform(df[target_col].astype(str))

    # Encode remaining categorical columns
    cat_cols = df.select_dtypes(
        include=["object", "category"]).columns.tolist()
    for col in cat_cols:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    X = df.drop(columns=[target_col]).values.astype(np.float32)
    y = df[target_col].values.astype(np.int64)

    # First split off test set
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_seed, stratify=y
    )

    # Split train/val from the remaining data
    relative_val = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=relative_val,
        random_state=random_seed, stratify=y_trainval
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    input_dim = X_train.shape[1]
    num_classes = int(y.max()) + 1

    return (
        TabularDataset(X_train, y_train),
        TabularDataset(X_val, y_val),
        TabularDataset(X_test, y_test),
        input_dim,
        num_classes,
    )


def build_loaders(
    train_ds: TabularDataset,
    val_ds: TabularDataset,
    test_ds: TabularDataset,
    batch_size: int = 512,
    num_workers: int = 0,
):
    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size * 2, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size * 2, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    )
    return train_loader, val_loader, test_loader
