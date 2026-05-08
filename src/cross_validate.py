"""Stratified K-Fold cross-validation engine for the tabular MLP pipeline."""

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from dataset import TabularDataset, build_loaders
from model import MLP
from train import _epoch


def _build_model(
    input_dim: int,
    num_classes: int,
    model_config: dict,
    device: torch.device,
) -> MLP:
    return MLP(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dims=model_config["hidden_dims"],
        dropout=model_config["dropout"],
        batch_norm=model_config["batch_norm"],
    ).to(device)


def _train_fold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: dict,
    device: torch.device,
    fold_idx: int,
) -> dict:
    """Train a single fold and return the best validation metrics."""
    tcfg = config["training"]
    mcfg = config["model"]

    input_dim = X_train.shape[1]
    num_classes = int(np.max(np.concatenate([y_train, y_val]))) + 1

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    # Re-use X_val as a placeholder for the unused test split in build_loaders
    train_loader, val_loader, _ = build_loaders(
        TabularDataset(X_train_s, y_train),
        TabularDataset(X_val_s, y_val),
        TabularDataset(X_val_s, y_val),
        batch_size=tcfg["batch_size"],
    )

    model = _build_model(input_dim, num_classes, mcfg, device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=tcfg["learning_rate"],
        weight_decay=tcfg["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=tcfg["lr_scheduler_patience"],
        factor=tcfg["lr_scheduler_factor"],
    )

    best_val_loss = float("inf")
    best_val_acc = 0.0
    patience_counter = 0
    t0 = time.time()

    for epoch in range(1, tcfg["epochs"] + 1):
        _epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = _epoch(model, val_loader, criterion, None, device)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= tcfg["early_stopping_patience"]:
                break

    elapsed = time.time() - t0
    print(
        f"  Fold {fold_idx + 1} | best val_loss={best_val_loss:.4f}"
        f"  val_acc={best_val_acc:.4f} | {elapsed:.1f}s"
    )
    return {"val_loss": best_val_loss, "val_acc": best_val_acc, "elapsed_s": elapsed}


def run_kfold(
    X: np.ndarray,
    y: np.ndarray,
    config: dict,
    n_splits: int = 5,
    random_seed: int = 42,
) -> list[dict]:
    """
    Run stratified K-fold cross-validation.

    Args:
        X: Feature matrix (float32, already label-encoded but NOT yet scaled).
        y: Integer class labels.
        config: Full training config dict (same schema as default.yaml).
        n_splits: Number of folds.
        random_seed: Seed for fold splitting reproducibility.

    Returns:
        List of per-fold result dicts with keys val_loss, val_acc, elapsed_s.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  K-Fold CV  |  k={n_splits}")
    print(f"Dataset: {len(X):,} samples  |  {X.shape[1]} features  |  {len(np.unique(y))} classes\n")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    fold_results: list[dict] = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(
            f"[Fold {fold_idx + 1}/{n_splits}]"
            f"  train={len(train_idx):,}  val={len(val_idx):,}"
        )
        result = _train_fold(
            X[train_idx], y[train_idx],
            X[val_idx], y[val_idx],
            config, device, fold_idx,
        )
        fold_results.append(result)

    return fold_results


def aggregate_fold_results(fold_results: list[dict]) -> dict[str, Any]:
    """Compute mean ± std summary statistics across all folds."""
    metrics = ["val_loss", "val_acc", "elapsed_s"]
    summary: dict[str, Any] = {"folds": fold_results, "aggregate": {}}

    for metric in metrics:
        values = [r[metric] for r in fold_results]
        summary["aggregate"][metric] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }

    acc_mean = summary["aggregate"]["val_acc"]["mean"]
    acc_std = summary["aggregate"]["val_acc"]["std"]
    loss_mean = summary["aggregate"]["val_loss"]["mean"]
    print(f"\n{'='*55}")
    print(f"  CV Summary ({len(fold_results)} folds)")
    print(f"  val_acc  : {acc_mean:.4f} ± {acc_std:.4f}")
    print(f"  val_loss : {loss_mean:.4f}")
    print(f"{'='*55}")

    return summary


def save_cv_results(summary: dict[str, Any], output_path: str) -> None:
    """Write cross-validation summary to a JSON file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"CV results saved → {path}")


def _parse_args():
    parser = argparse.ArgumentParser(
        description="K-Fold cross-validation for tabular MLP classifier"
    )
    parser.add_argument("--config", default="configs/default.yaml",
                        help="Path to YAML config file")
    parser.add_argument("--folds", type=int, default=5,
                        help="Number of CV folds (default: 5)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for fold splitting")
    parser.add_argument("--output", default="checkpoints/cv_results.json",
                        help="Path to save JSON results")
    return parser.parse_args()


if __name__ == "__main__":
    import yaml
    from dataset import load_dataset, preprocess

    args = _parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)

    dcfg = config["data"]
    df, target_col, source = load_dataset(dcfg)
    print(f"Data source: {source}")

    # Preprocess to get encoded X and y without the train/val/test split
    from sklearn.preprocessing import LabelEncoder
    import pandas as pd

    df = df.copy()
    le = LabelEncoder()
    df[target_col] = le.fit_transform(df[target_col].astype(str))
    for col in df.select_dtypes(include=["object", "category"]).columns:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    X = df.drop(columns=[target_col]).values.astype("float32")
    y = df[target_col].values.astype("int64")

    fold_results = run_kfold(X, y, config, n_splits=args.folds, random_seed=args.seed)
    summary = aggregate_fold_results(fold_results)
    save_cv_results(summary, args.output)
