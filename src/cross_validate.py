"""Stratified K-Fold cross-validation engine for the tabular MLP pipeline."""

import time

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
