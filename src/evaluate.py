"""Evaluation script: loads a checkpoint and reports full classification metrics."""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from torch.utils.data import DataLoader

from dataset import load_adult_income, preprocess, build_loaders
from model import MLP


def _collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
):
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for X, y in loader:
            logits = model(X.to(device)).cpu()
            all_logits.append(logits)
            all_labels.append(y)
    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels).numpy()
    probs = torch.softmax(logits, dim=1).numpy()
    preds = logits.argmax(1).numpy()
    return preds, probs, labels


def evaluate(checkpoint_path: str, config: dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load data ──────────────────────────────────────────────────────────────
    df, target_col = load_adult_income()
    dcfg = config["data"]
    train_ds, val_ds, test_ds, input_dim, num_classes = preprocess(
        df, target_col,
        test_size=dcfg["test_size"],
        val_size=dcfg["val_size"],
        random_seed=dcfg["random_seed"],
    )
    tcfg = config["training"]
    _, _, test_loader = build_loaders(
        train_ds, val_ds, test_ds, batch_size=tcfg["batch_size"]
    )

    # ── Load model ─────────────────────────────────────────────────────────────
    mcfg = config["model"]
    model = MLP(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dims=mcfg["hidden_dims"],
        dropout=mcfg["dropout"],
        batch_norm=mcfg["batch_norm"],
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state"])
    epoch = ckpt.get("epoch", "unknown")
    print(f"Loaded checkpoint from epoch {epoch}: {checkpoint_path}\n")

    # ── Metrics ────────────────────────────────────────────────────────────────
    preds, probs, labels = _collect_predictions(model, test_loader, device)

    acc = accuracy_score(labels, preds)
    print(f"Accuracy : {acc:.4f}")

    # ROC-AUC: binary → single column; multiclass → macro OvR
    if num_classes == 2:
        auc = roc_auc_score(labels, probs[:, 1])
    else:
        auc = roc_auc_score(labels, probs, multi_class="ovr", average="macro")
    print(f"ROC-AUC  : {auc:.4f}\n")

    print("Classification Report:")
    print(classification_report(labels, preds, digits=4))

    cm = confusion_matrix(labels, preds)
    print("Confusion Matrix:")
    print(cm)

    return {"accuracy": acc, "roc_auc": auc, "confusion_matrix": cm.tolist()}


def _parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a saved MLP checkpoint")
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--config", default="configs/default.yaml")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)
    evaluate(args.checkpoint, config)
