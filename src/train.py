"""Training loop with early stopping, LR scheduling, and checkpointing."""

import argparse
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import load_dataset, preprocess, build_loaders
from model import MLP, count_parameters


def _epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> tuple[float, float]:
    """Run one epoch; optimizer=None means eval mode."""
    training = optimizer is not None
    model.train(training)

    total_loss, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            loss = criterion(logits, y)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            total += len(y)

    return total_loss / total, correct / total


def train(config: dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Data ──────────────────────────────────────────────────────────────────
    dcfg = config["data"]
    df, target_col, source = load_dataset(dcfg)
    train_ds, val_ds, test_ds, input_dim, num_classes = preprocess(
        df, target_col,
        test_size=dcfg["test_size"],
        val_size=dcfg["val_size"],
        random_seed=dcfg["random_seed"],
    )
    print(f"Data source: {source}")
    tcfg = config["training"]
    train_loader, val_loader, _ = build_loaders(
        train_ds, val_ds, test_ds, batch_size=tcfg["batch_size"]
    )
    print(
        f"Train: {len(train_ds):,}  Val: {len(val_ds):,}  Test: {len(test_ds):,}")
    print(f"Input dim: {input_dim}  Classes: {num_classes}")

    # ── Model ─────────────────────────────────────────────────────────────────
    mcfg = config["model"]
    model = MLP(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dims=mcfg["hidden_dims"],
        dropout=mcfg["dropout"],
        batch_norm=mcfg["batch_norm"],
    ).to(device)
    print(f"Parameters: {count_parameters(model):,}")

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

    # ── Training loop with early stopping ────────────────────────────────────
    pcfg = config["paths"]
    Path(pcfg["checkpoint_dir"]).mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [],
               "train_acc": [], "val_acc": []}

    for epoch in range(1, tcfg["epochs"] + 1):
        t0 = time.time()
        train_loss, train_acc = _epoch(
            model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = _epoch(model, val_loader, criterion, None, device)

        scheduler.step(val_loss)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:3d}/{tcfg['epochs']} | "
            f"train_loss={train_loss:.4f} acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} acc={val_acc:.4f} | "
            f"lr={lr:.2e} | {elapsed:.1f}s"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(
                {"epoch": epoch, "model_state": model.state_dict(),
                 "val_loss": val_loss, "val_acc": val_acc},
                pcfg["best_model"],
            )
            print(f"  ✓ Saved best model (val_loss={val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= tcfg["early_stopping_patience"]:
                print(f"Early stopping triggered after {epoch} epochs.")
                break

    torch.save(
        {"epoch": epoch, "model_state": model.state_dict(), "history": history},
        pcfg["final_model"],
    )
    print(f"Training complete. Best val_loss={best_val_loss:.4f}")
    return model, history


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Train tabular MLP classifier")
    parser.add_argument("--config", default="configs/default.yaml")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)
    train(config)
