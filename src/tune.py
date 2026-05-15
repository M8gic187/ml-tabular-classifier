"""Optuna-based hyperparameter tuning for the tabular MLP pipeline."""

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import torch
import torch.nn as nn
import yaml
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from dataset import TabularDataset, build_loaders, load_dataset
from model import MLP
from train import _epoch


optuna.logging.set_verbosity(optuna.logging.WARNING)


def _build_search_space(trial: optuna.Trial, search_cfg: dict) -> dict:
    """Sample one set of hyperparameters from the configured search space."""
    n_layers = trial.suggest_int(
        "n_layers",
        search_cfg["n_layers"]["min"],
        search_cfg["n_layers"]["max"],
    )
    hidden_dims = [
        trial.suggest_categorical(
            f"hidden_dim_{i}",
            search_cfg["hidden_dim_choices"],
        )
        for i in range(n_layers)
    ]
    return {
        "hidden_dims": hidden_dims,
        "dropout": trial.suggest_float(
            "dropout",
            search_cfg["dropout"]["min"],
            search_cfg["dropout"]["max"],
        ),
        "batch_norm": trial.suggest_categorical(
            "batch_norm", search_cfg["batch_norm_choices"]
        ),
        "learning_rate": trial.suggest_float(
            "learning_rate",
            search_cfg["learning_rate"]["min"],
            search_cfg["learning_rate"]["max"],
            log=True,
        ),
        "weight_decay": trial.suggest_float(
            "weight_decay",
            search_cfg["weight_decay"]["min"],
            search_cfg["weight_decay"]["max"],
            log=True,
        ),
        "batch_size": trial.suggest_categorical(
            "batch_size", search_cfg["batch_size_choices"]
        ),
    }


def _objective(
    trial: optuna.Trial,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    base_config: dict,
    search_cfg: dict,
    device: torch.device,
) -> float:
    """Optuna objective: train one trial, return best val_loss."""
    hp = _build_search_space(trial, search_cfg)

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_v = scaler.transform(X_val)

    input_dim = X_tr.shape[1]
    num_classes = int(np.max(np.concatenate([y_train, y_val]))) + 1

    train_loader, val_loader, _ = build_loaders(
        TabularDataset(X_tr, y_train),
        TabularDataset(X_v, y_val),
        TabularDataset(X_v, y_val),
        batch_size=hp["batch_size"],
    )

    model = MLP(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dims=hp["hidden_dims"],
        dropout=hp["dropout"],
        batch_norm=hp["batch_norm"],
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=hp["learning_rate"],
        weight_decay=hp["weight_decay"],
    )

    tcfg = base_config["training"]
    max_epochs = search_cfg.get("max_epochs_per_trial", tcfg["epochs"])
    patience = search_cfg.get("early_stopping_patience", tcfg["early_stopping_patience"])

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, max_epochs + 1):
        _epoch(model, train_loader, criterion, optimizer, device)
        val_loss, _ = _epoch(model, val_loader, criterion, None, device)

        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    return best_val_loss


def run_tuning(config: dict) -> dict[str, Any]:
    """
    Run Optuna hyperparameter search.

    Reads `tuning` section from config to define search space and trial budget.
    Returns a dict with the best hyperparameters and study summary.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load & encode data ────────────────────────────────────────────────────
    dcfg = config["data"]
    df, target_col, source = load_dataset(dcfg)
    print(f"Data source: {source}  |  {len(df):,} samples")

    df = df.copy()
    le = LabelEncoder()
    df[target_col] = le.fit_transform(df[target_col].astype(str))
    for col in df.select_dtypes(include=["object", "category"]).columns:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    X = df.drop(columns=[target_col]).values.astype("float32")
    y = df[target_col].values.astype("int64")

    seed = dcfg.get("random_seed", 42)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=dcfg.get("val_size", 0.15),
        random_state=seed, stratify=y,
    )

    # ── Optuna study ──────────────────────────────────────────────────────────
    tcfg = config["tuning"]
    n_trials = tcfg.get("n_trials", 20)
    timeout = tcfg.get("timeout_seconds", None)
    search_cfg = tcfg["search_space"]

    sampler = TPESampler(seed=seed)
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=5)
    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
    )

    print(f"Starting Optuna search: {n_trials} trials")
    t0 = time.time()

    study.optimize(
        lambda trial: _objective(
            trial, X_train, y_train, X_val, y_val,
            config, search_cfg, device,
        ),
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=False,
    )

    elapsed = time.time() - t0
    best_trial = study.best_trial

    print(f"\n{'='*55}")
    print(f"  Tuning complete  |  {len(study.trials)} trials  |  {elapsed:.0f}s")
    print(f"  Best val_loss : {best_trial.value:.4f}")
    print(f"  Best params   :")
    for k, v in best_trial.params.items():
        print(f"    {k}: {v}")
    print(f"{'='*55}\n")

    # Reconstruct hidden_dims list from individual trial params
    n_layers = best_trial.params["n_layers"]
    best_hidden_dims = [best_trial.params[f"hidden_dim_{i}"] for i in range(n_layers)]

    best_hp = {
        "hidden_dims": best_hidden_dims,
        "dropout": best_trial.params["dropout"],
        "batch_norm": best_trial.params["batch_norm"],
        "learning_rate": best_trial.params["learning_rate"],
        "weight_decay": best_trial.params["weight_decay"],
        "batch_size": best_trial.params["batch_size"],
    }

    summary = {
        "best_val_loss": best_trial.value,
        "best_trial_number": best_trial.number,
        "best_hyperparameters": best_hp,
        "n_trials_completed": len(study.trials),
        "elapsed_seconds": elapsed,
        "all_trials": [
            {
                "number": t.number,
                "value": t.value,
                "state": str(t.state),
                "params": t.params,
            }
            for t in study.trials
        ],
    }

    return summary


def save_tuning_results(summary: dict[str, Any], output_path: str) -> None:
    """Write tuning results to a JSON file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Tuning results saved → {path}")


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Optuna hyperparameter tuning for tabular MLP classifier"
    )
    parser.add_argument(
        "--config", default="configs/tune.yaml",
        help="Path to YAML config with 'tuning' section (default: configs/tune.yaml)"
    )
    parser.add_argument(
        "--output", default="checkpoints/tune_results.json",
        help="Path to save JSON results (default: checkpoints/tune_results.json)"
    )
    parser.add_argument(
        "--trials", type=int, default=None,
        help="Override n_trials from config"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.trials is not None:
        config["tuning"]["n_trials"] = args.trials

    summary = run_tuning(config)
    save_tuning_results(summary, args.output)
