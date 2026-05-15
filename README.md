# ml-tabular-classifier

A clean, production-ready PyTorch pipeline for tabular data classification.

## Features

- Configurable MLP architecture with batch normalization and dropout
- Robust data pipeline with train/val/test splits and feature scaling
- Training loop with early stopping and learning rate scheduling
- Comprehensive evaluation metrics (accuracy, F1, ROC-AUC, confusion matrix)
- Model checkpointing (best val loss and final epoch)
- **Stratified K-Fold cross-validation** with per-fold metrics and JSON export
- **Optuna hyperparameter tuning** with TPE sampler, median pruner, and JSON export

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Train on the default dataset (UCI Adult Income)
python src/train.py --config configs/default.yaml

# Evaluate a saved checkpoint
python src/evaluate.py --checkpoint checkpoints/best_model.pt
```

### K-Fold Cross-Validation

Cross-validation gives a more reliable performance estimate by training on multiple
held-out splits and reporting mean ± std across folds.

```bash
# 5-fold CV with default config (Adult Income dataset)
python src/cross_validate.py --config configs/default.yaml --folds 5

# Quick 3-fold run with reduced epochs for rapid experimentation
python src/cross_validate.py --config configs/cv_quick.yaml

# Save results to a custom path
python src/cross_validate.py --config configs/default.yaml --output results/cv.json
```

Results are printed as a summary table and saved to `checkpoints/cv_results.json`:

```json
{
  "folds": [
    {"val_loss": 0.3241, "val_acc": 0.8612, "elapsed_s": 14.3},
    ...
  ],
  "aggregate": {
    "val_acc": {"mean": 0.8598, "std": 0.0021, "min": 0.857, "max": 0.862},
    "val_loss": {"mean": 0.3318, "std": 0.0043, "min": 0.324, "max": 0.338}
  }
}
```

### Hyperparameter Tuning (Optuna)

Automated search over architecture and optimiser settings using
[Optuna](https://optuna.org) with the TPE sampler and median pruning.

```bash
# Run 30 trials with the default tune preset
python src/tune.py --config configs/tune.yaml

# Override the number of trials on the fly
python src/tune.py --config configs/tune.yaml --trials 50

# Save results to a custom path
python src/tune.py --config configs/tune.yaml --output results/tune.json
```

The search covers: number of hidden layers (1–4), layer widths, dropout,
batch normalisation, learning rate (log-uniform), weight decay (log-uniform),
and batch size. Each trial uses per-trial early stopping so the search
stays fast.

Results are saved to `checkpoints/tune_results.json`:

```json
{
  "best_val_loss": 0.3102,
  "best_trial_number": 17,
  "best_hyperparameters": {
    "hidden_dims": [256, 128],
    "dropout": 0.21,
    "batch_norm": true,
    "learning_rate": 0.00083,
    "weight_decay": 0.000012,
    "batch_size": 512
  },
  "n_trials_completed": 30,
  "elapsed_seconds": 412.7
}
```

Copy `best_hyperparameters` into your `default.yaml` `model` and `training`
sections, then re-train with `src/train.py` to get the fully-converged model.

### Train with your own CSV dataset

Set the `data` section in your config:

```yaml
data:
	source: csv
	csv_path: data/my_dataset.csv
	target_col: target
	test_size: 0.15
	val_size: 0.15
	random_seed: 42
```

Then run training with that config file:

```bash
python src/train.py --config configs/default.yaml
```

## Project Structure

```
ml-tabular-classifier/
├── src/
│   ├── dataset.py          # Data loading and preprocessing
│   ├── model.py            # MLP model architecture
│   ├── train.py            # Training loop with early stopping
│   ├── evaluate.py         # Evaluation and metrics reporting
│   ├── cross_validate.py   # Stratified K-Fold cross-validation
│   └── tune.py             # Optuna hyperparameter search
├── configs/
│   ├── default.yaml        # Default hyperparameters and paths
│   ├── cv_quick.yaml       # 3-fold / 30-epoch preset for fast runs
│   └── tune.yaml           # Hyperparameter search space definition
├── data/               # Raw and processed data
├── checkpoints/        # Saved model weights
└── requirements.txt
```
