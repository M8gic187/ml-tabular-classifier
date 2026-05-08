# ml-tabular-classifier

A clean, production-ready PyTorch pipeline for tabular data classification.

## Features

- Configurable MLP architecture with batch normalization and dropout
- Robust data pipeline with train/val/test splits and feature scaling
- Training loop with early stopping and learning rate scheduling
- Comprehensive evaluation metrics (accuracy, F1, ROC-AUC, confusion matrix)
- Model checkpointing (best val loss and final epoch)
- **Stratified K-Fold cross-validation** with per-fold metrics and JSON export

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
│   └── cross_validate.py   # Stratified K-Fold cross-validation
├── configs/
│   ├── default.yaml        # Default hyperparameters and paths
│   └── cv_quick.yaml       # 3-fold / 30-epoch preset for fast runs
├── data/               # Raw and processed data
├── checkpoints/        # Saved model weights
└── requirements.txt
```
