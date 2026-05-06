# ml-tabular-classifier

A clean, production-ready PyTorch pipeline for tabular data classification.

## Features

- Configurable MLP architecture with batch normalization and dropout
- Robust data pipeline with train/val/test splits and feature scaling
- Training loop with early stopping and learning rate scheduling
- Comprehensive evaluation metrics (accuracy, F1, ROC-AUC, confusion matrix)
- Model checkpointing (best val loss and final epoch)

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
│   ├── dataset.py      # Data loading and preprocessing
│   ├── model.py        # MLP model architecture
│   ├── train.py        # Training loop with early stopping
│   └── evaluate.py     # Evaluation and metrics reporting
├── configs/
│   └── default.yaml    # Hyperparameters and paths
├── data/               # Raw and processed data
├── checkpoints/        # Saved model weights
└── requirements.txt
```
