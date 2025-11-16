"""Configuration constants and paths."""
from pathlib import Path
from dataclasses import dataclass

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
DATASET_PATH = DATA_DIR / "dataset.csv"

# Global labels
LABELS = ["type_blocker", "type_bug", "type_documentation", "type_enhancement"]

# Model parameters
@dataclass
class CNNConfig:
    """Configuration for CNN model."""
    vocab_size: int = 5000
    embedding_dim: int = 100
    max_len: int = 100
    output_dim: int = 4

@dataclass
class MLPConfig:
    """Configuration for MLP model."""
    hidden_layer_1: int = 256
    hidden_layer_2: int = 128
    dropout_rate: float = 0.5
    output_dim: int = 4

@dataclass
class TrainingConfig:
    """Configuration for model training."""
    epochs: int = 100
    batch_size: int = 16
    validation_split: float = 0.2
    early_stopping_patience: int = 5
    n_cv_splits: int = 10
