"""
Project Configuration and Paths
"""

from pathlib import Path

# Project root (parent of src)
BASE_DIR: Path = Path(__file__).resolve().parents[1]

# Data and Model paths
DATA_DIR: Path = BASE_DIR / 'data'
DATA_FILE: Path = DATA_DIR / 'digital_marketing_campaign_cleaned.csv'

MODEL_DIR: Path = BASE_DIR / 'models'
BEST_MODEL_PATH: Path = MODEL_DIR / 'best_model.joblib'
FEATURES_PATH: Path = MODEL_DIR / 'feature_list.json'   # fixed name

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Target column (set exactly to digital marketing campaign dataset.csv column name)
TARGET_COL: str = 'Conversion'

# Train/Test & CV settings
TEST_SIZE: float = 0.2
RANDOM_STATE: int = 42
CV_FOLDS: int = 5

# Evaluation metrics
SCORING: str = 'f1'   # renamed to match your training.py import
SCORING_METRICS = ['accuracy', 'precision', 'recall', 'f1']

# Parallel jobs
N_JOBS: int = -1