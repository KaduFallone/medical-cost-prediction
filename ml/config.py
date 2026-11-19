from pathlib import Path

# Root path to the project directory
ROOT_DIR = Path(__file__).resolve().parent.parent

# Path to the raw data, model, and reports directories
RAW_DATA_PATH = ROOT_DIR / "data" / "raw" / "insurance.csv"
MODEL_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"

# Target and training options
TARGET = "charges"
RANDOM_STATE = 42
TEST_SIZE = 0.2
