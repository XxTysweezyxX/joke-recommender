from pathlib import Path

# Project root is the folder that contains "src"
ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
NOTEBOOKS_DIR = ROOT / "notebooks"