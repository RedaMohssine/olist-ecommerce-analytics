"""Download the Olist dataset from Kaggle into Data/.

Dataset: https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce (public, ~43 MB zipped)

Usage (from the repository root):
    pip install kagglehub
    python scripts/download_data.py
"""

import shutil
from pathlib import Path

import kagglehub

DATA_DIR = Path(__file__).resolve().parent.parent / "Data"


def main():
    source = Path(kagglehub.dataset_download("olistbr/brazilian-ecommerce"))
    DATA_DIR.mkdir(exist_ok=True)
    for csv in sorted(source.glob("*.csv")):
        shutil.copy(csv, DATA_DIR / csv.name)
        print(f"{csv.name} -> Data/")


if __name__ == "__main__":
    main()
