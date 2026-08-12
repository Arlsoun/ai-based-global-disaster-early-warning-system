from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "disaster_events.csv"
)


def load_disaster_data() -> pd.DataFrame:
    """Load the raw disaster-event dataset."""

    if not RAW_DATA_FILE.exists():
        raise FileNotFoundError(
            f"Disaster dataset not found: {RAW_DATA_FILE}"
        )

    return pd.read_csv(RAW_DATA_FILE)