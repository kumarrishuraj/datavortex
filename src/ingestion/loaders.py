"""Raw file loading.

Everything is read as string with NA detection disabled. That matters: pandas
would otherwise silently convert 'NA' in settlement_account and 'nan'-looking
tokens into NaN before we get a chance to classify them, and would coerce
zero-padded MCCs like '05411' into integers, destroying exactly the damage we
are supposed to measure.

data/raw/ is treated as immutable — nothing in this package ever writes there.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.cleaning.chargebacks import load_chargebacks
from src.config import RAW_FILES


def _read_csv_as_text(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[])


def load_raw() -> dict[str, pd.DataFrame]:
    """Load all four source files exactly as they sit on disk."""
    missing = [str(p) for p in RAW_FILES.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing raw input(s): " + ", ".join(missing) + "\nPlace the organisers' Track 1 files in data/raw/ — see data/raw/README.md."
        )
    return {
        "transactions": _read_csv_as_text(RAW_FILES["transactions"]),
        "kyc": _read_csv_as_text(RAW_FILES["kyc"]),
        "merchants": _read_csv_as_text(RAW_FILES["merchants"]),
        "chargebacks": load_chargebacks(RAW_FILES["chargebacks"]),
    }


def raw_profile(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Row/column/size inventory of the raw layer, for the report."""
    rows = []
    for name, df in frames.items():
        path = RAW_FILES[name]
        rows.append(
            {
                "dataset": name,
                "file": path.name,
                "rows": len(df),
                "columns": df.shape[1],
                "size_mb": round(path.stat().st_size / 1024 / 1024, 2),
                "exact_duplicate_rows": int(df.duplicated().sum()),
            }
        )
    return pd.DataFrame(rows)
