from typing import Dict, Any, Optional, Union
import pandas as pd
import numpy as np
import logging
import json
from pathlib import Path

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _safe_describe(df: pd.DataFrame) -> Dict[str, Any]:
    """Return describe() as JSON-serializable dict, handling non-numeric cols."""
    try:
        desc = df.describe(include="all").to_dict()
        # numpy types -> native python
        def convert(obj):
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            if isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        return convert(desc)
    except Exception as e:
        logger.exception("Failed to describe dataframe: %s", e)
        return {}


def data_parser_agent(
    df_or_path: Union[str, Path, pd.DataFrame],
    sample_n: int = 5,
    save_clean_path: Optional[Union[str, Path]] = None
) -> Dict[str, Any]:
    """
    Parse the dataset and return a structured summary.

    """
    # Load dataframe if path provided
    if isinstance(df_or_path, (str, Path)):
        path = Path(df_or_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        # Attempt CSV, fallback to excel
        try:
            df = pd.read_csv(path)
        except Exception:
            df = pd.read_excel(path)
    elif isinstance(df_or_path, pd.DataFrame):
        df = df_or_path.copy()
    else:
        raise ValueError("df_or_path must be path or pandas.DataFrame")

    # Basic cleaning: trim whitespace in object columns
    obj_cols = df.select_dtypes(include="object").columns
    for c in obj_cols:
        df[c] = df[c].astype(str).str.strip().replace({"nan": np.nan})

    num_rows = len(df)
    num_columns = len(df.columns)

    missing_counts = df.isnull().sum().to_dict()
    missing_percentage = {k: (v / num_rows * 100) if num_rows > 0 else 0.0 for k, v in missing_counts.items()}

    dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}

    # Numeric summary
    summary_stats = _safe_describe(df)

    # Small sample for quick preview
    try:
        sample_rows = df.head(sample_n).to_dict(orient="records")
    except Exception:
        sample_rows = []

    summary = {
        "num_rows": num_rows,
        "num_columns": num_columns,
        "columns": list(df.columns),
        "dtypes": dtypes,
        "missing_counts": missing_counts,
        "missing_percentage": missing_percentage,
        "summary_stats": summary_stats,
        "sample_rows": sample_rows,
    }

    # Optionally save cleaned dataframe
    if save_clean_path:
        out_path = Path(save_clean_path)
        df.to_csv(out_path, index=False)
        summary["cleaned_csv_path"] = str(out_path)

    logger.info("Data parsed: %d rows, %d columns", num_rows, num_columns)
    return summary


# Quick local test when running this file directly
if __name__ == "__main__":
    # Example usage for quick manual test
    import sys
    example_csv = sys.argv[1] if len(sys.argv) > 1 else None
    if not example_csv:
        print("Usage: python data_parser_agent.py <path_to_csv>")
        raise SystemExit(1)

    s = data_parser_agent(example_csv, sample_n=3, save_clean_path="cleaned_sample.csv")
    print(json.dumps(s, indent=2))

 