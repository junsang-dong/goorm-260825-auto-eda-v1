"""Dataset profiling: shape, dtypes, missing values, duplicates, descriptive stats."""
from __future__ import annotations

import pandas as pd


def profile_dataset(df: pd.DataFrame) -> dict:
    n_rows, n_cols = df.shape
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = [c for c in df.columns if c not in numeric_cols]

    missing = df.isna().sum()
    missing_pct = (missing / max(n_rows, 1) * 100).round(2)
    missing_table = pd.DataFrame(
        {"missing_count": missing, "missing_pct": missing_pct}
    ).sort_values("missing_count", ascending=False)

    dup_count = int(df.duplicated().sum())

    constant_cols = [c for c in df.columns if df[c].nunique(dropna=True) <= 1]

    cardinality = pd.Series(
        {c: df[c].nunique(dropna=True) for c in categorical_cols}, dtype="int64"
    ).sort_values(ascending=False)

    numeric_summary = df[numeric_cols].describe().T if numeric_cols else pd.DataFrame()

    return {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "memory_mb": round(df.memory_usage(deep=True).sum() / (1024 * 1024), 3),
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "missing_table": missing_table,
        "duplicate_rows": dup_count,
        "constant_cols": constant_cols,
        "cardinality": cardinality,
        "numeric_summary": numeric_summary,
    }
