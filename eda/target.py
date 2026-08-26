"""Target column auto-detection and task-type inference."""
from __future__ import annotations

import re

import pandas as pd

NO_TARGET = "타겟 없음 (비지도 분석)"

# Checked in order; first tier is unambiguous, second tier is more generic and only
# consulted if nothing in the first tier matches.
STRONG_TARGET_HINTS = ["target", "label", "outcome", "decay", "churn"]
WEAK_TARGET_HINTS = ["class", "y", "output", "response", "result", "score"]

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(col_name: str) -> set[str]:
    """Split a column name into lowercase word tokens (e.g. 'employee_id' -> {'employee','id'}).

    Using whole-token matching (instead of substring search) avoids false positives like
    the hint 'y' matching inside 'employee' or 'salary'.
    """
    return set(_TOKEN_RE.findall(str(col_name).lower()))


def _looks_like_identifier(series: pd.Series, n_rows: int) -> bool:
    """Heuristic for ID/free-text columns that should never be auto-suggested as a target."""
    if n_rows == 0:
        return False
    n_unique = series.nunique(dropna=True)
    return n_unique == n_rows or (not pd.api.types.is_numeric_dtype(series) and n_unique > max(50, 0.5 * n_rows))


def suggest_target(df: pd.DataFrame) -> tuple[str | None, str]:
    """Return (suggested_column_or_None, reason)."""
    candidates = [c for c in df.columns if not _looks_like_identifier(df[c], len(df))]
    if not candidates:
        candidates = list(df.columns)

    for hint_tier, hints in [("강한", STRONG_TARGET_HINTS), ("약한", WEAK_TARGET_HINTS)]:
        for col in candidates:
            tokens = _tokenize(col)
            matched = tokens & set(hints)
            if matched:
                return col, f"컬럼명에 {hint_tier} 타겟 패턴 '{matched.pop()}' 포함"

    if candidates and len(df.columns) >= 2:
        return candidates[-1], "이름 패턴 매칭 실패 → 마지막 컬럼을 저신뢰 추정치로 제안"

    return None, "타겟을 추정할 수 없음"


def infer_task_type(series: pd.Series, max_classification_cardinality: int = 20) -> str:
    """Return 'classification' or 'regression' for a candidate target series."""
    non_null = series.dropna()
    if non_null.empty:
        return "regression"

    if not pd.api.types.is_numeric_dtype(non_null):
        return "classification"

    n_unique = non_null.nunique()
    looks_discrete = (non_null == non_null.round()).all()
    if n_unique <= max_classification_cardinality and looks_discrete:
        return "classification"
    return "regression"
