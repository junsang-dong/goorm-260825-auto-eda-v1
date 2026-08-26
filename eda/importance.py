"""Feature importance ranking: supervised (target given) and unsupervised (no target)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.preprocessing import StandardScaler

MAX_ONEHOT_CARDINALITY = 20


def candidate_feature_columns(df: pd.DataFrame, target_col: str | None, constant_cols: list[str]) -> list[str]:
    cols = [c for c in df.columns if c != target_col and c not in constant_cols]
    # Drop free-text / very high-cardinality categorical columns (likely IDs, not useful signal).
    keep = []
    for c in cols:
        if pd.api.types.is_numeric_dtype(df[c]):
            keep.append(c)
        elif df[c].nunique(dropna=True) <= MAX_ONEHOT_CARDINALITY:
            keep.append(c)
    return keep


def _build_feature_matrix(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    frame = df[feature_cols].copy()
    numeric_cols = frame.select_dtypes(include="number").columns.tolist()
    categorical_cols = [c for c in feature_cols if c not in numeric_cols]

    for c in numeric_cols:
        frame[c] = frame[c].fillna(frame[c].median())
    for c in categorical_cols:
        mode = frame[c].mode(dropna=True)
        frame[c] = frame[c].fillna(mode.iloc[0] if not mode.empty else "missing")

    if categorical_cols:
        frame = pd.get_dummies(frame, columns=categorical_cols, drop_first=False)
    return frame


def _normalize(series: pd.Series) -> pd.Series:
    if series.max() == series.min():
        return pd.Series(0.0, index=series.index)
    return (series - series.min()) / (series.max() - series.min())


def compute_supervised_importance(
    df: pd.DataFrame,
    target_col: str,
    task_type: str,
    feature_cols: list[str],
    random_state: int = 42,
) -> pd.DataFrame:
    work = df[feature_cols + [target_col]].dropna(subset=[target_col])
    y_raw = work[target_col]
    X = _build_feature_matrix(work, feature_cols)

    if task_type == "classification":
        n_classes = y_raw.nunique(dropna=True)
        if n_classes > 50 or n_classes > 0.5 * len(y_raw):
            raise ValueError(
                f"'{target_col}' 컬럼은 고유값이 {n_classes}개로 사실상 식별자(ID)에 가까워 "
                "분류 타겟으로 적합하지 않습니다. 다른 컬럼을 타겟으로 선택해주세요."
            )
        y = y_raw.astype("category").cat.codes
        rf = RandomForestClassifier(n_estimators=300, random_state=random_state, n_jobs=-1)
        rf.fit(X, y)
        mi = mutual_info_classif(X, y, random_state=random_state)
    else:
        y = pd.to_numeric(y_raw, errors="coerce")
        valid = y.notna()
        X, y = X[valid], y[valid]
        rf = RandomForestRegressor(n_estimators=300, random_state=random_state, n_jobs=-1)
        rf.fit(X, y)
        mi = mutual_info_regression(X, y, random_state=random_state)

    rf_importance = pd.Series(rf.feature_importances_, index=X.columns)
    mi_series = pd.Series(mi, index=X.columns)

    # Map one-hot columns back to their original feature name by summing scores.
    def collapse(scores: pd.Series) -> pd.Series:
        collapsed = {}
        for col, val in scores.items():
            base = col
            for orig in feature_cols:
                if col == orig or col.startswith(f"{orig}_"):
                    base = orig
                    break
            collapsed[base] = collapsed.get(base, 0.0) + val
        return pd.Series(collapsed)

    rf_collapsed = _normalize(collapse(rf_importance))
    mi_collapsed = _normalize(collapse(mi_series))

    scores = pd.DataFrame({"random_forest": rf_collapsed, "mutual_info": mi_collapsed})

    if task_type == "regression":
        numeric_features = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
        if numeric_features:
            corr = df[numeric_features + [target_col]].corr(numeric_only=True)[target_col].drop(target_col).abs()
            scores["target_correlation"] = _normalize(corr.reindex(scores.index).fillna(0))

    scores["importance_score"] = scores.mean(axis=1)
    return scores.sort_values("importance_score", ascending=False)


def compute_unsupervised_importance(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    if len(numeric_cols) < 2:
        raise ValueError("비지도 중요도 계산에는 최소 2개의 수치형 컬럼이 필요합니다.")

    X = df[numeric_cols].fillna(df[numeric_cols].median())
    X_scaled = StandardScaler().fit_transform(X)

    n_components = min(len(numeric_cols), X_scaled.shape[0])
    pca = PCA(n_components=n_components, random_state=42)
    pca.fit(X_scaled)

    cum_var = np.cumsum(pca.explained_variance_ratio_)
    k = int(np.searchsorted(cum_var, 0.80) + 1)
    k = max(1, min(k, n_components))

    loadings = pca.components_[:k]  # shape (k, n_features)
    weights = pca.explained_variance_ratio_[:k]
    contribution = np.sum((loadings.T ** 2) * weights, axis=1)

    pca_score = pd.Series(contribution, index=numeric_cols)

    corr_matrix = X.corr().abs()
    np.fill_diagonal(corr_matrix.values, np.nan)
    redundancy = corr_matrix.mean(axis=1)

    scores = pd.DataFrame(
        {
            "pca_contribution": _normalize(pca_score),
            "avg_abs_correlation": redundancy.reindex(pca_score.index),
        }
    )
    scores["importance_score"] = scores["pca_contribution"]
    return scores.sort_values("importance_score", ascending=False)


def rank_top_features(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str | None,
    task_type: str | None,
    top_n: int,
) -> pd.DataFrame:
    if target_col is not None:
        scores = compute_supervised_importance(df, target_col, task_type, feature_cols)
    else:
        scores = compute_unsupervised_importance(df, feature_cols)
    return scores.head(top_n)
