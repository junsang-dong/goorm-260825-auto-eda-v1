"""Chart builders (matplotlib/seaborn Figures) shared by the Streamlit app and the exported notebook."""
from __future__ import annotations

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")

# Chart titles/labels are Korean; matplotlib's default font (DejaVu Sans) has no Hangul
# glyphs and silently drops them. Pick the first Korean-capable font actually installed,
# falling back to DejaVu Sans (glyphs will show as tofu boxes, but nothing crashes).
_KOREAN_FONT_PREFERENCE = [
    "AppleGothic", "Apple SD Gothic Neo", "Malgun Gothic",
    "NanumGothic", "Nanum Gothic", "Noto Sans CJK KR", "Noto Sans KR",
]
_available = {f.name for f in fm.fontManager.ttflist}
for _font in _KOREAN_FONT_PREFERENCE:
    if _font in _available:
        plt.rcParams["font.family"] = _font
        break
plt.rcParams["axes.unicode_minus"] = False


def plot_missingness(missing_table: pd.DataFrame, top: int = 20) -> plt.Figure:
    data = missing_table[missing_table["missing_count"] > 0].head(top)
    fig, ax = plt.subplots(figsize=(8, max(2, 0.35 * len(data))))
    if data.empty:
        ax.text(0.5, 0.5, "결측치 없음", ha="center", va="center")
        ax.axis("off")
        return fig
    sns.barplot(x=data["missing_pct"], y=data.index, ax=ax, color="#4C72B0")
    ax.set_xlabel("결측 비율 (%)")
    ax.set_ylabel("")
    ax.set_title("컬럼별 결측치 비율")
    fig.tight_layout()
    return fig


def plot_importance_bar(scores: pd.DataFrame) -> plt.Figure:
    ordered = scores.sort_values("importance_score", ascending=True)
    fig, ax = plt.subplots(figsize=(8, max(2, 0.5 * len(ordered))))
    sns.barplot(x=ordered["importance_score"], y=ordered.index, ax=ax, color="#DD8452")
    ax.set_xlabel("종합 중요도 점수 (0~1 정규화)")
    ax.set_ylabel("")
    ax.set_title("Top 변수 중요도")
    fig.tight_layout()
    return fig


def plot_distribution(df: pd.DataFrame, column: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 3.5))
    series = df[column]
    if pd.api.types.is_numeric_dtype(series):
        sns.histplot(series.dropna(), kde=True, ax=ax, color="#4C72B0")
    else:
        series.value_counts().head(20).plot(kind="bar", ax=ax, color="#4C72B0")
    ax.set_title(f"{column} 분포")
    fig.tight_layout()
    return fig


def plot_vs_target(df: pd.DataFrame, column: str, target_col: str, task_type: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 3.5))
    if task_type == "regression" and pd.api.types.is_numeric_dtype(df[column]):
        sns.regplot(
            data=df, x=column, y=target_col, ax=ax,
            scatter_kws={"alpha": 0.4, "s": 15}, line_kws={"color": "#C44E52"},
        )
        ax.set_title(f"{column} vs {target_col}")
    else:
        sns.boxplot(data=df, x=target_col, y=column, ax=ax, color="#55A868")
        ax.set_title(f"{target_col}별 {column} 분포")
        ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    return fig


def plot_correlation_heatmap(df: pd.DataFrame, columns: list[str]) -> plt.Figure:
    numeric_cols = [c for c in columns if pd.api.types.is_numeric_dtype(df[c])]
    fig, ax = plt.subplots(figsize=(max(4, 0.6 * len(numeric_cols)), max(4, 0.6 * len(numeric_cols))))
    if len(numeric_cols) < 2:
        ax.text(0.5, 0.5, "수치형 변수가 2개 미만이라 상관행렬을 그릴 수 없음", ha="center", va="center")
        ax.axis("off")
        return fig
    corr = df[numeric_cols].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax, square=True)
    ax.set_title("Top 변수 상관관계")
    fig.tight_layout()
    return fig
