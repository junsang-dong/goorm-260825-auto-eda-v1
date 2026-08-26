"""Build a self-contained, executed Jupyter Notebook that reproduces the Streamlit analysis.

The notebook embeds the uploaded CSV as base64 so it can be reopened and re-run anywhere
without needing the original file or this ``eda`` package installed.
"""
from __future__ import annotations

import base64
import datetime as dt

import nbformat as nbf
from nbclient import NotebookClient
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook


def _csv_b64(csv_bytes: bytes) -> str:
    return base64.b64encode(csv_bytes).decode("ascii")


def build_notebook(
    csv_bytes: bytes,
    filename: str,
    target_col: str | None,
    task_type: str | None,
    top_n: int,
    feature_cols: list[str],
) -> bytes:
    cells = []
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M")

    cells.append(new_markdown_cell(
        f"# 자동 EDA 리포트: `{filename}`\n\n"
        f"생성 시각: {now}\n\n"
        f"- 타겟 변수: **{target_col or '없음 (비지도 분석)'}**\n"
        f"- 과제 유형: **{task_type or 'N/A'}**\n"
        f"- 선정 변수 개수: **{top_n}**\n"
    ))

    cells.append(new_code_cell(
        "import base64, io\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "import seaborn as sns\n"
        "import matplotlib.font_manager as fm\n"
        "import matplotlib.pyplot as plt\n"
        "from sklearn.decomposition import PCA\n"
        "from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor\n"
        "from sklearn.feature_selection import mutual_info_classif, mutual_info_regression\n"
        "from sklearn.preprocessing import StandardScaler\n\n"
        "sns.set_theme(style='whitegrid')\n"
        "pd.set_option('display.max_columns', 100)\n\n"
        "# 차트 제목이 한글이므로 한글 지원 폰트가 있으면 사용 (없으면 기본 폰트로 폴백)\n"
        "_korean_fonts = ['AppleGothic', 'Apple SD Gothic Neo', 'Malgun Gothic',\n"
        "                 'NanumGothic', 'Nanum Gothic', 'Noto Sans CJK KR', 'Noto Sans KR']\n"
        "_available = {f.name for f in fm.fontManager.ttflist}\n"
        "for _f in _korean_fonts:\n"
        "    if _f in _available:\n"
        "        plt.rcParams['font.family'] = _f\n"
        "        break\n"
        "plt.rcParams['axes.unicode_minus'] = False"
    ))

    cells.append(new_markdown_cell("## 1. 데이터 로드"))
    cells.append(new_code_cell(
        f"CSV_B64 = \"\"\"{_csv_b64(csv_bytes)}\"\"\"\n"
        "csv_bytes = base64.b64decode(CSV_B64)\n"
        "df = pd.read_csv(io.BytesIO(csv_bytes))\n"
        "df.head()"
    ))

    cells.append(new_markdown_cell("## 2. 데이터 프로파일링"))
    cells.append(new_code_cell(
        "print(f'행: {df.shape[0]:,} / 열: {df.shape[1]}')\n"
        "print(f'중복 행: {df.duplicated().sum():,}')\n"
        "missing = df.isna().sum()\n"
        "missing_pct = (missing / len(df) * 100).round(2)\n"
        "pd.DataFrame({'missing_count': missing, 'missing_pct': missing_pct})"
        ".sort_values('missing_count', ascending=False).head(20)"
    ))
    cells.append(new_code_cell("df.describe(include='number').T"))

    feature_cols_repr = repr(feature_cols)
    cells.append(new_markdown_cell("## 3. 타겟 및 과제 유형"))
    cells.append(new_code_cell(
        f"target_col = {target_col!r}\n"
        f"task_type = {task_type!r}\n"
        f"feature_cols = {feature_cols_repr}\n"
        "print('target:', target_col, '| task_type:', task_type)\n"
        "print('feature 개수:', len(feature_cols))"
    ))

    cells.append(new_markdown_cell("## 4. 변수 중요도 계산 및 Top 변수 선정"))
    cells.append(new_code_cell(_IMPORTANCE_CODE.format(top_n=top_n)))
    cells.append(new_code_cell(
        "fig, ax = plt.subplots(figsize=(8, max(2, 0.5 * len(scores))))\n"
        "ordered = scores.sort_values('importance_score', ascending=True)\n"
        "sns.barplot(x=ordered['importance_score'], y=ordered.index, ax=ax, color='#DD8452')\n"
        "ax.set_title('Top 변수 중요도'); ax.set_xlabel('종합 중요도 점수'); ax.set_ylabel('')\n"
        "fig.tight_layout(); plt.show()"
    ))

    cells.append(new_markdown_cell("## 5. Top 변수별 시각화"))
    cells.append(new_code_cell(
        "top_features = scores.index.tolist()\n"
        "for col in top_features:\n"
        "    fig, ax = plt.subplots(figsize=(6, 3.5))\n"
        "    if pd.api.types.is_numeric_dtype(df[col]):\n"
        "        sns.histplot(df[col].dropna(), kde=True, ax=ax, color='#4C72B0')\n"
        "    else:\n"
        "        df[col].value_counts().head(20).plot(kind='bar', ax=ax, color='#4C72B0')\n"
        "    ax.set_title(f'{col} 분포')\n"
        "    fig.tight_layout(); plt.show()\n"
        "    if target_col is not None:\n"
        "        fig, ax = plt.subplots(figsize=(6, 3.5))\n"
        "        if task_type == 'regression' and pd.api.types.is_numeric_dtype(df[col]):\n"
        "            sns.regplot(data=df, x=col, y=target_col, ax=ax,\n"
        "                        scatter_kws={'alpha': 0.4, 's': 15}, line_kws={'color': '#C44E52'})\n"
        "            ax.set_title(f'{col} vs {target_col}')\n"
        "        else:\n"
        "            sns.boxplot(data=df, x=target_col, y=col, ax=ax, color='#55A868')\n"
        "            ax.set_title(f'{target_col}별 {col} 분포')\n"
        "            ax.tick_params(axis='x', rotation=45)\n"
        "        fig.tight_layout(); plt.show()"
    ))

    cells.append(new_markdown_cell("## 6. Top 변수 상관관계"))
    cells.append(new_code_cell(
        "numeric_top = [c for c in top_features if pd.api.types.is_numeric_dtype(df[c])]\n"
        "if len(numeric_top) >= 2:\n"
        "    fig, ax = plt.subplots(figsize=(max(4, 0.6*len(numeric_top)), max(4, 0.6*len(numeric_top))))\n"
        "    sns.heatmap(df[numeric_top].corr(), annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax, square=True)\n"
        "    ax.set_title('Top 변수 상관관계'); fig.tight_layout(); plt.show()\n"
        "else:\n"
        "    print('수치형 변수가 2개 미만이라 상관행렬을 생략합니다.')"
    ))

    cells.append(new_markdown_cell("## 7. 결론"))
    cells.append(new_code_cell(
        "print('선정된 중요 변수 (중요도 내림차순):')\n"
        "for rank, (name, row) in enumerate(scores.iterrows(), start=1):\n"
        "    print(f'{rank}. {name}  (score={row[\"importance_score\"]:.3f})')"
    ))

    notebook = new_notebook(cells=cells)
    notebook.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }

    client = NotebookClient(notebook, timeout=600, kernel_name="python3")
    client.execute()

    return nbf.writes(notebook).encode("utf-8")


_IMPORTANCE_CODE = """
def _normalize(s):
    if s.max() == s.min():
        return pd.Series(0.0, index=s.index)
    return (s - s.min()) / (s.max() - s.min())

def _build_feature_matrix(frame, cols):
    frame = frame[cols].copy()
    numeric_cols = frame.select_dtypes(include='number').columns.tolist()
    categorical_cols = [c for c in cols if c not in numeric_cols]
    for c in numeric_cols:
        frame[c] = frame[c].fillna(frame[c].median())
    for c in categorical_cols:
        mode = frame[c].mode(dropna=True)
        frame[c] = frame[c].fillna(mode.iloc[0] if not mode.empty else 'missing')
    if categorical_cols:
        frame = pd.get_dummies(frame, columns=categorical_cols, drop_first=False)
    return frame

if target_col is not None:
    work = df[feature_cols + [target_col]].dropna(subset=[target_col])
    y_raw = work[target_col]
    X = _build_feature_matrix(work, feature_cols)

    if task_type == 'classification':
        y = y_raw.astype('category').cat.codes
        rf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1).fit(X, y)
        mi = mutual_info_classif(X, y, random_state=42)
    else:
        y = pd.to_numeric(y_raw, errors='coerce')
        valid = y.notna()
        X, y = X[valid], y[valid]
        rf = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1).fit(X, y)
        mi = mutual_info_regression(X, y, random_state=42)

    def collapse(s):
        out = {{}}
        for col, val in s.items():
            base = col
            for orig in feature_cols:
                if col == orig or col.startswith(orig + '_'):
                    base = orig
                    break
            out[base] = out.get(base, 0.0) + val
        return pd.Series(out)

    rf_c = _normalize(collapse(pd.Series(rf.feature_importances_, index=X.columns)))
    mi_c = _normalize(collapse(pd.Series(mi, index=X.columns)))
    scores = pd.DataFrame({{'random_forest': rf_c, 'mutual_info': mi_c}})

    if task_type == 'regression':
        numeric_features = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
        if numeric_features:
            corr = df[numeric_features + [target_col]].corr(numeric_only=True)[target_col].drop(target_col).abs()
            scores['target_correlation'] = _normalize(corr.reindex(scores.index).fillna(0))

    scores['importance_score'] = scores.mean(axis=1)
    scores = scores.sort_values('importance_score', ascending=False).head({top_n})
else:
    numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    X = df[numeric_cols].fillna(df[numeric_cols].median())
    X_scaled = StandardScaler().fit_transform(X)
    n_components = min(len(numeric_cols), X_scaled.shape[0])
    pca = PCA(n_components=n_components, random_state=42).fit(X_scaled)
    cum_var = np.cumsum(pca.explained_variance_ratio_)
    k = max(1, min(int(np.searchsorted(cum_var, 0.80) + 1), n_components))
    loadings = pca.components_[:k]
    weights = pca.explained_variance_ratio_[:k]
    contribution = np.sum((loadings.T ** 2) * weights, axis=1)
    pca_score = pd.Series(contribution, index=numeric_cols)
    scores = pd.DataFrame({{'pca_contribution': _normalize(pca_score)}})
    scores['importance_score'] = scores['pca_contribution']
    scores = scores.sort_values('importance_score', ascending=False).head({top_n})

scores
"""
