"""Streamlit CSV EDA 자동화 에이전트.

CSV를 업로드하면 자동으로 프로파일링 → 타겟 추정 → 중요 변수 랭킹 → 시각화를 수행하고,
분석을 재현하는 실행된 Jupyter Notebook을 다운로드할 수 있다.
"""
from __future__ import annotations

from pathlib import Path

import streamlit as st

from eda.importance import candidate_feature_columns, rank_top_features
from eda.loading import CSVLoadError, load_csv
from eda.notebook_builder import build_notebook
from eda.profiling import profile_dataset
from eda.target import NO_TARGET, infer_task_type, suggest_target
from eda.visualize import (
    plot_correlation_heatmap,
    plot_distribution,
    plot_importance_bar,
    plot_missingness,
    plot_vs_target,
)

SAMPLE_CSV_PATH = Path(__file__).parent / "sample_data" / "sample_dataset.csv"

st.set_page_config(page_title="CSV EDA 자동화 에이전트", layout="wide")
st.title("📊 CSV EDA 자동화 에이전트")
st.caption("CSV를 업로드하면 프로파일링, 중요 변수 선정(5~10개), 시각화를 자동으로 수행합니다.")

if "raw_bytes" not in st.session_state:
    st.session_state.raw_bytes = None
    st.session_state.filename = None

col_upload, col_sample = st.columns([3, 1])
with col_upload:
    uploaded = st.file_uploader("CSV 파일 업로드", type=["csv", "txt"])
with col_sample:
    st.write("")
    st.write("")
    if st.button("샘플 데이터로 체험하기", width="stretch"):
        st.session_state.raw_bytes = SAMPLE_CSV_PATH.read_bytes()
        st.session_state.filename = SAMPLE_CSV_PATH.name

if uploaded is not None:
    st.session_state.raw_bytes = uploaded.getvalue()
    st.session_state.filename = uploaded.name

if st.session_state.raw_bytes is None:
    st.info("좌측 상단에서 CSV 파일을 업로드하거나 '샘플 데이터로 체험하기'를 눌러보세요.")
    st.stop()

try:
    df, meta = load_csv(st.session_state.raw_bytes, st.session_state.filename)
except CSVLoadError as exc:
    st.error(str(exc))
    st.stop()

st.success(
    f"로드 완료: {meta['filename']} · 인코딩={meta['encoding']} · 구분자='{meta['delimiter']}'"
    + (" · ⚠️ 대용량이라 20만 행으로 샘플링됨" if meta["sampled"] else "")
)

st.subheader("데이터 미리보기")
st.dataframe(df.head(20), width="stretch")

profile = profile_dataset(df)

st.subheader("데이터 프로파일링")
m1, m2, m3, m4 = st.columns(4)
m1.metric("행 수", f"{profile['n_rows']:,}")
m2.metric("열 수", profile["n_cols"])
m3.metric("중복 행", f"{profile['duplicate_rows']:,}")
m4.metric("메모리 사용량", f"{profile['memory_mb']} MB")

pcol1, pcol2 = st.columns(2)
with pcol1:
    st.markdown("**결측치 비율**")
    st.pyplot(plot_missingness(profile["missing_table"]))
with pcol2:
    st.markdown("**수치형 컬럼 요약**")
    st.dataframe(profile["numeric_summary"], width="stretch")

if profile["constant_cols"]:
    st.warning(f"상수(값이 1개뿐인) 컬럼은 중요도 계산에서 제외됩니다: {profile['constant_cols']}")

st.divider()
st.subheader("타겟 선택 및 중요 변수 선정")

suggested_target, reason = suggest_target(df)
options = [NO_TARGET] + list(df.columns)
default_index = options.index(suggested_target) if suggested_target in options else 0

c1, c2 = st.columns([2, 1])
with c1:
    target_choice = st.selectbox(
        "타겟 컬럼 (자동 추정됨 — 필요시 변경하세요)",
        options,
        index=default_index,
        help=f"자동 추정 근거: {reason}",
    )
with c2:
    top_n = st.slider("선정할 변수 개수", min_value=5, max_value=10, value=7)

target_col = None if target_choice == NO_TARGET else target_choice
task_type = None
if target_col is not None:
    task_type = infer_task_type(df[target_col])
    st.caption(f"추정된 과제 유형: **{task_type}**")

run = st.button("분석 실행", type="primary")

if run:
    feature_cols = candidate_feature_columns(df, target_col, profile["constant_cols"])
    if len(feature_cols) < 2:
        st.error("중요도를 계산할 수 있는 유효한 특성 컬럼이 2개 미만입니다.")
        st.stop()

    with st.spinner("변수 중요도 계산 중..."):
        try:
            scores = rank_top_features(df, feature_cols, target_col, task_type, top_n)
        except ValueError as exc:
            st.error(str(exc))
            st.stop()

    st.session_state.analysis = {
        "scores": scores,
        "target_col": target_col,
        "task_type": task_type,
        "top_n": top_n,
        "feature_cols": feature_cols,
    }

if "analysis" in st.session_state:
    result = st.session_state.analysis
    scores = result["scores"]
    target_col = result["target_col"]
    task_type = result["task_type"]

    st.markdown("### 🏆 중요 변수 랭킹")
    display_scores = scores.copy()
    display_scores.insert(0, "순위", range(1, len(display_scores) + 1))
    st.dataframe(display_scores.style.format(precision=3), width="stretch")
    st.pyplot(plot_importance_bar(scores))

    st.markdown("### 변수별 상세 시각화")
    for col in scores.index:
        with st.expander(f"📈 {col}", expanded=False):
            vc1, vc2 = st.columns(2)
            with vc1:
                st.pyplot(plot_distribution(df, col))
            with vc2:
                if target_col is not None:
                    st.pyplot(plot_vs_target(df, col, target_col, task_type))
                else:
                    st.caption("타겟이 없어 개별 변수-타겟 관계 차트는 생략합니다.")

    st.markdown("### 상관관계")
    st.pyplot(plot_correlation_heatmap(df, list(scores.index)))

    st.divider()
    st.markdown("### 📓 분석 리포트 다운로드")
    if st.button("Jupyter Notebook 생성"):
        with st.spinner("노트북 생성 및 실행 중... (수 초~수십 초 소요)"):
            notebook_bytes = build_notebook(
                csv_bytes=st.session_state.raw_bytes,
                filename=st.session_state.filename,
                target_col=target_col,
                task_type=task_type,
                top_n=result["top_n"],
                feature_cols=result["feature_cols"],
            )
        st.session_state.notebook_bytes = notebook_bytes
        st.success("노트북 생성 완료!")

    if "notebook_bytes" in st.session_state:
        st.download_button(
            "⬇️ eda_report.ipynb 다운로드",
            data=st.session_state.notebook_bytes,
            file_name="eda_report.ipynb",
            mime="application/x-ipynb+json",
        )
