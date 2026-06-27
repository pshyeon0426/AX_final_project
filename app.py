"""
슈퍼SOL 고객 반응 · 경쟁 벤치마킹 · 리뷰 답글 데모 대시보드 (app.py)

실행:
    streamlit run app.py

설계:
- 사전 생성된 outputs/ 산출물(metrics.json, reply_drafts.csv 등)로 **오프라인 데모** 가능.
- OPENAI_API_KEY 가 없으면 API 미사용 + 룰 기반 fallback 상태를 화면에 표시한다.
- 답글은 **자동 게시가 아니라 담당자 검수용 초안**임을 화면에 명확히 표시한다.

화면별 함수 구조:
    main()
    ├─ render_sidebar()              # CSV 업로드 / 샘플 / 앱·기간·경쟁앱 선택 / OpenAI 상태
    ├─ render_kpi_cards(df, ...)     # 리뷰수·평균평점·부정률·토픽수·리포트 생성시간
    ├─ render_charts(df)             # 리뷰 추이·감성 분포·불만 유형 TOP·앱별 비교
    ├─ render_tables(df)             # 부정 리뷰·토픽 대표 리뷰·경쟁 비교표
    ├─ tab_ai_report()               # AI 요약 리포트 + 근거 문서
    ├─ tab_reply_generation(df)      # 리뷰 선택·답글 생성·재생성·수정·복사·다운로드
    └─ tab_validation()              # 데이터 검증·모델 성능·답글 품질·테스트 결과
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from src.config import PATHS

OUT = PATHS.outputs_dir
CLEAN_PATH = PATHS.processed_dir / "review_clean.csv"
OUR_APP = "슈퍼SOL"

st.set_page_config(page_title="슈퍼SOL 고객 반응 분석 대시보드", page_icon="📊", layout="wide")

# ----------------------------------------------------------------------
# 공용 디자인 테마 (src/theme.py 단일 진실원본 — 앱/차트/노트북 공유)
# ----------------------------------------------------------------------
from src.theme import (  # noqa: E402
    PALETTE as BRAND, SENTIMENT_COLORS, SEQ, chip_html as _chip, apply_plotly_theme,
)

apply_plotly_theme()  # 모든 Plotly 차트에 공통 폰트/그리드/배경 적용


def _kpi_card(col, label: str, value: str, tone: str = "neutral", sub: str = "", help: str = ""):
    """모던/절제된 KPI 카드(HTML). 숫자는 잉크색 기준, tone 은 은은한 강조.
    help 가 있으면 카드 hover 시 브라우저 툴팁 + 라벨 옆 ⓘ 표시."""
    accent = {"ok": BRAND["ok"], "warn": BRAND["warn"], "bad": BRAND["bad"],
              "primary": BRAND["primary"]}.get(tone, BRAND["neutral"])
    sub_html = (f"<div style='font-size:0.72rem;color:{BRAND['muted_text']};margin-top:3px'>{sub}</div>"
                if sub else "")
    info = (f"<span style='color:{BRAND['neutral']};font-size:0.72rem;cursor:help'>ⓘ</span>"
            if help else "")
    title_attr = f' title="{help}"' if help else ""
    col.markdown(
        f"""<div{title_attr} style="border:1px solid {BRAND['card_border']};border-radius:12px;
        padding:14px 16px;background:{BRAND['card_bg']};box-shadow:0 1px 2px rgba(16,24,40,.04)">
        <div style="display:flex;align-items:center;gap:6px">
          <span style="width:7px;height:7px;border-radius:50%;background:{accent};display:inline-block"></span>
          <span style="font-size:0.78rem;color:{BRAND['muted_text']};font-weight:500">{label}</span> {info}
        </div>
        <div style="font-size:1.65rem;font-weight:650;color:{BRAND['ink']};line-height:1.25;margin-top:4px">{value}</div>
        {sub_html}</div>""",
        unsafe_allow_html=True,
    )


# ======================================================================
# 데이터 로딩 (캐시)
# ----------------------------------------------------------------------
# ⚠️ 캐시 무효화: st.cache_data 는 인자만으로 키를 만든다. 파일명만 넘기면
#    파일이 갱신돼도 세션 내 옛 캐시가 반환된다(과거 버그). 따라서 파일의
#    지문(mtime:size)을 인자로 함께 넘겨, 파일이 바뀌면 캐시가 자동 무효화되게 한다.
# ======================================================================
def _file_fp(path: Path) -> str:
    """파일 지문(수정시각:크기). 파일이 바뀌면 값이 달라져 캐시 키가 갱신된다."""
    try:
        s = path.stat()
        return f"{int(s.st_mtime)}:{s.st_size}"
    except FileNotFoundError:
        return "missing"


@st.cache_data(show_spinner=False)
def _read_reviews(path: str, fingerprint: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    return df


def load_reviews(path: str = str(CLEAN_PATH)) -> pd.DataFrame:
    return _read_reviews(path, _file_fp(Path(path)))


@st.cache_data(show_spinner=False)
def _read_json(name: str, fingerprint: str) -> dict:
    p = OUT / name
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def load_json(name: str) -> dict:
    return _read_json(name, _file_fp(OUT / name))


@st.cache_data(show_spinner=False)
def _read_csv(name: str, fingerprint: str) -> pd.DataFrame:
    p = OUT / name
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def load_csv(name: str) -> pd.DataFrame:
    return _read_csv(name, _file_fp(OUT / name))


@st.cache_data(show_spinner=False)
def _read_text(name: str, fingerprint: str) -> str:
    p = OUT / name
    return p.read_text(encoding="utf-8") if p.exists() else ""


def load_text(name: str) -> str:
    return _read_text(name, _file_fp(OUT / name))


def _weak_label(r):
    if pd.isna(r):
        return "neutral"
    return "negative" if r <= 2 else ("neutral" if r == 3 else "positive")


def _live_complaint_top(df_app: pd.DataFrame, top_n: int = 5):
    """선택된(기준앱+기간) 부정 리뷰에서 불만 유형 TOP N 을 실시간 집계(룰 기반)."""
    from src.issue_classifier import classify_issue_type, ISSUE_OTHER
    neg = df_app[df_app["rating"] <= 2]
    if neg.empty:
        return []
    counts: dict[str, int] = {}
    text_col = "clean_text" if "clean_text" in neg.columns else "review_text"
    for txt in neg[text_col].fillna("").astype(str):
        for t in classify_issue_type(txt):
            if t != ISSUE_OTHER:
                counts[t] = counts.get(t, 0) + 1
    return sorted(counts.items(), key=lambda x: -x[1])[:top_n]


def _reply_candidates(source: pd.DataFrame, order: str, limit: int = 50) -> pd.DataFrame:
    """답글 대상 후보를 정렬해 상위 limit 개 반환.

    - '최신순': 날짜 내림차순.
    - '우선순위순': 심각도(high→positive) → 낮은 별점 → 최신. 비용 제한 위해 최근 400건만 심각도 계산.
    세션 메모로 선택 박스 조작 시 재계산을 피한다.
    """
    cand = source.dropna(subset=["clean_text"]).copy()
    cand = cand[cand["clean_text"].astype(str).str.strip() != ""]
    if cand.empty:
        return cand
    cand = cand.sort_values("date", ascending=False)
    if order == "최신순":
        return cand.head(limit)

    # 우선순위순 (메모: 동일 풀/정렬이면 재계산 안 함)
    sig = f"{order}|{len(cand)}|{cand['date'].max()}|{float(cand['rating'].sum())}"
    if st.session_state.get("_cand_sig") == sig and "_cand_idx" in st.session_state:
        idx = [i for i in st.session_state["_cand_idx"] if i in cand.index]
        return cand.loc[idx]

    from src.issue_classifier import classify_issue_type, classify_severity
    rank = {"high": 0, "medium": 1, "low": 2, "positive": 3}
    pool = cand.head(400).copy()
    sev = []
    for _, r in pool.iterrows():
        it = classify_issue_type(r["clean_text"])
        sev.append(rank.get(classify_severity(r["clean_text"], r.get("rating"), it), 1))
    pool["_sev"] = sev
    pool = pool.sort_values(["_sev", "rating", "date"], ascending=[True, True, False]).head(limit)
    st.session_state["_cand_sig"] = sig
    st.session_state["_cand_idx"] = pool.index.tolist()
    return pool


def _custom_row(default_app: str):
    """댓글을 붙여넣고 별점만 고르면 바로 답글 대상 행(Series)을 만든다. 비어 있으면 None."""
    import datetime as _dt
    if st.button("🆕 새 댓글(지우기)"):
        st.session_state.pop("paste_text", None)
        st.session_state.pop("reply_text", None)  # 이전 답글도 비움
        st.rerun()
    text = st.text_area("댓글 본문 (복사·붙여넣기)", height=110, key="paste_text",
                        placeholder="답글을 생성할 댓글 내용을 붙여넣으세요")
    c1, c2 = st.columns(2)
    app_in = c1.text_input("앱", value=default_app, key="paste_app")
    rating_in = c2.selectbox("별점", [1, 2, 3, 4, 5], index=2, key="paste_rating")
    if not text.strip():
        st.info("댓글을 붙여넣으면 아래에서 답글을 생성할 수 있습니다.")
        return None
    return pd.Series({
        "app_name": app_in or default_app, "store": "manual", "rating": int(rating_in),
        "date": pd.Timestamp(_dt.date.today()),
        "review_text": text.strip(), "clean_text": text.strip(), "version": "",
    })


# ----------------------------------------------------------------------
# 컬럼 한글 라벨 + 도움말 (표 표시 공통)
# ----------------------------------------------------------------------
COL_LABELS = {
    "app_name": "앱", "store": "스토어", "rating": "평점", "date": "날짜",
    "review_text": "리뷰 원문", "clean_text": "리뷰(정제)", "version": "앱 버전", "source_url": "출처",
    "review_count": "리뷰 수", "avg_rating": "평균 평점", "negative_ratio": "부정률",
    "recent30d_rating_change": "최근 30일 평점변화", "top_complaint_types": "불만 유형 TOP",
    "top_topic_keywords": "주요 토픽 키워드",
    "topic_id": "토픽 번호", "size": "리뷰 수", "auto_label": "자동 라벨", "topic_label": "토픽명(검수)",
    "top_keywords": "대표 키워드", "rep_review_1": "대표 리뷰",
    "needs_human_review": "검수 필요", "safety_flags": "안전 플래그", "reply_draft": "답글 초안",
    "reply_type": "답글 유형", "severity": "심각도", "issue_types": "이슈 유형", "sentiment": "감성",
    "reason": "사유", "true_label_star": "실제(별점라벨)", "pred_label": "예측",
    "app": "앱", "count": "건수", "status": "상태",
    "feat_혜택_mention_rate": "혜택 언급률", "feat_인증_mention_rate": "인증 언급률",
    "feat_송금_mention_rate": "송금 언급률", "feat_투자_mention_rate": "투자 언급률",
    "feat_UI/UX_mention_rate": "UI/UX 언급률", "feat_이벤트_mention_rate": "이벤트 언급률",
}
PCT_COLS = {"negative_ratio", "feat_혜택_mention_rate", "feat_인증_mention_rate",
            "feat_송금_mention_rate", "feat_투자_mention_rate",
            "feat_UI/UX_mention_rate", "feat_이벤트_mention_rate"}
# 값(셀) 한글 매핑
STORE_KOR = {"google_play": "구글플레이", "app_store": "앱스토어", "sample": "샘플", "manual": "수동입력"}
SENT_KOR = {"positive": "긍정", "neutral": "중립", "negative": "부정"}
VALUE_MAPS = {"store": STORE_KOR, "sentiment": SENT_KOR}
COL_HELP = {
    "clean_text": "이모지·특수문자·반복공백을 정리하고 개인정보를 [MASKED] 처리한 리뷰",
    "negative_ratio": "별점 1~2점 리뷰의 비율",
    "recent30d_rating_change": "최근 30일 평균평점 − 직전 30일 평균평점 (양수=개선)",
    "auto_label": "상위 키워드로 자동 생성한 임시 토픽명(검수 전)",
    "topic_label": "검수자가 입력하는 최종 토픽명(공란 가능)",
    "top_complaint_types": "부정 리뷰에서 가장 많이 나온 불만 유형(룰 기반)",
    "top_topic_keywords": "TF-IDF 상위 키워드로 본 앱의 주요 화제",
    "needs_human_review": "보안·금전 등 민감 사유로 담당자 검수가 필요한 답글",
    "safety_flags": "답글에서 탐지된 위험 신호(민감정보 요청·확정약속·투자조언 등)",
    "severity": "high(긴급)·medium·low·positive 4단계 우선순위",
    "true_label_star": "별점 기반 약지도 라벨 (실제 감정 정답이 아님)",
    "pred_label": "모델이 예측한 감성",
    "source_url": "리뷰를 수집한 공개 스토어 주소",
    "feat_혜택_mention_rate": "리뷰 중 '혜택' 관련 키워드를 언급한 비율",
    "feat_인증_mention_rate": "리뷰 중 '인증' 관련 키워드를 언급한 비율",
    "feat_송금_mention_rate": "리뷰 중 '송금' 관련 키워드를 언급한 비율",
    "feat_투자_mention_rate": "리뷰 중 '투자' 관련 키워드를 언급한 비율",
    "feat_UI/UX_mention_rate": "리뷰 중 'UI/UX' 관련 키워드를 언급한 비율",
    "feat_이벤트_mention_rate": "리뷰 중 '이벤트' 관련 키워드를 언급한 비율",
}


def show_table(df, height=None, hide_index: bool = True):
    """영문 컬럼을 한글로 매핑하고, 어려운 컬럼엔 도움말(헤더 hover)을 붙여 표시한다."""
    kw = {"use_container_width": True, "hide_index": hide_index}
    if height is not None:
        kw["height"] = height  # None 은 st.dataframe 이 허용하지 않으므로 생략
    if df is None or len(df) == 0:
        st.dataframe(df, **kw)
        return
    disp = df.copy()
    for c in list(disp.columns):
        # 날짜는 시간(항상 00:00:00)이 무의미하므로 날짜만 표시
        if pd.api.types.is_datetime64_any_dtype(disp[c]):
            disp[c] = disp[c].dt.strftime("%Y-%m-%d")
        elif c in PCT_COLS and pd.api.types.is_numeric_dtype(disp[c]):
            disp[c] = (disp[c].astype(float) * 100).round(1).astype(str) + "%"
        elif c in VALUE_MAPS:  # 영문 값(store/sentiment 등) → 한글
            disp[c] = disp[c].map(lambda v, _m=VALUE_MAPS[c]: _m.get(v, v))
    ren = {c: COL_LABELS.get(c, c) for c in disp.columns}
    disp = disp.rename(columns=ren)
    cfg = {ren[c]: st.column_config.Column(ren[c], help=COL_HELP[c])
           for c in df.columns if c in COL_HELP}
    st.dataframe(disp, column_config=cfg, **kw)


def live_openai_status() -> dict:
    """현재 .env 를 다시 읽어 OpenAI 사용 가능 여부를 실시간 판정한다.

    settings 싱글턴은 앱 시작 시 1회만 로드되므로, 실행 중 .env 변경을 반영하려면
    여기서 override=True 로 재로딩한다(F3 해결).
    """
    from dotenv import load_dotenv
    load_dotenv(PATHS.root / ".env", override=True)
    import os
    key = os.getenv("OPENAI_API_KEY", "").strip()
    model = os.getenv("OPENAI_MODEL", "").strip()
    placeholders = {"", "your_key_here", "your_model_here"}
    enabled = key not in placeholders and model not in placeholders
    return {"enabled": enabled, "model": model}


# ======================================================================
# 파이프라인 연동 (캐싱: 데이터 지문이 바뀌면 자동 재실행)
# ======================================================================
@st.cache_data(show_spinner="전체 파이프라인 실행 중...")
def run_pipeline_cached(data_fingerprint: str, _config: dict | None = None) -> dict:
    """run_pipeline 을 캐싱한다.

    data_fingerprint 가 바뀌면(=데이터 변경) 캐시가 무효화되어 재실행된다.
    _config 는 언더스코어 접두사로 해시 제외(매 호출 동일 객체 아님 대비).
    """
    from src.pipeline import run_pipeline
    return run_pipeline(config=_config)


def _data_fingerprint() -> str:
    """입력 데이터의 지문(경로+수정시각+크기). 변경 시 캐시 무효화 키로 사용."""
    p = CLEAN_PATH if CLEAN_PATH.exists() else (PATHS.raw_dir / "review_raw.csv")
    if p.exists():
        stt = p.stat()
        return f"{p.name}:{int(stt.st_mtime)}:{stt.st_size}"
    return "no-data"


# ======================================================================
# 사이드바
# ======================================================================
def _data_freshness() -> str:
    """데이터 최신성 문구: review_clean.csv 수정시각 + 최신 리뷰일."""
    import datetime as _dt
    p = CLEAN_PATH
    if not p.exists():
        return "데이터 없음"
    mt = _dt.datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
    return f"마지막 갱신 {mt}"


def render_sidebar() -> dict:
    st.sidebar.title("⚙️ 분석 설정")

    # ===== 1) 데이터 소스 (수집/업로드/샘플 통합) =====
    st.sidebar.header("1 · 데이터 소스")
    render_collection_sidebar()   # 🗂️ 리뷰 수집 (스토어 + 수동입력 + 앱관리)

    with st.sidebar.expander("📤 업로드 / 샘플", expanded=False):
        uploaded = st.file_uploader("리뷰 CSV 업로드", type=["csv"])
        use_sample = st.checkbox("샘플(사전 수집) 데이터 사용", value=True)

    if uploaded is not None and not use_sample:
        df = pd.read_csv(uploaded)
        df["date"] = pd.to_datetime(df.get("date"), errors="coerce")
        df["rating"] = pd.to_numeric(df.get("rating"), errors="coerce")
        st.sidebar.caption(f"📤 업로드 데이터 {len(df)}건 로드됨")
    else:
        df = load_reviews()
        if df.empty:
            st.sidebar.caption("⚠️ 데이터 없음 — 위 '🗂️ 리뷰 수집' 에서 수집하세요.")
    st.sidebar.caption(f"🕒 {_data_freshness()} · 총 {len(df):,}건")

    # ===== 2) 분석 필터 =====
    apps = sorted(df["app_name"].dropna().unique().tolist()) if not df.empty else []
    st.sidebar.header("2 · 분석 필터")
    sel_app = st.sidebar.selectbox(
        "기준 앱", apps, index=apps.index(OUR_APP) if OUR_APP in apps else 0,
        help="분석의 중심이 되는 앱입니다. KPI·차트·답글이 이 앱 기준으로 표시됩니다.") if apps else None
    competitors = [a for a in apps if a != sel_app]
    sel_comp = st.sidebar.multiselect(
        "경쟁 앱 비교", competitors, default=competitors[:4],
        help="'앱별 평균 평점 비교' 차트에 함께 표시할 경쟁 앱입니다.")

    if not df.empty and df["date"].notna().any():
        import datetime as _dt
        today = _dt.date.today()
        dmin = df["date"].min().date()
        # 기본 기간: 데이터 시작 ~ '오늘'(종료일을 오늘 기준으로)
        date_range = st.sidebar.date_input(
            "기간", value=(dmin, today), min_value=dmin, max_value=today,
            help="종료일은 기본적으로 오늘입니다. 데이터가 있는 최신일까지만 실제로 반영됩니다.")
    else:
        date_range = None

    # ===== 3) 실행 =====
    st.sidebar.header("3 · 실행")
    if st.sidebar.button("▶ 전체 파이프라인 실행", type="primary", use_container_width=True):
        result = run_pipeline_cached(_data_fingerprint())
        st.session_state["pipeline_result"] = result
        sc = result.get("success_criteria", {})
        st.session_state["_flash"] = (
            f"파이프라인 완료 · 종합 {'PASS' if sc.get('overall_pass') else 'FAIL'}")
    if st.sidebar.button("🔄 데이터/산출물 새로고침", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    st.sidebar.caption("ℹ️ '30초 이내'는 사전 학습 모델/사전 생성 인덱스를 준비한 상태에서 "
                       "분석을 요청할 때 기준입니다(최초 학습/인덱싱은 별도).")

    # ===== 4) 시스템 상태 =====
    st.sidebar.header("4 · 상태")
    oai = live_openai_status()
    if oai["enabled"]:
        st.sidebar.markdown(_chip(f"● OpenAI 사용 가능 · {oai['model']}", "ok"),
                            unsafe_allow_html=True)
    else:
        st.sidebar.markdown(_chip("● OpenAI 미사용 · 룰 기반 fallback", "neutral"),
                            unsafe_allow_html=True)
        st.sidebar.caption(".env 에 OPENAI_API_KEY/OPENAI_MODEL 설정 시 LLM 사용")

    return {"df": df, "app": sel_app, "competitors": sel_comp, "date_range": date_range}


def apply_filters(df: pd.DataFrame, app, date_range) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    if date_range and isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        start, end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
        out = out[(out["date"] >= start) & (out["date"] <= end + pd.Timedelta(days=1))]
    return out


# ======================================================================
# 컨텍스트 바 + KPI 카드
# ======================================================================
def render_context_bar(df_app, df_full, app, date_range):
    """현재 분석 컨텍스트(앱·기간·앱수) + 파이프라인 상태를 상단에 상시 표시."""
    n_apps = df_full["app_name"].nunique()
    # 선택한 '필터 기간'을 표시 (사이드바 기간과 일치). 실제 데이터 최신일은 보조 표기.
    if date_range and isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        period = f"{date_range[0]} ~ {date_range[1]}"
    elif df_app["date"].notna().any():
        period = f"{df_app['date'].min().date()} ~ {df_app['date'].max().date()}"
    else:
        period = "전체"
    data_max = df_app["date"].max().date() if df_app["date"].notna().any() else None
    coverage = (f"　·　실데이터 최신 {data_max}"
                if data_max and date_range and str(data_max) != str(date_range[1]) else "")
    left, right = st.columns([3, 1])
    left.markdown(
        f"<div style='padding:6px 0'>{_chip(f'기준 앱 · {app}', 'primary')} "
        f"<span style='color:{BRAND['muted_text']};font-size:0.85rem'>"
        f"　📅 {period}{coverage}　·　비교 앱 {n_apps}개</span></div>",
        unsafe_allow_html=True,
    )
    # 파이프라인 상태 (metrics.json 의 pipeline_summary = 단일 진실원본)
    ps = load_json("metrics.json").get("pipeline_summary", {})
    if ps:
        ok = ps.get("success_criteria_pass_fail") == "PASS"
        right.markdown(
            f"<div style='text-align:right;padding:6px 0'>"
            f"{_chip('파이프라인 ' + str(ps.get('success_criteria_pass_fail','-')), 'ok' if ok else 'bad')}</div>",
            unsafe_allow_html=True,
        )


def render_kpi_cards(df_app: pd.DataFrame, app: str):
    topic_count = load_json("metrics.json").get("topic_modeling", {}).get("topic_count", "-")
    n = len(df_app)
    avg = round(df_app["rating"].mean(), 2) if n else 0
    neg = (df_app["rating"] <= 2).mean() if n else 0

    # 조건부 색: 평점↑ 좋음, 부정률↑ 나쁨
    avg_tone = "ok" if avg >= 3.5 else ("warn" if avg >= 2.5 else "bad")
    neg_tone = "bad" if neg >= 0.5 else ("warn" if neg >= 0.3 else "ok")

    c1, c2, c3, c4 = st.columns(4)
    _kpi_card(c1, f"리뷰 수 · {app}", f"{n:,}", "primary",
              help="선택한 기준 앱·기간에 해당하는 리뷰 건수")
    _kpi_card(c2, "평균 평점", f"{avg}", avg_tone, sub="5점 만점",
              help="선택 구간 리뷰의 평균 별점(5점 만점)")
    _kpi_card(c3, "부정률", f"{neg:.1%}", neg_tone, sub="별점 ≤ 2",
              help="별점 1~2점 리뷰의 비율. 높을수록 불만이 큼")
    _kpi_card(c4, "주요 토픽 수", f"{topic_count}", "neutral", sub="전체 데이터 기준",
              help="토픽 모델링으로 도출된 토픽(군집) 수. 파이프라인 산출물(전체 데이터) 기준")


# ======================================================================
# 차트
# ======================================================================
def render_charts(df_full: pd.DataFrame, df_app: pd.DataFrame, app, competitors):
    st.subheader("📈 시각화")
    col1, col2 = st.columns(2)

    # 리뷰 추이 (데이터 범위에 따라 월별/일별 자동 선택 + 건수·평균평점)
    with col1:
        st.markdown("**리뷰 추이**")
        if df_app["date"].notna().any():
            tmp = df_app.dropna(subset=["date"]).copy()
            span_days = (tmp["date"].max() - tmp["date"].min()).days
            # 기간이 짧으면(≤ 70일) 일별, 길면 월별로 집계 → 2개월 쏠림 같은 빈약함 완화
            if span_days <= 70:
                tmp["구간"] = tmp["date"].dt.strftime("%Y-%m-%d"); unit = "일별"
            else:
                tmp["구간"] = tmp["date"].dt.strftime("%Y-%m"); unit = "월별"
            trend = (tmp.groupby("구간")
                     .agg(리뷰수=("rating", "size"), 평균평점=("rating", "mean"))
                     .reset_index().sort_values("구간"))
            trend["평균평점"] = trend["평균평점"].round(2)
            if len(trend) <= 1:
                st.info(f"표시 기간 내 {unit} 구간이 1개뿐이라 추이 그래프가 의미가 적습니다. "
                        f"(총 {int(trend['리뷰수'].sum())}건)")
            else:
                fig = px.bar(trend, x="구간", y="리뷰수", text="리뷰수",
                             color_discrete_sequence=[BRAND["primary"]])
                fig.add_scatter(x=trend["구간"], y=trend["평균평점"], yaxis="y2",
                                name="평균평점", mode="lines+markers+text",
                                text=trend["평균평점"], textposition="top center",
                                line=dict(color=BRAND["warn"]))
                fig.update_traces(textposition="outside", selector=dict(type="bar"))
                fig.update_layout(
                    height=320, margin=dict(t=30), xaxis=dict(type="category", title=unit),
                    yaxis=dict(title="리뷰 수"),
                    yaxis2=dict(overlaying="y", side="right", range=[0, 5], title="평균 평점"),
                    legend=dict(orientation="h", y=1.12, x=0))
                st.plotly_chart(fig, use_container_width=True)
                st.caption(f"{unit} 집계 · 막대=리뷰 수, 선=평균 평점(우측 0~5)")
        else:
            st.info("날짜 데이터가 없습니다.")

    # 감성 분포 (라벨 한글)
    with col2:
        st.markdown("**감성 분포 (별점 기반 약지도)**")
        sent = df_app["rating"].apply(_weak_label).value_counts()
        names_kr = [SENT_KOR.get(s, s) for s in sent.index]
        color_map_kr = {SENT_KOR[k]: v for k, v in SENTIMENT_COLORS.items()}
        fig = px.pie(values=sent.values, names=names_kr,
                     color=names_kr, color_discrete_map=color_map_kr)
        fig.update_layout(height=320, margin=dict(t=20))
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)
    # 불만 유형 TOP5 (선택 기준 앱·기간 데이터에서 실시간 룰 분류)
    with col3:
        st.markdown(f"**{app} 불만 유형 TOP5** · 선택 기간 기준")
        pairs = _live_complaint_top(df_app)
        if pairs:
            cdf = pd.DataFrame(pairs, columns=["유형", "건수"])
            st.plotly_chart(px.bar(cdf, x="건수", y="유형", orientation="h",
                                   color_discrete_sequence=[BRAND["bad"]])
                            .update_layout(height=320, margin=dict(t=20),
                                           yaxis=dict(categoryorder="total ascending")),
                            use_container_width=True)
        else:
            st.info("부정 리뷰가 없어 불만 유형을 집계할 수 없습니다.")

    # 앱별 비교 (평균 평점) — 선택 기간 반영
    with col4:
        st.markdown("**앱별 평균 평점 비교** · 선택 기간 기준")
        apps_show = [app] + (competitors or [])
        comp = df_full[df_full["app_name"].isin(apps_show)]
        if not comp.empty:
            agg = comp.groupby("app_name")["rating"].mean().round(2).reset_index()
            agg["구분"] = agg["app_name"].apply(lambda x: "기준" if x == app else "경쟁")
            st.plotly_chart(px.bar(agg, x="app_name", y="rating", color="구분", text="rating",
                                   labels={"app_name": "앱", "rating": "평균 평점"},
                                   color_discrete_map={"기준": BRAND["primary"], "경쟁": BRAND["neutral"]})
                            .update_layout(height=320, margin=dict(t=20)),
                            use_container_width=True)


# ======================================================================
# 메인: 부정 리뷰 원문 (선택 기준 앱·기간 반영)
# ======================================================================
def render_negative_reviews(df_app: pd.DataFrame):
    neg = df_app[df_app["rating"] <= 2]
    with st.expander(f"🔻 부정 리뷰 원문 보기 · 선택 기간 {len(neg)}건", expanded=False):
        cols = [c for c in ["app_name", "rating", "date", "clean_text"] if c in neg.columns]
        show_table(neg[cols].head(100), height=320)
        st.caption("선택한 기준 앱·기간 기준 상위 100건")


# ======================================================================
# 탭: 경쟁 벤치마킹 (전체 앱 요약은 기간 반영 / 상세표는 산출물)
# ======================================================================
def tab_benchmark(df_period: pd.DataFrame, app: str):
    st.subheader("🏆 경쟁 벤치마킹")

    st.markdown("**전체 앱 요약** · 선택 기간 기준 (실시간 집계)")
    rows = []
    for name, g in df_period.groupby("app_name"):
        neg = (g["rating"] <= 2).mean() if len(g) else 0
        rows.append({"앱": name + (" ⭐" if name == app else ""), "리뷰수": len(g),
                     "평균평점": round(g["rating"].mean(), 2), "부정률": f"{neg:.1%}"})
    st.dataframe(pd.DataFrame(rows).sort_values("평균평점", ascending=False),
                 use_container_width=True, hide_index=True)

    st.divider()
    st.markdown("**상세 벤치마크 지표** · 파이프라인 산출물(전체 데이터 기준)")
    st.caption("⚠️ 최근 30일 변화·기능 키워드 등 상세 지표는 마지막 파이프라인 실행 시점의 "
               "전체 데이터 기준입니다(선택 기간 미반영).")
    bench = load_csv("benchmark_summary.csv")
    if not bench.empty:
        show_table(bench, height=300)
    else:
        st.info("benchmark_summary.csv 가 없습니다. 사이드바 '전체 파이프라인 실행' 후 생성됩니다.")

    with st.expander("📎 경쟁 격차(Gap) 분석 리포트"):
        st.markdown(load_text("competitor_gap_analysis.md") or "competitor_gap_analysis.md 없음")


# ======================================================================
# 탭: 토픽 분석 (산출물 기반)
# ======================================================================
def tab_topics():
    st.subheader("🧩 토픽 분석")
    st.caption("⚠️ 토픽 모델링은 비용이 커 실시간 재계산 대신 **파이프라인 산출물**(전체 데이터 기준)을 "
               "표시합니다. 선택 기간을 반영하려면 사이드바 '전체 파이프라인 실행'을 다시 수행하세요.")
    topic = load_csv("topic_summary.csv")
    if not topic.empty:
        cols = [c for c in ["topic_id", "size", "auto_label", "topic_label",
                            "top_keywords", "rep_review_1"] if c in topic.columns]
        show_table(topic[cols], height=360)
        st.caption("'토픽명(검수)' 컬럼은 검수자가 최종 토픽명을 입력하는 칸입니다.")
    else:
        st.info("topic_summary.csv 가 없습니다.")


# ======================================================================
# 탭: AI 리포트
# ======================================================================
def _build_report_inputs(df_app, df_period, app):
    """현재 필터(기준 앱·기간) 기준으로 리포트 입력 dict 를 구성한다."""
    n = len(df_app)
    neg = (df_app["rating"] <= 2).mean() if n else 0
    sent = df_app["rating"].apply(_weak_label).value_counts().to_dict()
    bench_rows = [{"app_name": name, "review_count": len(g),
                   "avg_rating": round(g["rating"].mean(), 2),
                   "negative_ratio": round((g["rating"] <= 2).mean(), 3)}
                  for name, g in df_period.groupby("app_name")]
    text_col = "clean_text" if "clean_text" in df_app.columns else "review_text"
    rag = [{"source": f"{r['app_name']} 리뷰", "text": str(r[text_col])[:120]}
           for _, r in df_app[df_app["rating"] <= 2].head(3).iterrows()]
    return {
        "kpi": {"our_app": app, "total_reviews": n,
                "our_avg_rating": round(df_app["rating"].mean(), 2) if n else 0,
                "our_negative_ratio": round(neg, 3)},
        "sentiment_dist": sent,
        "top_complaints": {app: _live_complaint_top(df_app)},
        "topic_summary": [],  # 토픽은 산출물(전체 기준)
        "benchmark": bench_rows,
        "rag_evidence": rag,
    }


def tab_ai_report(df_app, df_period, app):
    st.subheader("🤖 AI 요약 리포트")

    # 생성 방식 = 실제 생성 기록(performance_log) 단일 진실원본
    perf = load_json("performance_log.json").get("report_generation_last", {})
    if perf:
        if perf.get("used_fallback"):
            st.warning(f"이 리포트는 **룰 기반 fallback** 으로 생성되었습니다 · 소요 {perf.get('elapsed_sec')}s")
        else:
            st.success(f"이 리포트는 **LLM ({perf.get('model')})** 로 생성되었습니다 · 소요 {perf.get('elapsed_sec')}s")

    # 현재 필터로 재생성 (선택 기준 앱·기간 반영)
    oai = live_openai_status()
    cap = f"현재 필터(기준 앱 {app}, 선택 기간)로 리포트를 다시 생성합니다."
    cap += " LLM 사용" if oai["enabled"] else " (OpenAI 미설정 → 룰 기반)"
    st.caption(cap)
    if st.button("🔁 현재 필터 기준으로 리포트 재생성"):
        from src.report_generator import generate_report
        with st.spinner("리포트 생성 중..."):
            generate_report(_build_report_inputs(df_app, df_period, app), save=True)
        st.cache_data.clear()
        st.session_state["_flash"] = "리포트 재생성 완료"
        st.rerun()

    report = load_text("ai_report.md")
    if report:
        st.markdown(report)
    else:
        st.info("ai_report.md 가 없습니다. 위 '재생성' 또는 사이드바 파이프라인 실행으로 생성하세요.")

    with st.expander("📎 근거 문서 (경쟁 격차 분석)"):
        st.markdown(load_text("competitor_gap_analysis.md") or "competitor_gap_analysis.md 없음")


# ======================================================================
# 탭: 답글 생성
# ======================================================================
def tab_reply_generation(df_app: pd.DataFrame):
    st.subheader("✍️ 리뷰 답글 초안 생성")
    st.warning("⚠️ 생성된 답글은 **자동 게시되지 않습니다.** 담당자 검수 후 복사·활용하는 **초안**입니다.")
    oai = live_openai_status()
    if oai["enabled"]:
        st.info(f"OpenAI API 사용 가능 (모델: {oai['model']}) → '재생성' 시 LLM 으로 새 초안을 만듭니다.")
    else:
        st.info("OpenAI API 미사용 → 룰 기반 fallback 으로 초안을 생성합니다.")

    drafts = load_csv("reply_drafts.csv")

    # 대상 풀: 기준 앱·기간 필터된 리뷰 (비면 전체)
    source = df_app if not df_app.empty else load_reviews()
    if source.empty:
        st.info("리뷰 데이터가 없습니다.")
        return

    # 대상 출처: 기존 데이터에서 선택 / 직접 붙여넣기
    mode = st.radio("답글 대상", ["기존 리뷰에서 선택", "직접 붙여넣기"], horizontal=True)

    if mode == "직접 붙여넣기":
        default_app = df_app["app_name"].iloc[0] if not df_app.empty else OUR_APP
        row = _custom_row(default_app)
        if row is None:
            return
    else:
        # 대상 선정 기준 선택 (우선순위순 / 최신순)
        order = st.radio("대상 선정 기준", ["우선순위순", "최신순"], horizontal=True,
                         help="우선순위순 = 검수 필요(심각) → 부정(낮은 별점) → 최신 순. "
                              "최신순 = 날짜가 가장 최근인 리뷰부터.")
        cand = _reply_candidates(source, order, limit=50)
        if cand.empty:
            st.info("표시할 리뷰가 없습니다.")
            return

        def _label(i):
            r = cand.loc[i]
            d = r["date"].date() if pd.notna(r["date"]) else "-"
            txt = str(r["clean_text"])[:40]
            return f"{'★'*int(r['rating']) if pd.notna(r['rating']) else '-'} · {d} · {r['app_name']} · {txt}…"

        sel_idx = st.selectbox("답글을 생성할 리뷰 선택", cand.index.tolist(), format_func=_label)
        row = cand.loc[sel_idx]

    sel = str(row["clean_text"])
    rating = int(row["rating"]) if pd.notna(row.get("rating")) else None
    sentiment = _weak_label(rating)

    # 이슈/심각도: 사전 초안에 있으면 사용, 없으면 즉석 분류
    pre = drafts[drafts["review_text"] == sel] if not drafts.empty else pd.DataFrame()
    if not pre.empty:
        issue_types = str(pre.iloc[0].get("issue_types", ""))
        severity = str(pre.iloc[0].get("severity", "medium"))
    else:
        from src.issue_classifier import classify_issue_type, classify_severity
        it = classify_issue_type(sel)
        severity = classify_severity(sel, rating, it)
        issue_types = ", ".join(it)

    # ── 답글 대상 리뷰 정보 카드 ──
    sev_tone = {"high": "bad", "medium": "warn", "low": "neutral", "positive": "ok"}.get(severity, "neutral")
    with st.container(border=True):
        st.markdown("**🎯 답글 대상 리뷰**")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("앱", row.get("app_name", "-"))
        m2.metric("별점", "★" * rating if rating else "-")
        d = row["date"].date() if pd.notna(row["date"]) else "-"
        m3.metric("날짜", str(d))
        m4.metric("스토어", STORE_KOR.get(row.get("store"), row.get("store", "-")))
        sent_kr = SENT_KOR.get(sentiment, sentiment)
        meta = (f"{_chip('감성 · ' + sent_kr, 'ok' if sentiment=='positive' else ('bad' if sentiment=='negative' else 'neutral'))} "
                f"{_chip('심각도 · ' + severity, sev_tone)} "
                f"{_chip('이슈 · ' + (issue_types or '기타'), 'primary')}")
        ver = row.get("version")
        if pd.notna(ver) and str(ver).strip():
            meta += f" {_chip('버전 · ' + str(ver), 'neutral')}"
        st.markdown(meta, unsafe_allow_html=True)
        # 원문(가능하면 정제 전 원문) 표시
        original = row.get("review_text")
        body = str(original) if (pd.notna(original) and str(original).strip()) else sel
        st.text_area("리뷰 원문", value=body, height=90, disabled=True, key="target_review_body")

    # LLM 에 항상 반영할 운영자 지침
    # 영구 키(llm_guidance_val) + 위젯 키(llm_guidance_widget) 분리:
    #  - 위젯이 미렌더로 GC 되어도 영구 키에서 재시드 → 초기화 방지
    #  - on_change 로 편집 즉시 영구 키에 반영 → 수정 반영 보장
    from src.reply_generator import load_policy
    if "llm_guidance_val" not in st.session_state:
        st.session_state["llm_guidance_val"] = str(load_policy().get("default_llm_guidance", "") or "")
    st.session_state.setdefault("llm_guidance_widget", st.session_state["llm_guidance_val"])

    def _sync_guidance():
        st.session_state["llm_guidance_val"] = st.session_state["llm_guidance_widget"]

    st.text_area(
        "🧭 LLM 반영 지침 (답글 생성 시 항상 반영)", key="llm_guidance_widget",
        height=80, on_change=_sync_guidance,
        help="예: 진행 중인 이벤트 안내, 특정 표현 사용/회피 등. "
             "금지 원칙(개인정보 요청·확정 약속 등)과 충돌하면 금지 원칙이 우선합니다. "
             "OpenAI 미사용(룰 기반)일 때는 반영되지 않습니다.")
    # 위젯의 현재값을 직접 사용(편집 즉시 반영). 영구 키는 on_change 로 보존만 담당.
    guidance = st.session_state["llm_guidance_widget"]

    # 세션 상태로 답글 보관 (재생성/수정 지원)
    key = "reply_text"
    colA, colB = st.columns(2)
    gen = colA.button("답글 생성 / 재생성", type="primary")
    if gen or key not in st.session_state:
        with st.spinner("답글 초안 생성 중..."):
            if not pre.empty and not gen:
                # 사전 생성 초안 우선(오프라인 데모)
                draft = str(pre.iloc[0]["reply_draft"])
                needs = bool(pre.iloc[0].get("needs_human_review", False))
                flags = str(pre.iloc[0].get("safety_flags", ""))
            else:
                from src.reply_generator import generate_reply
                res = generate_reply(sel, rating, sentiment,
                                     [s.strip() for s in issue_types.split(",") if s.strip()],
                                     severity, row.get("app_name", OUR_APP),
                                     extra_guidance=guidance)
                draft, needs, flags = res["reply_draft"], res["needs_human_review"], ", ".join(res["safety_flags"])
            st.session_state[key] = draft
            st.session_state["reply_needs"] = needs
            st.session_state["reply_flags"] = flags

    # 수정 가능한 텍스트 영역 + 복사/다운로드
    edited = st.text_area("답글 초안 (수정 가능)", value=st.session_state.get(key, ""), height=140)
    if st.session_state.get("reply_needs"):
        st.warning(f"🔒 담당자 검수 필요 (needs_human_review=True) · 플래그: {st.session_state.get('reply_flags') or '없음'}")
    else:
        st.success("자동 안전성 점검 통과 — 그래도 게시 전 검수를 권장합니다.")

    st.code(edited, language=None)  # 복사용 (코드블록 우상단 복사 버튼)
    st.caption("위 박스 우측 상단 아이콘으로 복사할 수 있습니다.")

    # 전체 초안 CSV 다운로드
    if not drafts.empty:
        st.download_button("📥 전체 답글 초안 CSV 다운로드",
                           drafts.to_csv(index=False).encode("utf-8-sig"),
                           file_name="reply_drafts.csv", mime="text/csv")


# ======================================================================
# 탭: 데이터 수집 (앱 등록 + 스토어 수집 + 미리보기 + 수동 입력)
# ======================================================================
RAW_PATH = PATHS.raw_dir / "review_raw.csv"


def _apply_and_preprocess(new_df, mode: str):
    """수집/수동 데이터를 review_raw.csv 에 반영(병합/덮어쓰기)하고 전처리까지 갱신."""
    from src.app_registry import merge_reviews
    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    if mode == "merge" and RAW_PATH.exists():
        base = pd.read_csv(RAW_PATH)
        final = merge_reviews(base, new_df)
    else:  # overwrite
        final = new_df.copy()
    final.to_csv(RAW_PATH, index=False, encoding="utf-8-sig")
    # 전처리 재실행 → review_clean.csv 갱신 (대시보드가 즉시 반영)
    try:
        from src.preprocess import preprocess
        preprocess()
        msg = f"review_raw.csv {len(final)}건 저장 + 전처리 완료"
    except Exception as exc:
        msg = f"raw 저장({len(final)}건)했으나 전처리 실패: {exc!r}"
    st.cache_data.clear()
    return msg


def render_collection_sidebar():
    """사이드바(분석 설정)에 데이터 수집 컨트롤을 렌더링한다."""
    from src.app_registry import load_registry, collect_apps
    registry = load_registry()
    app_names = [a["name"] for a in registry.get("apps", [])]
    defaults = registry.get("defaults", {"stores": ["google_play", "app_store"], "n_per_store": 300})

    with st.sidebar.expander("🗂️ 리뷰 수집", expanded=False):
        use_default = st.checkbox(
            "기본 세팅으로 진행", value=False,
            help="기본 세팅 = apps.yaml 의 등록된 전체 앱 · Google Play+App Store · "
                 "앱당 300건. App Store 는 공개 RSS 특성상 앱당 ~100~500건 상한이 있습니다.",
        )
        if use_default:
            sel_apps = app_names
            stores = defaults.get("stores", ["google_play", "app_store"])
            n_per = int(defaults.get("n_per_store", 300))
            st.caption(f"기본: 앱 {len(sel_apps)}개 · {stores} · 앱당 {n_per}건")
        else:
            sel_apps = st.multiselect("수집 대상 앱", app_names, default=app_names[:3])
            stores = st.multiselect("스토어", ["google_play", "app_store"],
                                    default=defaults.get("stores", ["google_play", "app_store"]))
            n_per = st.slider("앱당 최대 리뷰 수", 10, 500,
                              int(defaults.get("n_per_store", 300)), step=10)
            st.caption("ℹ️ '수집 일정'은 데모 범위에서 **분량·스토어 선택**으로 제공합니다.")

        if st.button("📥 리뷰 수집", type="primary",
                     disabled=not sel_apps or not stores, use_container_width=True):
            with st.spinner(f"{len(sel_apps)}개 앱 수집 중..."):
                df_new, report = collect_apps(sel_apps, stores, n_per_store=n_per, registry=registry)
            st.session_state["collected_df"] = df_new
            st.session_state["collected_report"] = report
            st.session_state["show_preview"] = True

        if st.button("✏️ 수동 입력 / 앱 관리", use_container_width=True):
            st.session_state["show_manual"] = True


@st.dialog("수집 결과 미리보기", width="large")
def collection_preview_dialog():
    df_new = st.session_state.get("collected_df")
    if df_new is None:
        st.session_state["show_preview"] = False
        return
    rep = st.session_state.get("collected_report", [])
    if rep:
        show_table(pd.DataFrame(rep))
    st.write(f"총 **{len(df_new)}건** 수집 (아직 저장 안 됨)")
    show_table(df_new.head(50), height=300)

    c1, c2, c3 = st.columns(3)
    if c1.button("✅ 기존에 병합", use_container_width=True):
        st.session_state["_flash"] = _apply_and_preprocess(df_new, "merge")
        st.session_state.pop("collected_df", None); st.session_state["show_preview"] = False
        st.rerun()
    if c2.button("♻️ 덮어쓰기", use_container_width=True):
        st.session_state["_flash"] = _apply_and_preprocess(df_new, "overwrite")
        st.session_state.pop("collected_df", None); st.session_state["show_preview"] = False
        st.rerun()
    if c3.button("🗑️ 버리기", use_container_width=True):
        st.session_state.pop("collected_df", None); st.session_state["show_preview"] = False
        st.rerun()


@st.dialog("수동 입력 / 앱 등록 관리", width="large")
def manual_registry_dialog():
    from src.app_registry import (
        load_registry, save_registry, apps_dataframe, registry_from_dataframe, normalize_manual,
    )
    registry = load_registry()
    defaults = registry.get("defaults", {"stores": ["google_play", "app_store"], "n_per_store": 300})

    st.markdown("#### 수동 입력 (결과 보정)")
    st.caption("표준 컬럼: app_name, store, rating, date, review_text, version, source_url")
    m1, m2 = st.tabs(["CSV 업로드", "직접 편집"])
    with m1:
        up = st.file_uploader("리뷰 CSV", type=["csv"], key="manual_csv")
        if up is not None:
            man = normalize_manual(pd.read_csv(up))
            st.dataframe(man.head(20), use_container_width=True, height=200)
            cc1, cc2 = st.columns(2)
            if cc1.button("✅ 병합", key="mu_merge"):
                st.session_state["_flash"] = _apply_and_preprocess(man, "merge")
                st.session_state["show_manual"] = False; st.rerun()
            if cc2.button("♻️ 덮어쓰기", key="mu_over"):
                st.session_state["_flash"] = _apply_and_preprocess(man, "overwrite")
                st.session_state["show_manual"] = False; st.rerun()
    with m2:
        empty = pd.DataFrame([{"app_name": "", "store": "sample", "rating": 5,
                               "date": "", "review_text": "", "version": "", "source_url": ""}])
        edited = st.data_editor(empty, num_rows="dynamic", use_container_width=True, key="manual_editor")
        if st.button("✅ 편집 내용 병합", key="me_merge"):
            man = normalize_manual(edited)
            man = man[man["review_text"].astype(str).str.strip() != ""]
            if man.empty:
                st.warning("review_text 가 입력된 행이 없습니다.")
            else:
                st.session_state["_flash"] = _apply_and_preprocess(man, "merge")
                st.session_state["show_manual"] = False; st.rerun()

    st.divider()
    st.markdown("#### 앱 등록 관리")
    st.caption("구글플레이 패키지명 / 앱스토어 숫자 ID 를 입력하세요.")
    edited_reg = st.data_editor(
        apps_dataframe(registry), num_rows="dynamic",
        use_container_width=True, key="registry_editor",
        column_config={
            "name": st.column_config.TextColumn("앱 이름"),
            "google_play": st.column_config.TextColumn("구글플레이 패키지명"),
            "app_store": st.column_config.TextColumn("앱스토어 ID"),
            "is_our_app": st.column_config.CheckboxColumn("기준앱"),
        })
    if st.button("💾 앱 목록 저장", key="reg_save"):
        new_reg = registry_from_dataframe(edited_reg, registry.get("our_app", "슈퍼SOL"), defaults)
        save_registry(new_reg)
        st.cache_data.clear()
        st.session_state["_flash"] = f"앱 {len(new_reg['apps'])}개 저장 완료"
        st.session_state["show_manual"] = False; st.rerun()
    if st.button("닫기", key="manual_close"):
        st.session_state["show_manual"] = False; st.rerun()


# ======================================================================
# 탭: 상세 데이터 (전체 앱 리뷰 열람/검색/다운로드)
# ======================================================================
def tab_detail_data(df_full: pd.DataFrame):
    st.subheader("📄 상세 데이터")
    st.caption("모든 앱의 리뷰 원본을 필터·검색해 열람합니다. (기준 앱에 한정되지 않음)")

    apps = sorted(df_full["app_name"].dropna().unique().tolist())
    c1, c2, c3 = st.columns([2, 1, 1])
    sel = c1.multiselect("앱", apps, default=apps)
    stores = sorted(df_full["store"].dropna().unique().tolist()) if "store" in df_full else []
    sel_store = c2.multiselect("스토어", stores, default=stores,
                               format_func=lambda s: STORE_KOR.get(s, s))
    rmin, rmax = c3.select_slider("평점", options=[1, 2, 3, 4, 5], value=(1, 5))
    kw = st.text_input("리뷰 본문 검색(부분 일치)", "")

    view = df_full[df_full["app_name"].isin(sel)].copy()
    if sel_store and "store" in view:
        view = view[view["store"].isin(sel_store)]
    view = view[(view["rating"] >= rmin) & (view["rating"] <= rmax)]
    text_col = "clean_text" if "clean_text" in view.columns else "review_text"
    if kw.strip():
        view = view[view[text_col].astype(str).str.contains(kw.strip(), case=False, na=False)]

    st.write(f"필터 결과: **{len(view)}건**")
    cols = [c for c in ["app_name", "store", "rating", "date", text_col,
                        "version", "source_url"] if c in view.columns]
    show_table(view[cols], height=420)
    st.download_button("📥 필터 결과 CSV 다운로드",
                       view[cols].to_csv(index=False).encode("utf-8-sig"),
                       file_name="reviews_filtered.csv", mime="text/csv")


# ======================================================================
# 탭: 검증
# ======================================================================
def tab_validation():
    st.subheader("✅ 품질 검증 결과")

    v1, v2, v3 = st.columns(3)
    metrics = load_json("metrics.json")
    dq = metrics.get("data_quality_pass")
    v1.metric("데이터 품질", "PASS" if dq else "FAIL/-")
    fin = metrics.get("final_model", {})
    v2.metric("감성 모델 F1(별점)", fin.get("f1_macro_star", "-"))
    re_last = load_json("performance_log.json").get("reply_generation_last", {})
    v3.metric("답글 fallback 건수", re_last.get("fallback_count", "-"))

    with st.expander("데이터 검증 리포트", expanded=True):
        st.markdown(load_text("data_validation_report.md") or "없음")
    with st.expander("감성 모델 성능 리포트"):
        st.markdown(load_text("model_report.md") or "없음")
    with st.expander("답글 품질 평가 리포트"):
        st.markdown(load_text("reply_eval_report.md") or "없음")
    with st.expander("실패/리스크 케이스"):
        st.markdown("**감성 오분류 사례**")
        show_table(load_csv("error_cases.csv").head(20))
        st.markdown("**답글 안전성 리스크 케이스**")
        risk = load_csv("reply_risk_cases.csv")
        if risk.empty:
            st.success("답글 안전성 위반 케이스 0건")
        else:
            show_table(risk)


# ======================================================================
# 메인
# ======================================================================
def main():
    st.title("📊 슈퍼SOL 고객 반응 분석 대시보드")
    st.caption("앱 리뷰 기반 VOC 분석 · 경쟁 앱 벤치마킹 · 리뷰 답글 초안 생성(검수용)")

    ctx = render_sidebar()   # 수집 컨트롤 포함
    df_full = ctx["df"]

    if st.session_state.pop("_flash", None):
        st.toast("완료 ✅")

    # 수집 미리보기 / 수동 입력 모달 (세션 플래그로 제어)
    # 플래그를 '열 때 즉시 소비(pop)'한다. st.dialog 는 열린 뒤 내부 위젯 rerun 동안
    # Streamlit 이 자동으로 열린 상태를 유지하므로, 매 rerun 마다 재호출하면 안 된다.
    # (sticky 플래그로 두면 ✕/바깥클릭/ESC 네이티브 닫기 후에도 플래그가 True 로 남아
    #  다른 조작 때마다 모달이 다시 열리는 버그가 발생한다.)
    if st.session_state.pop("show_preview", False) and st.session_state.get("collected_df") is not None:
        collection_preview_dialog()
    if st.session_state.pop("show_manual", False):
        manual_registry_dialog()

    # 데이터가 없으면 안내 (수집은 사이드바 '1 · 데이터 소스' 에서)
    if df_full.empty:
        st.info("아직 분석할 리뷰 데이터가 없습니다. 좌측 사이드바 **'1 · 데이터 소스 → 🗂️ 리뷰 수집'** "
                "에서 앱을 고르고 리뷰를 가져오거나, '✏️ 수동 입력' 으로 직접 추가하세요.")
        return

    app = ctx["app"]
    # 기간 필터를 전체 앱에 적용(df_period) → 기준 앱은 거기서 추출(df_app)
    df_period = apply_filters(df_full, None, ctx["date_range"])
    df_app = df_period[df_period["app_name"] == app]

    # 컨텍스트 배지 + 파이프라인 상태 (상시)
    render_context_bar(df_app, df_period, app, ctx["date_range"])

    # ── 핵심 분석 (기준 앱·기간 실시간 반영) ──
    render_kpi_cards(df_app, app)
    render_charts(df_period, df_app, app, ctx["competitors"])
    render_negative_reviews(df_app)
    st.divider()

    # ── 탭: 우선순위 순 (원본 리뷰는 맨 끝 서브) ──
    tabs = st.tabs(["🤖 AI 리포트", "🏆 경쟁 벤치마킹", "🧩 토픽 분석",
                    "💬 답글 생성", "✅ 검증", "📄 원본 리뷰"])
    with tabs[0]:
        tab_ai_report(df_app, df_period, app)
    with tabs[1]:
        tab_benchmark(df_period, app)
    with tabs[2]:
        tab_topics()
    with tabs[3]:
        tab_reply_generation(df_app)
    with tabs[4]:
        tab_validation()
    with tabs[5]:
        tab_detail_data(df_full)


if __name__ == "__main__":
    main()
