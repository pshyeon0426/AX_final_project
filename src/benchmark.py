"""
경쟁 앱 벤치마킹 분석 (benchmark.py)

목적:
- 슈퍼SOL 과 경쟁 앱(3~5개)의 공개 리뷰/평점/불만 유형/토픽/공개 기능 키워드를
  비교해 벤치마킹 테이블과 격차(gap) 분석 리포트를 생성한다.

⚠️ 분석 원칙:
- **공개 리뷰 + 공개 문서(docs/competitors/)** 기준으로만 작성한다.
- 경쟁사 내부 전략/비공개 지표는 근거 없이 추정하지 않는다.
- 기능 키워드는 공개 리뷰에서 추출하며, 공개 설명 문서가 있으면 함께 반영한다.

실행:
    python -m src.benchmark
"""
from __future__ import annotations

import json
import logging
import re

import pandas as pd

from src.config import PATHS, settings
from src.issue_classifier import classify_dataframe, ISSUE_OTHER
from src.tokenizer import get_tokenizer

logging.basicConfig(
    level=getattr(logging, settings.log_level, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("benchmark")

# ----------------------------------------------------------------------
# 경로 / 상수
# ----------------------------------------------------------------------
CLEAN_PATH = PATHS.processed_dir / "review_clean.csv"
COMPETITORS_DOC_DIR = PATHS.competitors_dir
BENCHMARK_CSV_PATH = PATHS.outputs_dir / "benchmark_summary.csv"
GAP_MD_PATH = PATHS.outputs_dir / "competitor_gap_analysis.md"
METRICS_PATH = PATHS.outputs_dir / "metrics.json"

OUR_APP = "슈퍼SOL"
NEGATIVE_RATING_MAX = 2   # rating ≤ 2 → 부정
RECENT_DAYS = 30
TOP_N = 5

# 공개 기능 키워드 사전 (혜택/인증/송금/투자/UI·UX/이벤트)
FEATURE_KEYWORDS = {
    "혜택": ["혜택", "포인트", "적립", "캐시백", "우대", "리워드", "할인", "쿠폰"],
    "인증": ["인증", "공동인증", "공인인증", "인증서", "otp", "지문", "안면", "얼굴", "신분증"],
    "송금": ["송금", "이체", "출금", "입금", "자동이체", "수수료", "한도"],
    "투자": ["투자", "펀드", "주식", "적금", "예금", "환율", "외화", "청약", "수익률"],
    "UI/UX": ["ui", "ux", "화면", "디자인", "메뉴", "직관", "편하", "편리", "가독성"],
    "이벤트": ["이벤트", "복권", "출석", "미션", "룰렛", "응모", "경품"],
}


def _normalize(text: str) -> str:
    return re.sub(r"\s+", "", str(text).lower()) if text else ""


# ----------------------------------------------------------------------
# 데이터 로딩 + 분류
# ----------------------------------------------------------------------
def load_classified() -> pd.DataFrame:
    if not CLEAN_PATH.exists():
        raise FileNotFoundError(f"정제 데이터가 없습니다: {CLEAN_PATH}")
    df = pd.read_csv(CLEAN_PATH)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df = classify_dataframe(df)  # issue_types 컬럼 추가 (룰 기반)
    return df


# ----------------------------------------------------------------------
# 앱별 핵심 지표
# ----------------------------------------------------------------------
def _recent_rating_change(g: pd.DataFrame, ref_date) -> float | None:
    """최근 30일 평점 - 직전 30일 평점 (둘 다 데이터 있을 때만)."""
    if ref_date is None or g["date"].isna().all():
        return None
    recent_start = ref_date - pd.Timedelta(days=RECENT_DAYS)
    prev_start = ref_date - pd.Timedelta(days=2 * RECENT_DAYS)
    recent = g[(g["date"] > recent_start) & (g["date"] <= ref_date)]["rating"]
    prev = g[(g["date"] > prev_start) & (g["date"] <= recent_start)]["rating"]
    if recent.empty or prev.empty:
        return None
    return round(float(recent.mean() - prev.mean()), 3)


def app_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """앱별 평균 평점/부정 비율/리뷰 수/최근 30일 평점 변화량."""
    ref_date = df["date"].max()
    logger.info("최근 30일 기준일: %s", ref_date.date() if pd.notna(ref_date) else "N/A")

    rows = []
    for app, g in df.groupby("app_name"):
        neg_ratio = float((g["rating"] <= NEGATIVE_RATING_MAX).mean())
        rows.append({
            "app_name": app,
            "review_count": int(len(g)),
            "avg_rating": round(float(g["rating"].mean()), 3),
            "negative_ratio": round(neg_ratio, 3),
            "recent30d_rating_change": _recent_rating_change(g, ref_date),
        })
    out = pd.DataFrame(rows).sort_values("avg_rating", ascending=False).reset_index(drop=True)
    return out


# ----------------------------------------------------------------------
# 앱별 불만 유형 TOP5 / 토픽 키워드 TOP5
# ----------------------------------------------------------------------
def top_complaint_types(df: pd.DataFrame, top_n: int = TOP_N) -> dict:
    """앱별 불만 유형 TOP N (부정 리뷰 기준, 기타 제외)."""
    neg = df[df["rating"] <= NEGATIVE_RATING_MAX].explode("issue_types")
    neg = neg[neg["issue_types"] != ISSUE_OTHER]
    result = {}
    for app, g in neg.groupby("app_name"):
        result[app] = list(g["issue_types"].value_counts().head(top_n).items())
    return result


def top_topic_keywords(df: pd.DataFrame, top_n: int = TOP_N) -> dict:
    """앱별 주요 토픽 키워드 TOP N (TF-IDF 상위 단어)."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    tok = get_tokenizer()
    result = {}
    for app, g in df.groupby("app_name"):
        texts = g["clean_text"].fillna("").astype(str)
        texts = texts[texts.str.strip() != ""]
        if len(texts) < 5:
            result[app] = []
            continue
        try:
            vec = TfidfVectorizer(tokenizer=tok.tokenize, token_pattern=None,
                                  ngram_range=(1, 2), min_df=3, max_df=0.5)
            X = vec.fit_transform(texts)
            scores = X.sum(axis=0).A1
            terms = vec.get_feature_names_out()
            order = scores.argsort()[::-1][:top_n]
            result[app] = [(terms[i], round(float(scores[i]), 2)) for i in order]
        except ValueError:
            result[app] = []
    return result


# ----------------------------------------------------------------------
# 공개 기능 키워드 추출 (리뷰 + 공개 문서)
# ----------------------------------------------------------------------
def _load_competitor_docs() -> dict[str, str]:
    """docs/competitors/ 의 공개 설명 문서를 앱별로 읽는다(있으면)."""
    docs = {}
    if not COMPETITORS_DOC_DIR.exists():
        return docs
    for p in COMPETITORS_DOC_DIR.glob("*.md"):
        if p.name.lower() == "readme.md":
            continue
        docs[p.stem] = _normalize(p.read_text(encoding="utf-8"))
    return docs


def feature_keyword_mentions(df: pd.DataFrame) -> pd.DataFrame:
    """앱별 공개 기능 카테고리 언급률(혜택/인증/송금/투자/UI·UX/이벤트).

    리뷰 텍스트 기준 언급 비율(%) + 공개 문서가 있으면 해당 카테고리 매칭 여부 반영.
    """
    docs = _load_competitor_docs()
    if docs:
        logger.info("공개 설명 문서 %d개 반영: %s", len(docs), list(docs.keys()))
    else:
        logger.info("공개 설명 문서 없음 → 공개 리뷰 기준으로만 기능 키워드 추출.")

    rows = []
    for app, g in df.groupby("app_name"):
        norm_texts = g["clean_text"].fillna("").apply(_normalize)
        n = len(norm_texts)
        row = {"app_name": app}
        for feat, kws in FEATURE_KEYWORDS.items():
            mention = norm_texts.apply(lambda t: any(k in t for k in kws)).sum()
            row[f"feat_{feat}_mention_rate"] = round(mention / n, 3) if n else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# 격차(gap) 분석
# ----------------------------------------------------------------------
def gap_analysis(metrics: pd.DataFrame, complaints: dict, our_app: str = OUR_APP) -> dict:
    """슈퍼SOL 대비 경쟁 앱의 강점/약점/개선 기회 정리 (공개 데이터 근거)."""
    m = metrics.set_index("app_name")
    if our_app not in m.index:
        return {"error": f"{our_app} 데이터 없음"}

    our = m.loc[our_app]
    competitors = [a for a in m.index if a != our_app]

    strengths, weaknesses, opportunities = [], [], []

    for comp in competitors:
        c = m.loc[comp]
        # 경쟁 앱이 우리보다 평점이 높으면 → 경쟁 앱 강점 / 우리 약점
        if c["avg_rating"] > our["avg_rating"]:
            strengths.append(
                f"{comp}: 평균 평점 {c['avg_rating']} (슈퍼SOL {our['avg_rating']}보다 높음)"
            )
        if c["negative_ratio"] < our["negative_ratio"]:
            strengths.append(
                f"{comp}: 부정 리뷰 비율 {c['negative_ratio']:.1%} "
                f"(슈퍼SOL {our['negative_ratio']:.1%}보다 낮음)"
            )

    # 슈퍼SOL 자체 약점: 주요 불만 유형
    our_complaints = complaints.get(our_app, [])
    if our_complaints:
        top_str = ", ".join(f"{t}({c})" for t, c in our_complaints[:3])
        weaknesses.append(f"슈퍼SOL 주요 불만 유형: {top_str}")

    # 개선 기회: 경쟁 앱 대비 평점/부정비율 위치
    rank = (m["avg_rating"].rank(ascending=False).loc[our_app])
    opportunities.append(
        f"슈퍼SOL 평점 순위: {int(rank)}/{len(m)}위 "
        f"(평균 {our['avg_rating']}, 부정 {our['negative_ratio']:.1%})"
    )
    # 경쟁 앱 공통 불만이 적은 영역 → 차별화 기회
    all_comp_types = {}
    for comp in competitors:
        for t, cnt in complaints.get(comp, []):
            all_comp_types[t] = all_comp_types.get(t, 0) + cnt
    if all_comp_types:
        worst = max(all_comp_types, key=all_comp_types.get)
        opportunities.append(
            f"경쟁 앱 공통 최다 불만 유형='{worst}' → 이 영역을 잘 처리하면 차별화 가능"
        )

    return {
        "our_app": our_app,
        "strengths_of_competitors": strengths or ["공개 지표상 경쟁 앱이 뚜렷이 앞서는 항목 없음"],
        "weaknesses_of_our_app": weaknesses or ["주요 불만 유형 데이터 부족"],
        "opportunities": opportunities,
        "basis": "공개 리뷰/공개 문서 기준. 경쟁사 내부 전략은 추정하지 않음.",
    }


# ----------------------------------------------------------------------
# 저장 + 통합 실행
# ----------------------------------------------------------------------
def _save_csv(metrics, features, complaints, topics) -> pd.DataFrame:
    """벤치마킹 요약 테이블(앱 1행) 저장."""
    summary = metrics.merge(features, on="app_name", how="left")
    summary["top_complaint_types"] = summary["app_name"].map(
        lambda a: " | ".join(f"{t}:{c}" for t, c in complaints.get(a, [])) or "-"
    )
    summary["top_topic_keywords"] = summary["app_name"].map(
        lambda a: " | ".join(t for t, _ in topics.get(a, [])) or "-"
    )
    BENCHMARK_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(BENCHMARK_CSV_PATH, index=False, encoding="utf-8-sig")
    logger.info("벤치마킹 요약 저장: %s", BENCHMARK_CSV_PATH)
    return summary


def _save_gap_md(metrics, complaints, topics, features, gap) -> None:
    lines = [
        "# 경쟁 앱 벤치마킹 · 격차(Gap) 분석",
        "",
        "> ⚠️ **분석 근거**: 공개 앱 리뷰 + 공개 설명 문서 기준. "
        "경쟁사 내부 전략/비공개 지표는 추정하지 않았습니다.",
        "",
        "## 1. 앱별 핵심 지표",
        "",
        "| 앱 | 리뷰수 | 평균평점 | 부정비율 | 최근30일 평점변화 |",
        "|----|------:|--------:|--------:|----------------:|",
    ]
    for _, r in metrics.iterrows():
        chg = r["recent30d_rating_change"]
        chg_str = f"{chg:+.3f}" if pd.notna(chg) else "N/A"
        mark = " ⭐" if r["app_name"] == OUR_APP else ""
        lines.append(
            f"| {r['app_name']}{mark} | {r['review_count']} | {r['avg_rating']} "
            f"| {r['negative_ratio']:.1%} | {chg_str} |"
        )

    lines += ["", "## 2. 앱별 주요 불만 유형 TOP5 (부정 리뷰 기준)", ""]
    for app in metrics["app_name"]:
        items = complaints.get(app, [])
        s = ", ".join(f"{t}({c})" for t, c in items) or "-"
        lines.append(f"- **{app}**: {s}")

    lines += ["", "## 3. 앱별 주요 토픽 키워드 TOP5 (TF-IDF)", ""]
    for app in metrics["app_name"]:
        items = topics.get(app, [])
        s = ", ".join(t for t, _ in items) or "-"
        lines.append(f"- **{app}**: {s}")

    lines += ["", "## 4. 공개 기능 키워드 언급률 (리뷰 기준, %)", "",
              "| 앱 | 혜택 | 인증 | 송금 | 투자 | UI/UX | 이벤트 |",
              "|----|----:|----:|----:|----:|------:|------:|"]
    feat_idx = features.set_index("app_name")
    for app in metrics["app_name"]:
        fr = feat_idx.loc[app]
        lines.append(
            f"| {app} | {fr['feat_혜택_mention_rate']:.1%} | {fr['feat_인증_mention_rate']:.1%} "
            f"| {fr['feat_송금_mention_rate']:.1%} | {fr['feat_투자_mention_rate']:.1%} "
            f"| {fr['feat_UI/UX_mention_rate']:.1%} | {fr['feat_이벤트_mention_rate']:.1%} |"
        )

    lines += ["", "## 5. 슈퍼SOL 대비 강점 / 약점 / 개선 기회", "",
              "### 경쟁 앱의 강점 (슈퍼SOL이 따라가야 할 점)"]
    for s in gap["strengths_of_competitors"]:
        lines.append(f"- {s}")
    lines += ["", "### 슈퍼SOL의 약점"]
    for w in gap["weaknesses_of_our_app"]:
        lines.append(f"- {w}")
    lines += ["", "### 개선 기회"]
    for o in gap["opportunities"]:
        lines.append(f"- {o}")
    lines += ["", f"> {gap['basis']}", ""]

    GAP_MD_PATH.write_text("\n".join(lines), encoding="utf-8")
    logger.info("격차 분석 리포트 저장: %s", GAP_MD_PATH)


def _merge_metrics(metrics, gap) -> None:
    existing = {}
    if METRICS_PATH.exists():
        try:
            with open(METRICS_PATH, encoding="utf-8") as f:
                existing = json.load(f)
        except Exception as exc:
            logger.warning("metrics.json 로드 실패 (%r).", exc)
    m = metrics.set_index("app_name")
    existing["benchmark"] = {
        "app_count": int(len(m)),
        "our_app": OUR_APP,
        "our_avg_rating": float(m.loc[OUR_APP, "avg_rating"]) if OUR_APP in m.index else None,
        "our_negative_ratio": float(m.loc[OUR_APP, "negative_ratio"]) if OUR_APP in m.index else None,
        "our_rating_rank": int(m["avg_rating"].rank(ascending=False).loc[OUR_APP]) if OUR_APP in m.index else None,
        "competitor_apps": [a for a in m.index if a != OUR_APP],
    }
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)
    logger.info("metrics.json 병합 저장: %s", METRICS_PATH)


def run(save: bool = True) -> dict:
    """벤치마킹 분석을 실행하고 Streamlit/AI 리포트용 dict 를 반환한다."""
    df = load_classified()
    metrics = app_metrics(df)
    complaints = top_complaint_types(df)
    topics = top_topic_keywords(df)
    features = feature_keyword_mentions(df)
    gap = gap_analysis(metrics, complaints)

    if save:
        _save_csv(metrics, features, complaints, topics)
        _save_gap_md(metrics, complaints, topics, features, gap)
        _merge_metrics(metrics, gap)

    # Streamlit / AI 리포트 입력용 dict
    return {
        "metrics": metrics.to_dict(orient="records"),
        "top_complaint_types": {a: complaints.get(a, []) for a in metrics["app_name"]},
        "top_topic_keywords": {a: topics.get(a, []) for a in metrics["app_name"]},
        "feature_mentions": features.to_dict(orient="records"),
        "gap_analysis": gap,
        "basis": "공개 리뷰/공개 문서 기준. 경쟁사 내부 전략 비추정.",
    }


if __name__ == "__main__":
    result = run()
    print("\n=== 경쟁 앱 벤치마킹 ===")
    print("※ 공개 리뷰/문서 기준, 경쟁사 내부 전략 비추정")
    for r in result["metrics"]:
        print(f"  {r['app_name']}: 평점 {r['avg_rating']} / 부정 {r['negative_ratio']:.1%} "
              f"/ 리뷰 {r['review_count']} / 최근30일 {r['recent30d_rating_change']}")
