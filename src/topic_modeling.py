"""
한국어 토픽 모델링 (topic_modeling.py)

목적:
- clean_text 를 토큰화/불용어 제거 후 TF-IDF 로 벡터화하고,
  KMeans 로 토픽(군집)을 도출한다.
- 토픽별 대표 키워드 10개 + 대표 리뷰 5개를 추출해 사람이 검수할 수 있게 한다.

토큰화:
- kiwipiepy 기본, 실패 시 공백 기반 fallback (src/tokenizer.py 재사용).

⚠️ 토픽명(topic_label)은 자동 생성한 임시 라벨이며, 키워드만으로는 애매할 수 있어
   사람이 수정할 수 있도록 별도 컬럼으로 분리한다.

실행:
    python -m src.topic_modeling            # 기본 5토픽
    python -m src.topic_modeling --k 7      # 토픽 수 조정
"""
from __future__ import annotations

import argparse
import json
import logging

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

from src.config import PATHS, settings
from src.tokenizer import get_tokenizer

logging.basicConfig(
    level=getattr(logging, settings.log_level, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("topic_modeling")

# ----------------------------------------------------------------------
# 경로 / 상수
# ----------------------------------------------------------------------
CLEAN_PATH = PATHS.processed_dir / "review_clean.csv"
TOPIC_SUMMARY_PATH = PATHS.outputs_dir / "topic_summary.csv"
TOPIC_VALIDATION_PATH = PATHS.outputs_dir / "topic_validation_sample.md"
METRICS_PATH = PATHS.outputs_dir / "metrics.json"

DEFAULT_K = 5
RANDOM_STATE = 42
TOP_KEYWORDS = 10
TOP_REVIEWS = 5
# 짧고 변별력 낮은 리뷰(예: "좋아요")는 거대 군집 쏠림의 주원인이므로
# 토큰 수가 이 값 미만이면 토픽 모델링에서 제외한다(검수 샘플엔 영향 없음).
MIN_TOKENS = 4

# 토픽 모델링용 도메인 불용어 (금융앱 리뷰에 흔하지만 변별력 낮은 표현)
_EXTRA_STOPWORDS = {
    "은행", "앱", "어플", "사용", "이용", "그냥", "진짜", "정말", "너무", "조금",
    "하다", "되다", "있다", "없다", "같다", "보다", "좋다", "이거", "저거", "요즘",
}


# 모듈 레벨 토크나이저 (불용어 제거 포함)
_TOKENIZER = None


def tokenize(text: str) -> list[str]:
    """kiwi/whitespace 토큰화 + 도메인 불용어 제거."""
    global _TOKENIZER
    if _TOKENIZER is None:
        _TOKENIZER = get_tokenizer()
    return [t for t in _TOKENIZER.tokenize(text) if t not in _EXTRA_STOPWORDS]


def load_texts(path=CLEAN_PATH, min_tokens: int = MIN_TOKENS) -> pd.DataFrame:
    """정제 리뷰를 로드하고, 토큰 수가 min_tokens 미만인 짧은 리뷰를 제외한다.

    짧은 일반 리뷰("좋아요" 등)는 변별력이 낮아 KMeans 거대 군집의 원인이 되므로
    토픽 모델링 입력에서 제외한다(원본/검수 데이터는 그대로 유지).
    """
    if not path.exists():
        raise FileNotFoundError(f"정제 데이터가 없습니다: {path}")
    df = pd.read_csv(path)
    df["clean_text"] = df["clean_text"].fillna("").astype(str)
    df = df[df["clean_text"].str.strip() != ""].copy().reset_index(drop=True)

    n_before = len(df)
    df["n_tokens"] = df["clean_text"].apply(lambda t: len(tokenize(t)))
    df = df[df["n_tokens"] >= min_tokens].copy().reset_index(drop=True)
    logger.info("짧은 리뷰 필터(토큰<%d) 제외: %d → %d건", min_tokens, n_before, len(df))
    return df


def run_topic_modeling(df: pd.DataFrame, k: int = DEFAULT_K) -> dict:
    """TF-IDF + KMeans 로 토픽을 도출한다."""
    tok = get_tokenizer()
    logger.info("토크나이저 backend: %s", tok.backend)

    vectorizer = TfidfVectorizer(
        tokenizer=tokenize,
        token_pattern=None,
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.4,          # 너무 흔한 표현 제외 (쏠림 완화)
        sublinear_tf=True,   # 빈도 영향 완화 → 군집 균형 개선
    )
    X = vectorizer.fit_transform(df["clean_text"])
    terms = vectorizer.get_feature_names_out()
    logger.info("TF-IDF: %d docs x %d terms", X.shape[0], X.shape[1])

    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    labels = km.fit_predict(X)
    df = df.assign(topic_id=labels)

    # 토픽별 대표 키워드: 군집 중심에서 TF-IDF 가중치 상위 단어
    centroids = km.cluster_centers_
    topics = []
    for tid in range(k):
        top_idx = centroids[tid].argsort()[::-1][:TOP_KEYWORDS]
        keywords = [terms[i] for i in top_idx]
        members = df[df["topic_id"] == tid]
        # 대표 리뷰: 군집 내에서 중심과 가장 가까운(=대표성 높은) 리뷰
        rep_reviews = _representative_reviews(members, X, labels, tid, centroids)
        topics.append({
            "topic_id": tid,
            "size": int(len(members)),
            "keywords": keywords,
            "rep_reviews": rep_reviews,
            # 자동 임시 라벨(상위 키워드 3개) — 사람이 수정할 컬럼과 분리
            "auto_label": " / ".join(keywords[:3]),
        })
        logger.info("토픽 %d (n=%d): %s", tid, len(members), ", ".join(keywords[:5]))

    # 커버리지: 분석 대상(필터 통과) 리뷰는 모두 군집에 할당되므로 1.0.
    coverage = round(float((df["topic_id"] >= 0).mean()), 4)
    sizes = [t["size"] for t in topics]
    dominant_ratio = round(max(sizes) / len(df), 4) if sizes else 0.0
    logger.info("최대 토픽 비율: %.1f%% (균형 지표, 낮을수록 양호)", dominant_ratio * 100)
    return {"df": df, "topics": topics, "k": k, "coverage": coverage,
            "n_docs": int(X.shape[0]), "dominant_ratio": dominant_ratio}


def _representative_reviews(members, X, labels, tid, centroids) -> list[str]:
    """군집 중심과 가까운 대표 리뷰 TOP을 반환한다."""
    import numpy as np

    if members.empty:
        return []
    member_pos = members.index.to_numpy()
    sub = X[member_pos].toarray()           # dense 변환
    diff = sub - centroids[tid]             # (n_members, n_terms)
    dists = np.einsum("ij,ij->i", diff, diff)  # 중심과의 제곱거리(작을수록 대표성↑)
    order = dists.argsort()[:TOP_REVIEWS]
    return members.iloc[order]["clean_text"].tolist()


def save_outputs(result: dict) -> None:
    df, topics, k = result["df"], result["topics"], result["k"]

    # 1) topic_summary.csv (topic_label 은 사람이 채울 컬럼으로 분리)
    rows = []
    for t in topics:
        rows.append({
            "topic_id": t["topic_id"],
            "size": t["size"],
            "auto_label": t["auto_label"],
            "topic_label": "",  # ⚠️ 사람이 검수/수정할 최종 토픽명 (공란)
            "top_keywords": ", ".join(t["keywords"]),
            "rep_review_1": t["rep_reviews"][0] if len(t["rep_reviews"]) > 0 else "",
        })
    summary = pd.DataFrame(rows)
    TOPIC_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(TOPIC_SUMMARY_PATH, index=False, encoding="utf-8-sig")
    logger.info("토픽 요약 저장: %s", TOPIC_SUMMARY_PATH)

    # 2) topic_validation_sample.md (검수용: 키워드 10 + 대표리뷰 5)
    lines = [
        "# 토픽 품질 검증 샘플",
        "",
        f"- 토픽 수: **{k}**",
        f"- 분석 문서 수: {result['n_docs']} (토큰<{MIN_TOKENS} 짧은 리뷰 제외 후)",
        f"- 토픽 커버리지: {result['coverage']:.2%}",
        f"- 최대 토픽 비율: {result.get('dominant_ratio', 0):.2%} (낮을수록 균형 양호)",
        "",
        "> ⚠️ auto_label 은 상위 키워드로 만든 **자동 임시 라벨**입니다. "
        "키워드만으로 토픽 의미가 애매할 수 있으니, `topic_summary.csv` 의 "
        "`topic_label` 컬럼에 검수자가 최종 토픽명을 입력하세요.",
        "",
    ]
    for t in topics:
        lines += [
            f"## 토픽 {t['topic_id']} (n={t['size']}) — auto: {t['auto_label']}",
            "",
            f"**대표 키워드 10**: {', '.join(t['keywords'])}",
            "",
            "**대표 리뷰 5**:",
        ]
        for i, rv in enumerate(t["rep_reviews"], 1):
            lines.append(f"{i}. {rv}")
        lines += ["", "**검수자 토픽명**: ____________________", "", "---", ""]
    TOPIC_VALIDATION_PATH.write_text("\n".join(lines), encoding="utf-8")
    logger.info("토픽 검증 샘플 저장: %s", TOPIC_VALIDATION_PATH)

    # 3) metrics.json 병합
    existing = {}
    if METRICS_PATH.exists():
        try:
            with open(METRICS_PATH, encoding="utf-8") as f:
                existing = json.load(f)
        except Exception as exc:
            logger.warning("metrics.json 로드 실패 (%r).", exc)
    existing["topic_modeling"] = {
        "topic_count": k,
        "topic_review_coverage": result["coverage"],
        "n_docs": result["n_docs"],
        "dominant_topic_ratio": result.get("dominant_ratio"),
        "topic_sizes": {str(t["topic_id"]): t["size"] for t in topics},
    }
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)
    logger.info("metrics.json 병합 저장: %s", METRICS_PATH)


def run(k: int = DEFAULT_K, save: bool = True) -> dict:
    df = load_texts()
    logger.info("분석 문서: %d건", len(df))
    result = run_topic_modeling(df, k=k)
    if save:
        save_outputs(result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="한국어 토픽 모델링 (TF-IDF + KMeans)")
    parser.add_argument("--k", type=int, default=DEFAULT_K, help="토픽 수 (기본 5)")
    args = parser.parse_args()

    from src.topic_modeling import run as _run  # pickle/모듈경로 안전
    out = _run(k=args.k)
    print(f"\n=== 토픽 모델링 완료 (k={out['k']}) ===")
    for t in out["topics"]:
        print(f"  토픽 {t['topic_id']} (n={t['size']}): {', '.join(t['keywords'][:6])}")
    print(f"커버리지: {out['coverage']:.2%}")
