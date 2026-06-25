"""
VOC 이슈 유형 분류 + 심각도 부여 (issue_classifier.py)

⚠️ MVP 단계 주의:
    이 모듈은 머신러닝이 아니라 **키워드 사전 기반 룰(rule-based) 분류**다.
    유형 키워드는 config/issue_keywords.yaml 로 분리되어 있으며, 코드 수정 없이
    사전만 갱신해 오탐/누락을 보정할 수 있다.

유형(10종): 로그인, 인증, 속도, 오류, 혜택, 송금, 투자, UX, 보안, 기타
    - 한 리뷰가 여러 유형에 해당할 수 있으므로 list 로 반환한다.
    - 어떤 유형에도 매칭되지 않으면 ["기타"].

심각도(severity): high, medium, low, positive
    - positive: 긍정 리뷰(고별점 + 긍정 신호, 불만 유형 없음)
    - high    : 핵심 기능 사용 불가 / 거래 실패 / 보안 우려 / 오류 반복
    - medium  : 부정적이나 치명적이지 않은 불만
    - low     : 경미하거나 단순 문의성

실행:
    python -m src.issue_classifier
"""
from __future__ import annotations

import logging
import re

import pandas as pd
import yaml

from src.config import PATHS, settings

logging.basicConfig(
    level=getattr(logging, settings.log_level, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("issue_classifier")

# ----------------------------------------------------------------------
# 경로 / 상수
# ----------------------------------------------------------------------
KEYWORDS_PATH = PATHS.config_dir / "issue_keywords.yaml"
CLEAN_PATH = PATHS.processed_dir / "review_clean.csv"
REVIEW_SAMPLE_PATH = PATHS.outputs_dir / "issue_review_sample.csv"

ISSUE_OTHER = "기타"
SEVERITIES = ("high", "medium", "low", "positive")


def _normalize(text: str) -> str:
    """매칭용 정규화: 소문자화 + 공백 제거(붙여쓰기 변형 대응)."""
    if not text:
        return ""
    t = str(text).lower()
    t = re.sub(r"\s+", "", t)  # '로그인 이 안' → '로그인이안' 형태 매칭
    return t


# ----------------------------------------------------------------------
# 키워드 사전 로딩 (캐시)
# ----------------------------------------------------------------------
_KEYWORDS_CACHE: dict | None = None


def load_keywords(path=KEYWORDS_PATH, reload: bool = False) -> dict:
    """issue_keywords.yaml 을 로드한다(캐시)."""
    global _KEYWORDS_CACHE
    if _KEYWORDS_CACHE is not None and not reload:
        return _KEYWORDS_CACHE
    if not path.exists():
        raise FileNotFoundError(f"키워드 사전이 없습니다: {path}")
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    # 정규화된 키워드로 미리 변환해 매칭 비용을 줄인다.
    norm = {
        "issue_types": {
            t: [_normalize(k) for k in kws]
            for t, kws in data.get("issue_types", {}).items()
        },
        "high_severity_triggers": [_normalize(k) for k in data.get("high_severity_triggers", [])],
        "negative_signals": [_normalize(k) for k in data.get("negative_signals", [])],
        "positive_signals": [_normalize(k) for k in data.get("positive_signals", [])],
    }
    _KEYWORDS_CACHE = norm
    return norm


# ----------------------------------------------------------------------
# 1) 이슈 유형 분류 (룰 기반, 다중 라벨)
# ----------------------------------------------------------------------
def classify_issue_type(text: str) -> list[str]:
    """리뷰 텍스트의 이슈 유형을 list 로 반환한다 (룰 기반, 다중 라벨).

    어떤 유형에도 매칭되지 않으면 ["기타"].
    """
    kw = load_keywords()
    norm = _normalize(text)
    if not norm:
        return [ISSUE_OTHER]

    found = [
        issue_type
        for issue_type, keywords in kw["issue_types"].items()
        if any(k and k in norm for k in keywords)
    ]
    return found if found else [ISSUE_OTHER]


# ----------------------------------------------------------------------
# 2) 심각도 부여
# ----------------------------------------------------------------------
def classify_severity(
    text: str,
    rating: int | float | None,
    issue_types: list[str] | None = None,
) -> str:
    """리뷰의 답글 우선순위용 심각도를 반환한다: high|medium|low|positive.

    판정 로직(우선순위 순):
        1) high  : 고위험 트리거(로그인 불가/거래 실패/보안/오류 반복) 매칭,
                   또는 보안/오류 유형 + 저별점(≤2)
        2) positive: 고별점(≥4) + 긍정 신호 + 불만 유형 없음
        3) medium: 부정 신호 또는 저별점(≤2) 또는 불만성 유형 존재
        4) low   : 그 외(경미/문의성/중립)
    """
    kw = load_keywords()
    norm = _normalize(text)
    issue_types = issue_types if issue_types is not None else classify_issue_type(text)
    try:
        r = float(rating) if rating is not None else None
    except (ValueError, TypeError):
        r = None

    has_high_trigger = any(t and t in norm for t in kw["high_severity_triggers"])
    has_neg = any(s and s in norm for s in kw["negative_signals"])
    has_pos = any(s and s in norm for s in kw["positive_signals"])
    risky_types = {"보안", "오류", "로그인", "송금"}
    has_risky_type = bool(set(issue_types) & risky_types)

    # 1) high
    if has_high_trigger:
        return "high"
    if has_risky_type and r is not None and r <= 2:
        return "high"

    # 2) positive
    complaint_types = {"로그인", "인증", "속도", "오류", "보안", "송금"}
    has_complaint = bool(set(issue_types) & complaint_types)
    if r is not None and r >= 4 and has_pos and not has_complaint and not has_neg:
        return "positive"

    # 3) medium
    if has_neg or (r is not None and r <= 2) or has_complaint:
        return "medium"

    # 4) low
    return "low"


# ----------------------------------------------------------------------
# 3) 앱별/기간별 불만 유형 TOP N 집계
# ----------------------------------------------------------------------
def aggregate_top_issues(
    df: pd.DataFrame,
    top_n: int = 5,
    period_fmt: str = "%Y-%m",
) -> dict:
    """앱별/기간별 불만 유형 TOP N 을 집계한다.

    df 는 issue_types(list) 컬럼을 포함해야 한다.
    period_fmt 로 기간 단위를 정한다(기본 월: '%Y-%m', 일: '%Y-%m-%d').
    반환: {"by_app": {앱: [(유형, 건수), ...]}, "by_period": {기간: [...]}}
    """
    # 다중 라벨을 행 단위로 펼친다(기타 제외, 불만 유형만)
    exploded = df.explode("issue_types")
    exploded = exploded[exploded["issue_types"] != ISSUE_OTHER]

    by_app = {}
    for app, g in exploded.groupby("app_name"):
        by_app[app] = list(g["issue_types"].value_counts().head(top_n).items())

    by_period = {}
    if "date" in exploded.columns:
        dt = pd.to_datetime(exploded["date"], errors="coerce")
        # to_period 대신 strftime 으로 기간 문자열 생성(환경 의존 이슈 회피)
        period = dt.dt.strftime(period_fmt)
        exploded = exploded.assign(_period=period)
        for p, g in exploded.dropna(subset=["_period"]).groupby("_period"):
            by_period[p] = list(g["issue_types"].value_counts().head(top_n).items())

    return {"by_app": by_app, "by_period": by_period}


# ----------------------------------------------------------------------
# 4) 검수 샘플 생성
# ----------------------------------------------------------------------
def make_review_sample(df: pd.DataFrame, n: int = 100) -> pd.DataFrame:
    """사람이 검수할 수 있는 샘플을 생성한다.

    issue_types/severity 자동 분류 결과 + 검수자 입력용 공란 컬럼 포함.
    """
    n = min(n, len(df))
    sample = df.sample(n, random_state=42).copy()

    out = pd.DataFrame({
        "app_name": sample["app_name"].values,
        "store": sample["store"].values,
        "rating": sample["rating"].values,
        "clean_text": sample["clean_text"].values,
    })
    out["issue_types_auto"] = sample["issue_types"].apply(lambda x: ", ".join(x)).values
    out["severity_auto"] = sample["severity"].values
    # 검수자 입력용 공란
    out["issue_types_manual"] = ""
    out["severity_manual"] = ""
    out["reviewer_note"] = ""
    return out


def classify_dataframe(df: pd.DataFrame, text_col: str = "clean_text") -> pd.DataFrame:
    """DataFrame 전체에 유형/심각도 분류를 적용한다."""
    df = df.copy()
    df["issue_types"] = df[text_col].fillna("").apply(classify_issue_type)
    df["severity"] = df.apply(
        lambda row: classify_severity(row[text_col], row.get("rating"), row["issue_types"]),
        axis=1,
    )
    return df


def run(save: bool = True) -> pd.DataFrame:
    if not CLEAN_PATH.exists():
        raise FileNotFoundError(f"정제 데이터가 없습니다: {CLEAN_PATH}")
    df = pd.read_csv(CLEAN_PATH)
    df = classify_dataframe(df)

    # 분포 로그
    sev_dist = df["severity"].value_counts().to_dict()
    logger.info("심각도 분포: %s", sev_dist)
    top = aggregate_top_issues(df, top_n=5)
    logger.info("앱별 불만 유형 TOP5:")
    for app, items in top["by_app"].items():
        logger.info("  - %s: %s", app, items)

    if save:
        sample = make_review_sample(df, n=100)
        REVIEW_SAMPLE_PATH.parent.mkdir(parents=True, exist_ok=True)
        sample.to_csv(REVIEW_SAMPLE_PATH, index=False, encoding="utf-8-sig")
        logger.info("검수 샘플 저장: %s (%d건)", REVIEW_SAMPLE_PATH, len(sample))

    return df


if __name__ == "__main__":
    df = run()
    print("\n=== VOC 이슈 분류 (룰 기반 / ML 아님) ===")
    print("심각도 분포:", df["severity"].value_counts().to_dict())
    print("\n[유형별 건수]")
    exploded = df.explode("issue_types")
    print(exploded["issue_types"].value_counts().to_string())
