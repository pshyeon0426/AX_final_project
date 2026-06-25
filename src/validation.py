"""
데이터 품질 검증 모듈 (validation.py)

목적:
- 정제된 리뷰 데이터가 분석/모델링에 쓰기 적합한지 자동 검증한다.
- pass/fail 과 상세 사유를 구조화해 반환하고, 리포트(md)/지표(json)로 남긴다.

검증 기준:
    1) 필수 컬럼 100% 존재
    2) 리뷰 본문 결측률 5% 이하
    3) 날짜 파싱 성공률 95% 이상
    4) 중복률 3% 이하
    5) rating 1~5 범위
    6) 총 리뷰 수 1,000건 이상
    7) 슈퍼SOL + 경쟁 앱 3개 이상 포함 (총 4개 이상 앱)

실행:
    python -m src.validation
"""
from __future__ import annotations

import json
import logging

import pandas as pd

from src.config import PATHS, settings

logging.basicConfig(
    level=getattr(logging, settings.log_level, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("validation")

# ----------------------------------------------------------------------
# 경로 / 기준값
# ----------------------------------------------------------------------
CLEAN_PATH = PATHS.processed_dir / "review_clean.csv"
RAW_PATH = PATHS.raw_dir / "review_raw.csv"
REPORT_PATH = PATHS.outputs_dir / "data_validation_report.md"
METRICS_PATH = PATHS.outputs_dir / "metrics.json"

REQUIRED_COLUMNS = ["app_name", "store", "rating", "date", "review_text", "version", "source_url"]

OUR_APP = "슈퍼SOL"

# 임계값
MAX_NULL_RATE = 0.05       # 결측률 ≤ 5%
MIN_DATE_PARSE_RATE = 0.95  # 날짜 파싱 성공률 ≥ 95%
MAX_DUP_RATE = 0.03        # 중복률 ≤ 3%
MIN_TOTAL_REVIEWS = 1000   # 총 ≥ 1,000건
MIN_COMPETITORS = 3        # 경쟁 앱 ≥ 3개 (우리 앱 제외)
RATING_MIN, RATING_MAX = 1, 5


def _check(name: str, passed: bool, detail: str) -> dict:
    return {"name": name, "passed": bool(passed), "detail": detail}


def validate_review_data(df: pd.DataFrame) -> dict:
    """리뷰 DataFrame 의 품질을 검증한다.

    Returns:
        {
          "passed": bool,                 # 전체 통과 여부
          "checks": [ {name, passed, detail}, ... ],
          "metrics": { total_reviews, app_count, ... },
          "by_app": { 앱명: 건수, ... },
        }
    """
    checks: list[dict] = []
    n = len(df)

    # 1) 필수 컬럼 존재
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    checks.append(_check(
        "필수 컬럼 100% 존재",
        not missing,
        "모든 필수 컬럼 존재" if not missing else f"누락 컬럼: {missing}",
    ))
    has_cols = not missing

    # 2) 리뷰 본문 결측률 ≤ 5%
    if has_cols and n:
        null_text = int(df["review_text"].isna().sum() +
                        (df["review_text"].fillna("").astype(str).str.strip() == "").sum())
        null_rate = null_text / n
        checks.append(_check(
            "리뷰 본문 결측률 ≤ 5%",
            null_rate <= MAX_NULL_RATE,
            f"결측률 {null_rate:.2%} (결측 {null_text}건 / {n}건)",
        ))
    else:
        null_rate = 1.0
        checks.append(_check("리뷰 본문 결측률 ≤ 5%", False, "컬럼 없음 또는 데이터 없음"))

    # 3) 날짜 파싱 성공률 ≥ 95%
    if has_cols and n:
        parsed = pd.to_datetime(df["date"], errors="coerce")
        ok = int(parsed.notna().sum())
        parse_rate = ok / n
        checks.append(_check(
            "날짜 파싱 성공률 ≥ 95%",
            parse_rate >= MIN_DATE_PARSE_RATE,
            f"성공률 {parse_rate:.2%} (성공 {ok}건 / {n}건)",
        ))
    else:
        parse_rate = 0.0
        checks.append(_check("날짜 파싱 성공률 ≥ 95%", False, "컬럼 없음 또는 데이터 없음"))

    # 4) 중복률 ≤ 3%
    if has_cols and n:
        dup = int(df.duplicated(subset=["app_name", "store", "review_text"]).sum())
        dup_rate = dup / n
        checks.append(_check(
            "중복률 ≤ 3%",
            dup_rate <= MAX_DUP_RATE,
            f"중복률 {dup_rate:.2%} (중복 {dup}건 / {n}건)",
        ))
    else:
        dup_rate = 1.0
        checks.append(_check("중복률 ≤ 3%", False, "컬럼 없음 또는 데이터 없음"))

    # 5) rating 1~5 범위
    if has_cols and n:
        ratings = pd.to_numeric(df["rating"], errors="coerce")
        out_of_range = int(((ratings < RATING_MIN) | (ratings > RATING_MAX) | ratings.isna()).sum())
        checks.append(_check(
            "rating 1~5 범위",
            out_of_range == 0,
            "모든 rating 정상" if out_of_range == 0 else f"범위 이탈/결측 {out_of_range}건",
        ))
    else:
        checks.append(_check("rating 1~5 범위", False, "컬럼 없음 또는 데이터 없음"))

    # 6) 총 리뷰 수 ≥ 1,000
    checks.append(_check(
        "총 리뷰 수 ≥ 1,000건",
        n >= MIN_TOTAL_REVIEWS,
        f"총 {n}건",
    ))

    # 7) 슈퍼SOL + 경쟁 앱 3개 이상
    by_app = {}
    if has_cols and n:
        by_app = df["app_name"].value_counts().to_dict()
        apps = set(by_app.keys())
        has_our = OUR_APP in apps
        n_competitors = len(apps - {OUR_APP})
        ok_apps = has_our and n_competitors >= MIN_COMPETITORS
        detail = (f"슈퍼SOL 포함={has_our}, 경쟁 앱 {n_competitors}개 "
                  f"(전체 {len(apps)}개)")
        checks.append(_check("슈퍼SOL + 경쟁 앱 3개 이상", ok_apps, detail))
    else:
        checks.append(_check("슈퍼SOL + 경쟁 앱 3개 이상", False, "컬럼 없음 또는 데이터 없음"))

    passed = all(c["passed"] for c in checks)

    metrics = {
        "total_reviews": int(n),
        "app_count": int(len(by_app)),
        "null_text_rate": round(null_rate, 4),
        "date_parse_rate": round(parse_rate, 4),
        "duplicate_rate": round(dup_rate, 4),
        "data_quality_pass": bool(passed),
    }

    return {"passed": passed, "checks": checks, "metrics": metrics, "by_app": by_app}


def render_report(result: dict) -> str:
    """검증 결과를 Markdown 리포트 문자열로 만든다."""
    status = "✅ PASS" if result["passed"] else "❌ FAIL"
    lines = [
        "# 데이터 품질 검증 리포트",
        "",
        f"**종합 결과: {status}**",
        "",
        "## 검증 항목",
        "",
        "| 항목 | 결과 | 상세 |",
        "|------|------|------|",
    ]
    for c in result["checks"]:
        mark = "✅" if c["passed"] else "❌"
        lines.append(f"| {c['name']} | {mark} | {c['detail']} |")

    lines += ["", "## app_name 별 리뷰 수", "", "| 앱 | 리뷰 수 |", "|----|--------|"]
    for app, cnt in sorted(result["by_app"].items(), key=lambda x: -x[1]):
        lines.append(f"| {app} | {cnt} |")

    lines += [
        "",
        "## 주요 지표",
        "",
        f"- 총 리뷰 수: {result['metrics']['total_reviews']}",
        f"- 앱 수: {result['metrics']['app_count']}",
        f"- 결측률: {result['metrics']['null_text_rate']:.2%}",
        f"- 날짜 파싱 성공률: {result['metrics']['date_parse_rate']:.2%}",
        f"- 중복률: {result['metrics']['duplicate_rate']:.2%}",
        f"- 품질 통과 여부: {result['metrics']['data_quality_pass']}",
        "",
    ]
    return "\n".join(lines)


def _merge_metrics(new_metrics: dict, path=METRICS_PATH) -> dict:
    """기존 metrics.json 에 검증 지표를 병합 저장한다."""
    existing = {}
    if path.exists():
        try:
            with open(path, encoding="utf-8") as f:
                existing = json.load(f)
        except Exception as exc:
            logger.warning("기존 metrics.json 로드 실패 (%r). 새로 생성.", exc)
    existing.update(new_metrics)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)
    return existing


def run(df: pd.DataFrame | None = None, save: bool = True) -> dict:
    """검증을 실행하고 리포트/지표를 저장한다."""
    if df is None:
        src = CLEAN_PATH if CLEAN_PATH.exists() else RAW_PATH
        if not src.exists():
            raise FileNotFoundError(
                f"검증할 데이터가 없습니다: {CLEAN_PATH} / {RAW_PATH}"
            )
        logger.info("검증 데이터 로드: %s", src)
        df = pd.read_csv(src)

    result = validate_review_data(df)

    # app_name별 리뷰 수 출력
    logger.info("app_name 별 리뷰 수:")
    for app, cnt in sorted(result["by_app"].items(), key=lambda x: -x[1]):
        logger.info("  - %s: %d", app, cnt)

    if save:
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text(render_report(result), encoding="utf-8")
        logger.info("검증 리포트 저장: %s", REPORT_PATH)

        merged = _merge_metrics({
            "total_reviews": result["metrics"]["total_reviews"],
            "app_count": result["metrics"]["app_count"],
            "data_quality_pass": result["metrics"]["data_quality_pass"],
        })
        logger.info("metrics.json 병합 저장: %s", METRICS_PATH)
        logger.debug("metrics: %s", merged)

    return result


if __name__ == "__main__":
    res = run()
    print("\n=== 데이터 품질 검증 ===")
    print("종합:", "PASS" if res["passed"] else "FAIL")
    for c in res["checks"]:
        print(f"  [{'PASS' if c['passed'] else 'FAIL'}] {c['name']} - {c['detail']}")
