"""validate_review_data 의 정상/실패 케이스 테스트."""
from __future__ import annotations

import pandas as pd
import pytest

from src.validation import (
    REQUIRED_COLUMNS,
    OUR_APP,
    validate_review_data,
)


def _make_valid_df(n: int = 1200) -> pd.DataFrame:
    """모든 기준을 통과하는 합성 데이터.

    - 1,200건 (≥1,000)
    - 슈퍼SOL + 경쟁 앱 3개 (총 4개)
    - 결측/중복/날짜오류 없음, rating 1~5
    """
    apps = [OUR_APP, "경쟁A", "경쟁B", "경쟁C"]
    rows = []
    for i in range(n):
        rows.append({
            "app_name": apps[i % len(apps)],
            "store": "google_play" if i % 2 else "app_store",
            "rating": (i % 5) + 1,
            "date": f"2026-01-{(i % 28) + 1:02d}",
            # 고유 텍스트로 중복 방지
            "review_text": f"리뷰 본문 내용입니다 번호 {i}",
            "version": "1.0.0",
            "source_url": "https://example.com",
        })
    return pd.DataFrame(rows, columns=REQUIRED_COLUMNS)


# ----------------------------------------------------------------------
# 정상 케이스
# ----------------------------------------------------------------------
def test_valid_data_passes():
    df = _make_valid_df()
    result = validate_review_data(df)
    assert result["passed"] is True
    assert all(c["passed"] for c in result["checks"])
    assert result["metrics"]["total_reviews"] == len(df)
    assert result["metrics"]["app_count"] == 4
    assert result["metrics"]["data_quality_pass"] is True


def test_by_app_counts():
    df = _make_valid_df(1200)
    result = validate_review_data(df)
    # 4개 앱에 균등 분배 → 각 300건
    assert result["by_app"][OUR_APP] == 300
    assert sum(result["by_app"].values()) == 1200


# ----------------------------------------------------------------------
# 실패 케이스
# ----------------------------------------------------------------------
def _failed_check(result: dict, name: str) -> dict:
    return next(c for c in result["checks"] if c["name"] == name)


def test_missing_required_column_fails():
    df = _make_valid_df().drop(columns=["version"])
    result = validate_review_data(df)
    assert result["passed"] is False
    assert _failed_check(result, "필수 컬럼 100% 존재")["passed"] is False


def test_too_few_reviews_fails():
    df = _make_valid_df(500)  # < 1,000
    result = validate_review_data(df)
    assert result["passed"] is False
    assert _failed_check(result, "총 리뷰 수 ≥ 1,000건")["passed"] is False


def test_high_null_rate_fails():
    df = _make_valid_df()
    # 10% 결측 처리 (> 5%)
    df.loc[df.sample(frac=0.10, random_state=1).index, "review_text"] = ""
    result = validate_review_data(df)
    assert result["passed"] is False
    assert _failed_check(result, "리뷰 본문 결측률 ≤ 5%")["passed"] is False


def test_high_duplicate_rate_fails():
    df = _make_valid_df()
    # 10% 를 동일 리뷰로 덮어써 중복률 > 3%
    idx = df.sample(frac=0.10, random_state=2).index
    df.loc[idx, "app_name"] = OUR_APP
    df.loc[idx, "store"] = "google_play"
    df.loc[idx, "review_text"] = "완전히 동일한 중복 리뷰"
    result = validate_review_data(df)
    assert result["passed"] is False
    assert _failed_check(result, "중복률 ≤ 3%")["passed"] is False


def test_rating_out_of_range_fails():
    df = _make_valid_df()
    df.loc[df.index[:5], "rating"] = 7  # 범위 이탈
    result = validate_review_data(df)
    assert result["passed"] is False
    assert _failed_check(result, "rating 1~5 범위")["passed"] is False


def test_bad_dates_fail():
    df = _make_valid_df()
    # 10% 를 파싱 불가 문자열로 (성공률 < 95%)
    df.loc[df.sample(frac=0.10, random_state=3).index, "date"] = "날짜아님"
    result = validate_review_data(df)
    assert result["passed"] is False
    assert _failed_check(result, "날짜 파싱 성공률 ≥ 95%")["passed"] is False


def test_too_few_apps_fails():
    df = _make_valid_df()
    # 경쟁 앱을 모두 슈퍼SOL 로 바꿔 앱 종류 부족
    df["app_name"] = OUR_APP
    result = validate_review_data(df)
    assert result["passed"] is False
    assert _failed_check(result, "슈퍼SOL + 경쟁 앱 3개 이상")["passed"] is False


def test_missing_our_app_fails():
    df = _make_valid_df()
    df["app_name"] = df["app_name"].replace(OUR_APP, "경쟁D")
    result = validate_review_data(df)
    assert result["passed"] is False
    assert _failed_check(result, "슈퍼SOL + 경쟁 앱 3개 이상")["passed"] is False
