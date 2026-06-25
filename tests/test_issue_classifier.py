"""issue_classifier 의 룰 기반 유형/심각도 분류 테스트 (대표 문장)."""
from __future__ import annotations

import pytest

from src.issue_classifier import (
    ISSUE_OTHER,
    SEVERITIES,
    classify_issue_type,
    classify_severity,
)


# ----------------------------------------------------------------------
# 이슈 유형 분류
# ----------------------------------------------------------------------
@pytest.mark.parametrize("text,expected", [
    ("로그인이 자꾸 안돼요", "로그인"),
    ("본인 인증 단계에서 계속 실패합니다", "인증"),
    ("앱이 너무 느리고 버벅거려요", "속도"),
    ("결제하다가 자꾸 강제종료됩니다", "오류"),
    ("포인트 적립 혜택이 좋아요", "혜택"),
    ("송금 한도 설정이 헷갈려요", "송금"),
    ("펀드 가입하고 수익률 보기 좋아요", "투자"),
    ("메뉴가 복잡하고 디자인이 불편해요", "UX"),
    ("개인정보 유출된 것 같아 불안해요", "보안"),
])
def test_single_issue_type(text, expected):
    result = classify_issue_type(text)
    assert expected in result


def test_multi_label():
    """한 리뷰가 여러 유형에 매칭되면 list 로 모두 반환한다."""
    text = "로그인도 안되고 송금도 오류나고 너무 느려요"
    result = classify_issue_type(text)
    assert isinstance(result, list)
    # 로그인 / 송금 / 오류 / 속도 가 모두 잡혀야 한다
    for t in ["로그인", "송금", "오류", "속도"]:
        assert t in result
    assert len(result) >= 3


def test_returns_list_type():
    assert isinstance(classify_issue_type("아무 내용"), list)


def test_other_when_no_match():
    result = classify_issue_type("날씨가 좋네요 산책하기 딱이에요")
    assert result == [ISSUE_OTHER]


def test_empty_text():
    assert classify_issue_type("") == [ISSUE_OTHER]
    assert classify_issue_type(None) == [ISSUE_OTHER]


# ----------------------------------------------------------------------
# 심각도 분류
# ----------------------------------------------------------------------
def test_severity_values_valid():
    s = classify_severity("로그인 안됨", 1, ["로그인"])
    assert s in SEVERITIES


def test_high_login_failure():
    """로그인 불가 → high"""
    s = classify_severity("로그인이 안돼서 앱을 못 써요", 1)
    assert s == "high"


def test_high_transfer_failure():
    """이체/결제 실패 → high"""
    assert classify_severity("이체실패 떴는데 돈이 빠져나갔어요", 1) == "high"
    assert classify_severity("결제했는데 결제실패 뜨고 두 번 빠짐", 1) == "high"


def test_high_security_concern():
    """보안 우려 → high"""
    s = classify_severity("계정이 해킹된 것 같아요 명의도용 의심", 1)
    assert s == "high"


def test_high_repeated_error():
    """오류 반복 → high"""
    s = classify_severity("계속오류 나고 자꾸 튕겨요", 2)
    assert s == "high"


def test_positive_review():
    """고별점 + 긍정 신호 + 불만 없음 → positive"""
    s = classify_severity("전반적으로 편리하고 만족스러워요 추천합니다", 5)
    assert s == "positive"


def test_medium_complaint():
    """치명적이지 않은 불만 → medium"""
    s = classify_severity("메뉴가 좀 복잡하고 불편하네요", 2, ["UX"])
    assert s == "medium"


def test_low_neutral_inquiry():
    """경미/문의성/중립 → low (high/positive/medium 아님)"""
    s = classify_severity("수수료 면제 조건이 궁금합니다", 3, [ISSUE_OTHER])
    assert s == "low"


def test_severity_consistency_with_issue_types():
    """issue_types 를 명시로 넘겨도 동일 로직으로 동작한다."""
    text = "로그인 안됨"
    types = classify_issue_type(text)
    s = classify_severity(text, 1, types)
    assert s == "high"
