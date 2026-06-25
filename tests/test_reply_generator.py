"""답글 생성 모듈 검증 테스트 (7개 핵심 케이스 + 안전성).

pytest 는 OPENAI 키 유무와 무관하게 통과해야 한다.
- 키가 없으면 fallback 경로를, 있으면 LLM 경로를 타되 반환 스키마/안전성은 동일하게 보장.
"""
from __future__ import annotations

import pytest

from src.reply_generator import (
    generate_reply,
    rule_based_reply,
    safety_check_reply,
    call_openai_reply,
)

REQUIRED_KEYS = {"reply_draft", "reply_type", "safety_flags", "needs_human_review", "reason"}


def _check_schema(result: dict):
    assert REQUIRED_KEYS <= set(result.keys())
    assert isinstance(result["reply_draft"], str) and result["reply_draft"].strip()
    assert isinstance(result["safety_flags"], list)
    assert isinstance(result["needs_human_review"], bool)


# ----------------------------------------------------------------------
# 케이스 1: 긍정 리뷰 → 감사 답글
# ----------------------------------------------------------------------
def test_case1_positive_thanks():
    r = generate_reply("정말 편리하고 만족스러워요 잘 쓰고 있습니다",
                       rating=5, sentiment="positive", issue_types=[], severity="positive")
    _check_schema(r)
    assert r["reply_type"] == "thanks_positive"
    assert r["safety_flags"] == []


# ----------------------------------------------------------------------
# 케이스 2: 로그인 오류 → 사과 + 고객센터/앱 내 문의 안내
# ----------------------------------------------------------------------
def test_case2_login_error_apology_guidance():
    r = generate_reply("로그인이 안돼서 너무 불편해요",
                       rating=2, sentiment="negative",
                       issue_types=["로그인"], severity="high")
    _check_schema(r)
    # high → 고객센터/문의 안내 문구 포함
    assert "고객센터" in r["reply_draft"] or "문의" in r["reply_draft"]


# ----------------------------------------------------------------------
# 케이스 3: 보안 우려 → 민감정보 요청 금지 + human review
# ----------------------------------------------------------------------
def test_case3_security_human_review():
    r = generate_reply("계정이 해킹된 것 같고 개인정보 유출이 의심돼요",
                       rating=1, sentiment="negative",
                       issue_types=["보안"], severity="high")
    _check_schema(r)
    assert r["needs_human_review"] is True
    # 답글이 민감정보 입력을 요청하지 않아야 한다
    assert safety_check_reply(r["reply_draft"])["is_safe"] or \
        "personal_info_request" not in r["safety_flags"]


# ----------------------------------------------------------------------
# 케이스 4: 개인정보 포함(MASKED) 리뷰 → human review
# ----------------------------------------------------------------------
def test_case4_masked_personal_info_human_review():
    # 전처리에서 PII 는 [MASKED] 로 치환됨 → 송금/계좌 관련 민감
    r = generate_reply("제 계좌 [MASKED]에서 출금이 안돼요 돈이 안 빠져요",
                       rating=1, sentiment="negative",
                       issue_types=["송금"], severity="high")
    _check_schema(r)
    assert r["needs_human_review"] is True


# ----------------------------------------------------------------------
# 케이스 5: 기능 개선 요청 → 의견 감사 + 개선 참고
# ----------------------------------------------------------------------
def test_case5_feature_request_ack():
    r = rule_based_reply("이체 메모 기능을 추가해주세요 건의합니다",
                         rating=3, sentiment="neutral",
                         issue_types=["송금"], severity="low",
                         app_name="슈퍼SOL")
    _check_schema(r)
    # 룰 기반에서 감사/검토 표현 확인
    assert any(w in r["reply_draft"] for w in ["감사", "참고", "검토", "의견"])


# ----------------------------------------------------------------------
# 케이스 6: OpenAI 키 누락 → fallback 정상 생성
# ----------------------------------------------------------------------
def test_case6_no_api_key_fallback(monkeypatch):
    import src.reply_generator as rg
    # openai_enabled 를 False 로 강제
    monkeypatch.setattr(rg.settings.__class__, "openai_enabled", property(lambda self: False))
    r = generate_reply("앱이 느려요", rating=2, sentiment="negative",
                       issue_types=["속도"], severity="medium")
    _check_schema(r)
    assert r["reason"].startswith("[fallback]")


# ----------------------------------------------------------------------
# 케이스 7: API 오류/JSON 파싱 실패 → 앱 중단 없이 fallback
# ----------------------------------------------------------------------
def test_case7_api_error_fallback(monkeypatch):
    import src.reply_generator as rg
    # openai_enabled True 로 보이게 하고, call_openai_reply 가 예외를 던지게 함
    monkeypatch.setattr(rg.settings.__class__, "openai_enabled", property(lambda self: True))

    def _boom(prompt, model):
        raise RuntimeError("API timeout (simulated)")

    monkeypatch.setattr(rg, "call_openai_reply", _boom)
    # call_openai_reply 가 패치되어 예외를 던지므로 model 값은 무관하다.

    r = generate_reply("결제 오류가 나요", rating=1, sentiment="negative",
                       issue_types=["오류"], severity="high")
    _check_schema(r)  # 중단 없이 dict 반환
    assert "LLM 오류" in r["reason"]


def test_case7b_json_parse_failure(monkeypatch):
    import src.reply_generator as rg
    monkeypatch.setattr(rg.settings.__class__, "openai_enabled", property(lambda self: True))

    def _bad_json(prompt, model):
        return "이건 JSON 이 아닙니다"  # 파싱 실패 유도

    monkeypatch.setattr(rg, "call_openai_reply", _bad_json)
    r = generate_reply("화면이 안 떠요", rating=2, sentiment="negative",
                       issue_types=["오류"], severity="medium")
    _check_schema(r)
    assert r["reason"].startswith("[fallback]")


# ----------------------------------------------------------------------
# safety_check_reply 단위 테스트
# ----------------------------------------------------------------------
@pytest.mark.parametrize("text,flag", [
    ("확인을 위해 비밀번호를 입력해 주세요", "personal_info_request"),
    ("반드시 보상해 드리겠습니다", "definitive_promise"),
    ("이 펀드 추천드립니다 수익 보장", "investment_advice"),
])
def test_safety_detects_violations(text, flag):
    r = safety_check_reply(text)
    assert r["is_safe"] is False
    assert flag in r["safety_flags"]


def test_safety_passes_clean_reply():
    r = safety_check_reply("불편을 드려 죄송합니다. 앱 내 고객센터로 문의 부탁드립니다.")
    assert r["is_safe"] is True


def test_safety_warning_not_false_positive():
    """비밀번호 공유 금지 '경고'는 안전해야 한다(요청이 아님)."""
    r = safety_check_reply("비밀번호·인증번호는 어떠한 경우에도 공유하지 마세요")
    assert r["is_safe"] is True
