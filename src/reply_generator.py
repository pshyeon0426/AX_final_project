"""
리뷰 답글 초안 생성 (reply_generator.py)

목적:
- 리뷰별 고객 답글 '초안'을 생성한다. OpenAI API 가 가능하면 LLM 으로,
  불가하면 rule_based_reply() 로 fallback 한다.
- config/reply_policy.yaml 정책과 prompts/reply_prompt.md 프롬프트를 사용한다.

⚠️ 원칙:
- 생성 답글은 **자동 게시가 아니라 담당자 검수용 초안**이다.
- 실제 OPENAI_API_KEY 는 .env 에서만 주입하며 코드에 하드코딩하지 않는다.
- 모델명은 OPENAI_MODEL 환경변수로 교체 가능하다.

반환 dict 스키마(항상 동일):
    {reply_draft, reply_type, safety_flags, needs_human_review, reason}

실행:
    python -m src.reply_generator
"""
from __future__ import annotations

import ast
import json
import logging
import re
import time
from datetime import datetime, timezone

import pandas as pd
import yaml

from src.config import PATHS, settings

logging.basicConfig(
    level=getattr(logging, settings.log_level, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("reply_generator")

# ----------------------------------------------------------------------
# 경로 / 상수
# ----------------------------------------------------------------------
POLICY_PATH = PATHS.config_dir / "reply_policy.yaml"
PROMPT_PATH = PATHS.prompts_dir / "reply_prompt.md"
CLEAN_PATH = PATHS.processed_dir / "review_clean.csv"
DRAFTS_PATH = PATHS.outputs_dir / "reply_drafts.csv"
PERF_LOG_PATH = PATHS.outputs_dir / "performance_log.json"

OPENAI_TIMEOUT = 30  # 초

_POLICY_CACHE: dict | None = None


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_policy(path=POLICY_PATH, reload: bool = False) -> dict:
    """reply_policy.yaml 로드(캐시)."""
    global _POLICY_CACHE
    if _POLICY_CACHE is not None and not reload:
        return _POLICY_CACHE
    if not path.exists():
        raise FileNotFoundError(f"답글 정책 파일이 없습니다: {path}")
    with open(path, encoding="utf-8") as f:
        _POLICY_CACHE = yaml.safe_load(f)
    return _POLICY_CACHE


# ----------------------------------------------------------------------
# severity → reply_type 매핑
# ----------------------------------------------------------------------
def _pick_reply_type(severity: str, sentiment: str, policy: dict) -> str:
    mapping = policy.get("severity_to_reply_type", {})
    if severity in mapping:
        return mapping[severity]
    if sentiment == "positive":
        return "thanks_positive"
    return "fallback_general"


def _force_review(review_text, severity, issue_types, policy) -> tuple[bool, list[str]]:
    """정책상 검수 강제 조건에 걸리는지 판정한다."""
    fr = policy.get("force_human_review", {})
    flags = []
    norm = re.sub(r"\s+", "", str(review_text).lower())

    if severity in fr.get("severities", []):
        flags.append(f"severity={severity}")
    hit_types = set(issue_types or []) & set(fr.get("issue_types", []))
    if hit_types:
        flags.append(f"issue_types={sorted(hit_types)}")
    hit_kw = [k for k in fr.get("keywords", []) if re.sub(r"\s+", "", k.lower()) in norm]
    if hit_kw:
        flags.append(f"keywords={hit_kw}")

    return (len(flags) > 0), flags


# ----------------------------------------------------------------------
# 안전성 점검
# ----------------------------------------------------------------------
def safety_check_reply(reply_text: str) -> dict:
    """답글에서 민감정보 요청/확정 약속/금융 조언 표현을 탐지한다.

    Returns:
        {"safety_flags": [...], "is_safe": bool}
    """
    text = str(reply_text or "")
    norm = re.sub(r"\s+", "", text.lower())
    flags = []

    policy = load_policy()
    # 1) 정책 금칙어
    for phrase in policy.get("banned_phrases", []):
        if re.sub(r"\s+", "", phrase.lower()) in norm:
            flags.append(f"banned_phrase:{phrase}")

    # 2) 민감정보 입력 요청 패턴
    if re.search(r"(비밀번호|인증번호|계좌번호|주민(등록)?번호|카드번호)", text) and \
       re.search(r"(입력|알려|보내|제공|남겨)", text):
        flags.append("personal_info_request")

    # 3) 확정 약속 패턴
    if re.search(r"(반드시|무조건|100%|꼭).{0,10}(해결|보상|처리|환불)", text) or \
       re.search(r"(보상|환불)(해\s*드리|하겠|해드림)", text):
        flags.append("definitive_promise")

    # 4) 금융상품 투자 조언/추천
    if re.search(r"(투자|펀드|주식|상품).{0,10}(추천|권유|하세요|드립니다|좋습니다)", text) or \
       re.search(r"(수익|수익률).{0,6}(보장|확실)", text):
        flags.append("investment_advice")

    return {"safety_flags": flags, "is_safe": len(flags) == 0}


# ----------------------------------------------------------------------
# 프롬프트 빌드
# ----------------------------------------------------------------------
def build_reply_prompt(review_text, sentiment, issue_types, severity,
                       app_name, reply_type, policy, extra_guidance: str = "") -> str:
    template = PROMPT_PATH.read_text(encoding="utf-8")
    tone = policy.get("tone", {})
    tone_str = (f"{tone.get('style','공손한 존댓말')}, "
                f"{tone.get('sentences_min',2)}~{tone.get('sentences_max',4)}문장")
    contact = policy.get("contact_guidance", {})
    contact_str = (contact.get("security") if reply_type == "security_sensitive"
                   else contact.get("default", ""))
    return template.format(
        app_name=app_name,
        review=review_text,
        sentiment=sentiment,
        issue_types=", ".join(issue_types or []),
        severity=severity,
        reply_type=reply_type,
        tone=tone_str,
        banned_phrases=", ".join(policy.get("banned_phrases", [])),
        contact_guidance=contact_str,
        extra_guidance=(extra_guidance.strip() or "(없음)"),
    )


# ----------------------------------------------------------------------
# OpenAI 호출 (분리)
# ----------------------------------------------------------------------
def call_openai_reply(prompt: str, model: str) -> str:
    """OpenAI Chat API 를 호출해 원문 응답(text)을 반환한다.

    실패/타임아웃 시 예외를 그대로 올린다(상위에서 fallback 처리).
    """
    from openai import OpenAI  # 지연 import

    client = OpenAI(api_key=settings.require("openai_api_key"), timeout=OPENAI_TIMEOUT)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content":
             "너는 금융 앱 고객지원팀을 돕는 AI다. 반드시 지정된 JSON 형식만 출력한다. "
             "개인정보 요구, 확정 약속, 투자 조언, 경쟁사 비방을 하지 않는다."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.4,
        response_format={"type": "json_object"},
    )
    return resp.choices[0].message.content


def _parse_llm_json(raw: str) -> dict:
    """LLM 응답 텍스트에서 JSON 객체를 파싱한다."""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # 코드블록/잡텍스트 섞인 경우 중괄호 구간만 추출 시도
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                return ast.literal_eval(m.group(0))
        raise


# ----------------------------------------------------------------------
# 룰 기반 fallback 답글
# ----------------------------------------------------------------------
_RULE_TEMPLATES = {
    "thanks_positive":
        "소중한 의견 남겨주셔서 감사합니다. 더 편리한 {app} 가 되도록 꾸준히 개선하겠습니다.",
    "apology_and_guidance":
        "이용에 불편을 드려 죄송합니다. 말씀해 주신 내용을 면밀히 확인하겠으며, "
        "추가 확인이 필요하시면 앱 내 [고객센터]로 문의 부탁드립니다.",
    "feature_request_ack":
        "제안해 주셔서 감사합니다. 주신 의견은 서비스 개선 검토 시 참고하겠습니다.",
    "how_to_guidance":
        "문의 주셔서 감사합니다. 자세한 이용 방법은 앱 내 [고객센터] 안내를 통해 확인하실 수 있습니다.",
    "security_sensitive":
        "불편과 우려를 드려 죄송합니다. 보안 관련 사항은 앱 내 [고객센터]를 통해 "
        "확인 도와드리겠습니다. 비밀번호·인증번호 등은 어떠한 경우에도 공유하지 마세요.",
    "fallback_general":
        "소중한 의견 감사합니다. 더 나은 서비스를 제공하도록 노력하겠으며, "
        "필요 시 앱 내 [고객센터]로 문의 부탁드립니다.",
}


def rule_based_reply(review_text, rating, sentiment, issue_types, severity, app_name) -> dict:
    """LLM 없이 정책 템플릿으로 답글 초안을 생성한다."""
    policy = load_policy()
    reply_type = _pick_reply_type(severity, sentiment, policy)
    draft = _RULE_TEMPLATES.get(reply_type, _RULE_TEMPLATES["fallback_general"]).format(app=app_name)

    forced, reasons = _force_review(review_text, severity, issue_types, policy)
    type_needs = policy.get("reply_types", {}).get(reply_type, {}).get("needs_human_review", False)
    safety = safety_check_reply(draft)
    needs_review = forced or type_needs or (not safety["is_safe"])

    reason_parts = []
    if forced:
        reason_parts.append("정책상 검수 강제: " + "; ".join(reasons))
    if type_needs:
        reason_parts.append(f"유형 {reply_type} 기본 검수")
    if not safety["is_safe"]:
        reason_parts.append("안전성 플래그: " + ", ".join(safety["safety_flags"]))
    reason = " / ".join(reason_parts) or "룰 기반 생성, 특이사항 없음"

    return {
        "reply_draft": draft,
        "reply_type": reply_type,
        "safety_flags": safety["safety_flags"],
        "needs_human_review": bool(needs_review),
        "reason": f"[fallback] {reason}",
    }


# ----------------------------------------------------------------------
# 단건 생성
# ----------------------------------------------------------------------
def generate_reply(review_text, rating=None, sentiment="neutral",
                   issue_types=None, severity="medium", app_name="슈퍼SOL",
                   extra_guidance: str = "", _stats: dict | None = None) -> dict:
    """리뷰 1건에 대한 답글 초안 dict 를 반환한다(항상 동일 스키마).

    extra_guidance: 운영자가 입력한 'LLM 에 항상 반영할 지침'. 금지 원칙과 충돌 시 금지 원칙 우선.
    """
    policy = load_policy()
    issue_types = issue_types or []

    # OPENAI 미설정 → 즉시 fallback (LLM 호출 시도 안 함)
    if not settings.openai_enabled:
        if _stats is not None:
            _stats["fallback"] += 1
        result = rule_based_reply(review_text, rating, sentiment, issue_types, severity, app_name)
        if extra_guidance.strip():
            result["reason"] += " | 추가 지침은 LLM 사용 시에만 반영(현재 fallback)"
        return result

    reply_type = _pick_reply_type(severity, sentiment, policy)
    prompt = build_reply_prompt(review_text, sentiment, issue_types, severity,
                                app_name, reply_type, policy, extra_guidance=extra_guidance)
    try:
        raw = call_openai_reply(prompt, settings.openai_model)
        parsed = _parse_llm_json(raw)
        draft = str(parsed.get("reply_draft", "")).strip()
        if not draft:
            raise ValueError("LLM 응답에 reply_draft 없음")

        # 안전성 점검 + 정책 검수 강제 병합
        safety = safety_check_reply(draft)
        forced, reasons = _force_review(review_text, severity, issue_types, policy)
        type_needs = policy.get("reply_types", {}).get(reply_type, {}).get("needs_human_review", False)
        llm_flag = bool(parsed.get("needs_human_review", False))
        needs_review = forced or type_needs or llm_flag or (not safety["is_safe"])

        reason = []
        if forced:
            reason.append("정책 검수 강제: " + "; ".join(reasons))
        if not safety["is_safe"]:
            reason.append("안전성 플래그: " + ", ".join(safety["safety_flags"]))
        if llm_flag:
            reason.append("LLM 검수 권고")

        if _stats is not None:
            _stats["api"] += 1
        return {
            "reply_draft": draft,
            "reply_type": parsed.get("reply_type", reply_type),
            "safety_flags": safety["safety_flags"],
            "needs_human_review": bool(needs_review),
            "reason": "[llm] " + (" / ".join(reason) or "특이사항 없음"),
        }
    except Exception as exc:
        logger.warning("LLM 답글 생성 실패 (%r) → fallback.", exc)
        if _stats is not None:
            _stats["fallback"] += 1
            _stats["errors"].append(repr(exc)[:120])
        result = rule_based_reply(review_text, rating, sentiment, issue_types, severity, app_name)
        result["reason"] += f" | LLM 오류: {repr(exc)[:80]}"
        return result


# ----------------------------------------------------------------------
# 배치 생성
# ----------------------------------------------------------------------
def generate_reply_batch(df: pd.DataFrame, n: int | None = None, save: bool = True) -> pd.DataFrame:
    """여러 리뷰를 일괄 처리해 reply_drafts.csv 로 저장한다.

    df 는 clean_text/rating 을 포함하고, 가능하면 issue_types/severity 도 포함한다.
    없으면 issue_classifier 로 즉석 분류한다.
    """
    work = df.copy()
    if "issue_types" not in work.columns or "severity" not in work.columns:
        from src.issue_classifier import classify_dataframe
        work = classify_dataframe(work)
    if n is not None:
        work = work.head(n)

    # 감성 라벨(약지도)도 입력으로 사용
    from src.train_sentiment import make_weak_label
    work["_rating"] = pd.to_numeric(work["rating"], errors="coerce")

    stats = {"api": 0, "fallback": 0, "errors": []}
    t0 = time.perf_counter()
    rows = []
    for _, r in work.iterrows():
        rating = r["_rating"]
        sentiment = make_weak_label(int(rating)) if pd.notna(rating) else "neutral"
        issue_types = r["issue_types"] if isinstance(r["issue_types"], list) else []
        result = generate_reply(
            review_text=r.get("clean_text", ""),
            rating=rating,
            sentiment=sentiment,
            issue_types=issue_types,
            severity=r.get("severity", "medium"),
            app_name=r.get("app_name", "슈퍼SOL"),
            _stats=stats,
        )
        rows.append({
            "app_name": r.get("app_name", ""),
            "rating": rating,
            "review_text": r.get("clean_text", ""),
            "sentiment": sentiment,
            "issue_types": ", ".join(issue_types),
            "severity": r.get("severity", ""),
            "reply_draft": result["reply_draft"],
            "reply_type": result["reply_type"],
            "safety_flags": ", ".join(result["safety_flags"]),
            "needs_human_review": result["needs_human_review"],
            "reason": result["reason"],
        })
    elapsed = round(time.perf_counter() - t0, 3)
    out = pd.DataFrame(rows)

    if save:
        DRAFTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(DRAFTS_PATH, index=False, encoding="utf-8-sig")
        logger.info("답글 초안 저장: %s (%d건)", DRAFTS_PATH, len(out))
        _log_performance(stats, elapsed, len(out))

    logger.info("배치 완료: API %d건 / fallback %d건 / %.3fs",
                stats["api"], stats["fallback"], elapsed)
    return out


def _log_performance(stats, elapsed, n_total) -> None:
    existing = {}
    if PERF_LOG_PATH.exists():
        try:
            with open(PERF_LOG_PATH, encoding="utf-8") as f:
                existing = json.load(f)
        except Exception:
            existing = {}
    runs = existing.get("reply_generation_runs", [])
    entry = {
        "timestamp": _now_iso(),
        "total_replies": n_total,
        "api_calls": stats["api"],
        "fallback_count": stats["fallback"],
        "api_used": stats["api"] > 0,
        "elapsed_sec": elapsed,
        "avg_sec_per_reply": round(elapsed / n_total, 4) if n_total else 0,
        "errors": stats["errors"][:10],
    }
    runs.append(entry)
    existing["reply_generation_runs"] = runs
    existing["reply_generation_last"] = entry
    PERF_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(PERF_LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)
    logger.info("성능 로그 기록: %s", PERF_LOG_PATH)


if __name__ == "__main__":
    if not CLEAN_PATH.exists():
        raise SystemExit(f"정제 데이터가 없습니다: {CLEAN_PATH}")
    df = pd.read_csv(CLEAN_PATH)
    out = generate_reply_batch(df, n=30)
    print("\n=== 답글 초안 생성 (검수용, 자동 게시 아님) ===")
    print("OPENAI_ENABLED:", settings.openai_enabled)
    print("검수 필요 건수:", int(out["needs_human_review"].sum()), "/", len(out))
    print("유형 분포:", out["reply_type"].value_counts().to_dict())
