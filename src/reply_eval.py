"""
답글 생성 품질 검증 (reply_eval.py)

목적:
- 생성된 답글 초안의 안전성/품질을 자동 점검하고, 사람이 검수할 평가 샘플과
  리포트를 만든다.
- 안전성 위반 후보는 별도 리스크 파일로 분리한다.

자동 점수(3점 만점, 사람이 human_comment 로 보정 가능):
    tone_score        : 존댓말/길이(2~4문장) 등 톤 적합성
    safety_score      : 안전성 위반 없으면 3, 있으면 0
    helpfulness_score : 안내/공감/행동 유도 표현 포함도

성공 기준:
    - 안전성 위반 0건 목표
    - 품질(tone/helpfulness) 평균 ≥ 2.0 (3점 만점)

실행:
    python -m src.reply_eval
"""
from __future__ import annotations

import json
import logging
import re

import pandas as pd

from src.config import PATHS, settings
from src.reply_generator import safety_check_reply

logging.basicConfig(
    level=getattr(logging, settings.log_level, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("reply_eval")

DRAFTS_PATH = PATHS.outputs_dir / "reply_drafts.csv"
EVAL_SAMPLE_PATH = PATHS.outputs_dir / "reply_eval_sample.csv"
EVAL_REPORT_PATH = PATHS.outputs_dir / "reply_eval_report.md"
RISK_CASES_PATH = PATHS.outputs_dir / "reply_risk_cases.csv"

QUALITY_THRESHOLD = 2.0   # 품질 평균 성공 기준
SAMPLE_SIZE = 20


def _tone_score(reply: str) -> int:
    """톤 점수(0~3): 존댓말 어미 + 문장 수(2~4) 기준."""
    if not reply:
        return 0
    score = 1
    # 존댓말 어미
    if re.search(r"(습니다|니다|세요|해요|드립니다|십시오)", reply):
        score += 1
    # 문장 수 2~4 적정
    n_sent = len([s for s in re.split(r"[.!?]\s*", reply) if s.strip()])
    if 2 <= n_sent <= 4:
        score += 1
    return min(score, 3)


def _helpfulness_score(reply: str) -> int:
    """유용성 점수(0~3): 공감/사과 + 안내/행동유도 표현 포함도."""
    if not reply:
        return 0
    score = 1
    empathy = any(w in reply for w in ["죄송", "감사", "불편", "공감", "양해"])
    action = any(w in reply for w in ["고객센터", "문의", "안내", "확인", "검토", "참고"])
    if empathy:
        score += 1
    if action:
        score += 1
    return min(score, 3)


def evaluate(df: pd.DataFrame) -> dict:
    """답글 초안 전체를 점수화하고 리스크 케이스를 식별한다."""
    work = df.copy()
    safety_results = work["reply_draft"].apply(safety_check_reply)
    work["safety_flags_eval"] = safety_results.apply(lambda r: ", ".join(r["safety_flags"]))
    work["is_safe"] = safety_results.apply(lambda r: r["is_safe"])
    work["tone_score"] = work["reply_draft"].apply(_tone_score)
    work["safety_score"] = work["is_safe"].apply(lambda s: 3 if s else 0)
    work["helpfulness_score"] = work["reply_draft"].apply(_helpfulness_score)

    n = len(work)
    violations = int((~work["is_safe"]).sum())
    quality_avg = round(float((work["tone_score"] + work["helpfulness_score"]).mean() / 2), 3)
    tone_avg = round(float(work["tone_score"].mean()), 3)
    help_avg = round(float(work["helpfulness_score"].mean()), 3)
    safety_avg = round(float(work["safety_score"].mean()), 3)

    return {
        "df": work,
        "n": n,
        "violations": violations,
        "quality_avg": quality_avg,
        "tone_avg": tone_avg,
        "helpfulness_avg": help_avg,
        "safety_avg": safety_avg,
        "safety_pass": violations == 0,
        "quality_pass": quality_avg >= QUALITY_THRESHOLD,
    }


def run(save: bool = True) -> dict:
    if not DRAFTS_PATH.exists():
        raise FileNotFoundError(
            f"답글 초안이 없습니다: {DRAFTS_PATH} (먼저 python -m src.reply_generator 실행)"
        )
    df = pd.read_csv(DRAFTS_PATH)
    result = evaluate(df)
    work = result["df"]

    # 평가 샘플 20건 이상 (다양성 위해 reply_type 층화 후 부족분 보충)
    sample_n = max(SAMPLE_SIZE, 0)
    if len(work) <= sample_n:
        sample = work.copy()
    else:
        parts = []
        for _, g in work.groupby("reply_type"):
            parts.append(g.sample(min(len(g), max(1, sample_n // work["reply_type"].nunique())),
                                  random_state=42))
        sample = pd.concat(parts)
        if len(sample) < sample_n:
            extra = work.drop(sample.index).sample(sample_n - len(sample), random_state=42)
            sample = pd.concat([sample, extra])
        sample = sample.head(max(sample_n, len(sample)))

    if save:
        _save_sample(sample)
        _save_risk_cases(work)
        _save_report(result)

    logger.info("평가 완료: n=%d, 위반 %d건, 품질평균 %.2f (tone %.2f / help %.2f)",
                result["n"], result["violations"], result["quality_avg"],
                result["tone_avg"], result["helpfulness_avg"])
    return result


def _save_sample(sample: pd.DataFrame) -> None:
    cols = ["app_name", "rating", "review_text", "reply_draft", "reply_type",
            "severity", "needs_human_review", "tone_score", "safety_score",
            "helpfulness_score"]
    cols = [c for c in cols if c in sample.columns]
    out = sample[cols].copy()
    out["human_comment"] = ""   # 검수자 입력용 공란
    EVAL_SAMPLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(EVAL_SAMPLE_PATH, index=False, encoding="utf-8-sig")
    logger.info("평가 샘플 저장: %s (%d건)", EVAL_SAMPLE_PATH, len(out))


def _save_risk_cases(work: pd.DataFrame) -> None:
    risk = work[~work["is_safe"]].copy()
    cols = ["app_name", "rating", "review_text", "reply_draft", "reply_type",
            "safety_flags_eval", "needs_human_review"]
    cols = [c for c in cols if c in risk.columns]
    RISK_CASES_PATH.parent.mkdir(parents=True, exist_ok=True)
    risk[cols].to_csv(RISK_CASES_PATH, index=False, encoding="utf-8-sig")
    logger.info("리스크 케이스 저장: %s (%d건)", RISK_CASES_PATH, len(risk))


def _save_report(result: dict) -> None:
    work = result["df"]
    safety_mark = "✅ 통과 (위반 0건)" if result["safety_pass"] else f"❌ 위반 {result['violations']}건"
    quality_mark = "✅ 통과" if result["quality_pass"] else "❌ 미달"
    lines = [
        "# 답글 생성 품질 검증 리포트",
        "",
        "> 생성 답글은 **자동 게시가 아니라 담당자 검수용 초안**입니다.",
        "",
        "## 종합 결과",
        "",
        f"- 평가 대상: **{result['n']}건**",
        f"- LLM 사용 여부: {'예' if settings.openai_enabled else '아니오 (룰 기반 fallback)'}",
        f"- **안전성**: {safety_mark} (성공 기준: 위반 0건)",
        f"- **품질 평균**: {result['quality_avg']:.2f} / 3.0 → {quality_mark} (성공 기준: ≥ {QUALITY_THRESHOLD})",
        f"  - 톤 평균: {result['tone_avg']:.2f} / 유용성 평균: {result['helpfulness_avg']:.2f} / 안전성 평균: {result['safety_avg']:.2f}",
        "",
        "## reply_type 분포",
        "",
        "| 유형 | 건수 | 검수필요 |",
        "|------|-----:|--------:|",
    ]
    for rtype, g in work.groupby("reply_type"):
        nr = int(g["needs_human_review"].sum()) if "needs_human_review" in g else 0
        lines.append(f"| {rtype} | {len(g)} | {nr} |")

    # 점수 분포
    lines += ["", "## 점수 분포 (0~3)", "",
              "| 점수 | tone | safety | helpfulness |",
              "|------|-----:|-------:|------------:|"]
    for s in [0, 1, 2, 3]:
        t = int((work["tone_score"] == s).sum())
        sa = int((work["safety_score"] == s).sum())
        h = int((work["helpfulness_score"] == s).sum())
        lines.append(f"| {s}점 | {t} | {sa} | {h} |")

    # 성공 기준 판정 및 미달 원인
    lines += ["", "## 성공 기준 판정", ""]
    if result["safety_pass"] and result["quality_pass"]:
        lines.append("- ✅ **발표 가능 수준**: 안전성 위반 0건, 품질 평균 기준 충족.")
    else:
        lines.append("- ⚠️ **개선 필요**:")
        if not result["safety_pass"]:
            lines.append(
                f"  - 안전성 위반 {result['violations']}건 → `reply_risk_cases.csv` 참고. "
                "원인: 금칙어/민감정보 요청/확정 약속/투자 조언 표현. "
                "개선: 프롬프트 제약 강화, banned_phrases 보강, 위반 답글 자동 폐기·재생성.")
        if not result["quality_pass"]:
            lines.append(
                f"  - 품질 평균 {result['quality_avg']:.2f} < {QUALITY_THRESHOLD} → "
                "원인: 템플릿 단조로움(룰 기반)·공감/안내 표현 부족. "
                "개선: 유형별 템플릿 다양화, LLM(OPENAI_API_KEY) 활성화, 톤 가이드 강화.")

    lines += [
        "",
        "## 한계",
        "- 점수는 규칙 기반 **자동 산정**이며, 최종 품질은 `reply_eval_sample.csv` 의 "
        "`human_comment` 로 사람이 보정해야 한다.",
        f"- 현재 답글은 {'LLM' if settings.openai_enabled else '룰 기반 fallback'} 으로 생성됨.",
        "",
    ]
    EVAL_REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    logger.info("평가 리포트 저장: %s", EVAL_REPORT_PATH)


if __name__ == "__main__":
    res = run()
    print("\n=== 답글 품질 검증 ===")
    print(f"평가 {res['n']}건 | 안전성 위반 {res['violations']}건 "
          f"({'PASS' if res['safety_pass'] else 'FAIL'})")
    print(f"품질 평균 {res['quality_avg']:.2f}/3.0 "
          f"({'PASS' if res['quality_pass'] else 'FAIL'})")
