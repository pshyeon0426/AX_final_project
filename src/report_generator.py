"""
CX 분석 리포트 생성 (report_generator.py)

목적:
- 분석 지표 + RAG 근거를 바탕으로 '근거 기반' CX 요약 리포트를 생성한다.
- OpenAI API 사용이 가능하면 LLM 으로, 불가하면 룰 기반 요약으로 fallback 한다.

출력 6개 섹션:
    1. 핵심 요약  2. 부정 반응 원인  3. 개선 우선순위
    4. 경쟁 앱 벤치마킹 포인트  5. 근거 리뷰/문서  6. 데이터 한계

⚠️ 원칙:
- 근거 없는 추정 금지 / 내부 전략 추정 금지 / 공개 데이터 기준 (프롬프트에 명시).
- 실제 API 키는 .env(OPENAI_API_KEY)에서만 주입하며 하드코딩하지 않는다.

실행:
    python -m src.report_generator
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone

from src.config import PATHS, settings

logging.basicConfig(
    level=getattr(logging, settings.log_level, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("report_generator")

# ----------------------------------------------------------------------
# 경로
# ----------------------------------------------------------------------
PROMPT_PATH = PATHS.prompts_dir / "report_prompt.md"
REPORT_PATH = PATHS.outputs_dir / "ai_report.md"
PERF_LOG_PATH = PATHS.outputs_dir / "performance_log.json"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ----------------------------------------------------------------------
# 입력 직렬화
# ----------------------------------------------------------------------
def _fmt_block(value) -> str:
    """dict/list 를 사람이 읽기 좋은 문자열로."""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, indent=2)
    return str(value)


def build_prompt(inputs: dict) -> str:
    """report_prompt.md 템플릿에 입력값을 채운다."""
    template = PROMPT_PATH.read_text(encoding="utf-8")
    return template.format(
        kpi=_fmt_block(inputs.get("kpi", {})),
        sentiment_dist=_fmt_block(inputs.get("sentiment_dist", {})),
        top_complaints=_fmt_block(inputs.get("top_complaints", {})),
        topic_summary=_fmt_block(inputs.get("topic_summary", [])),
        benchmark=_fmt_block(inputs.get("benchmark", [])),
        rag_evidence=_fmt_block(inputs.get("rag_evidence", [])),
    )


# ----------------------------------------------------------------------
# LLM 생성 (OpenAI)
# ----------------------------------------------------------------------
def _generate_llm(prompt: str) -> str:
    """OpenAI API 로 리포트를 생성한다. 실패 시 예외를 올린다(상위에서 fallback)."""
    from openai import OpenAI  # 지연 import

    client = OpenAI(api_key=settings.require("openai_api_key"))
    model = settings.require("openai_model")
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content":
             "너는 공개 데이터 기반으로만 작성하는 금융 앱 CX 분석가다. "
             "근거 없는 추정과 경쟁사 내부 전략 추정을 하지 않는다."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
    )
    return resp.choices[0].message.content


# ----------------------------------------------------------------------
# 룰 기반 fallback
# ----------------------------------------------------------------------
def _generate_rule_based(inputs: dict) -> str:
    """LLM 없이 입력 지표를 조합해 6개 섹션 리포트를 만든다."""
    kpi = inputs.get("kpi", {})
    sent = inputs.get("sentiment_dist", {})
    complaints = inputs.get("top_complaints", {})
    topics = inputs.get("topic_summary", [])
    benchmark = inputs.get("benchmark", [])
    rag = inputs.get("rag_evidence", [])

    our = kpi.get("our_app", "슈퍼SOL")

    # 핵심 요약용 수치
    avg = kpi.get("our_avg_rating", "N/A")
    neg = kpi.get("our_negative_ratio")
    neg_str = f"{neg:.1%}" if isinstance(neg, (int, float)) else "N/A"
    rank = kpi.get("our_rating_rank", "N/A")
    total = kpi.get("total_reviews", "N/A")

    # 우리 앱 불만 유형
    our_complaints = complaints.get(our, [])
    comp_str = ", ".join(f"{t}({c})" for t, c in our_complaints[:5]) or "데이터 부족"

    lines = [
        f"# {our} CX 분석 리포트 (룰 기반 자동 생성)",
        "",
        "> ⚠️ 이 리포트는 LLM 미사용 시 **룰 기반 fallback**으로 생성되었습니다. "
        "모든 서술은 공개 리뷰/문서 지표에 근거하며, 근거 없는 추정·내부 전략 추정을 하지 않습니다.",
        "",
        "## 1. 핵심 요약",
        f"- 분석 리뷰 총 {total}건 기준, {our} 평균 평점 **{avg}**, 부정 리뷰 비율 **{neg_str}**.",
        f"- 경쟁 앱 포함 평점 순위: **{rank}위**.",
        f"- 감성 분포(약지도): {_fmt_inline(sent)}.",
        f"- 주요 불만 유형: {comp_str}.",
        "",
        "## 2. 부정 반응 원인",
    ]
    if our_complaints:
        for t, c in our_complaints[:3]:
            lines.append(f"- **{t}** 관련 불만이 부정 리뷰에서 {c}건으로 두드러짐.")
    else:
        lines.append("- 부정 반응 유형 데이터가 부족합니다.")
    if topics:
        kw = ", ".join(_topic_label(t) for t in topics[:3])
        lines.append(f"- 주요 토픽 키워드: {kw}.")

    lines += ["", "## 3. 개선 우선순위"]
    prio = ["상", "중", "하"]
    for i, (t, c) in enumerate(our_complaints[:3]):
        lines.append(f"- [{prio[i] if i < 3 else '하'}] **{t}** 개선 — 부정 리뷰 {c}건 근거.")
    if not our_complaints:
        lines.append("- 우선순위 산정에 필요한 불만 유형 데이터가 부족합니다.")

    lines += ["", "## 4. 경쟁 앱 벤치마킹 포인트"]
    if benchmark:
        # 우리보다 평점 높은 경쟁 앱
        better = [b for b in benchmark
                  if b.get("app_name") != our
                  and isinstance(b.get("avg_rating"), (int, float))
                  and isinstance(avg, (int, float))
                  and b["avg_rating"] > avg]
        if better:
            for b in better[:3]:
                lines.append(
                    f"- **{b['app_name']}**: 평점 {b['avg_rating']}, "
                    f"부정 {b.get('negative_ratio', 0):.1%} — 공개 지표상 {our}보다 우위."
                )
        else:
            lines.append(f"- 공개 지표상 {our}보다 뚜렷이 앞서는 경쟁 앱은 확인되지 않음.")
        lines.append("- (경쟁사 내부 전략은 추정하지 않음. 공개 리뷰 지표 기준.)")
    else:
        lines.append("- 벤치마킹 데이터가 제공되지 않았습니다.")

    lines += ["", "## 5. 근거 리뷰/문서"]
    if rag:
        for i, r in enumerate(rag[:3], 1):
            lines.append(f"{i}. {_rag_line(r)}")
    else:
        lines.append("- 제공된 RAG 근거가 없습니다.")

    lines += [
        "",
        "## 6. 데이터 한계",
        "- 감성 라벨은 **별점 기반 약지도 라벨**로 실제 감정 정답이 아니며, 성능은 별점 재현도 기준.",
        "- 리뷰는 공개 스토어 표본으로 전체 사용자를 대표하지 않으며, App Store 는 RSS 수집 상한이 있음.",
        "- 불만 유형은 **키워드 룰 기반** 분류로 오탐/누락 가능성이 있음.",
        "- 수집 기간/표본이 앱마다 달라 절대 비교 시 주의가 필요함.",
        "",
    ]
    return "\n".join(lines)


def _fmt_inline(d) -> str:
    if isinstance(d, dict):
        return ", ".join(f"{k} {v}" for k, v in d.items())
    return str(d)


def _topic_label(t) -> str:
    if isinstance(t, dict):
        return t.get("auto_label") or t.get("topic_label") or str(t.get("topic_id", ""))
    return str(t)


def _rag_line(r) -> str:
    if isinstance(r, dict):
        src = r.get("source", r.get("app_name", "출처미상"))
        text = r.get("text", r.get("clean_text", ""))
        return f"({src}) {text}"
    return str(r)


# ----------------------------------------------------------------------
# 메인
# ----------------------------------------------------------------------
def generate_report(inputs: dict, save: bool = True, force_fallback: bool = False) -> dict:
    """리포트를 생성한다.

    Returns:
        {"report": str, "used_fallback": bool, "elapsed_sec": float, "model": str|None}
    """
    used_fallback = False
    model_used = None
    error = None
    t0 = time.perf_counter()

    if not force_fallback and settings.openai_enabled:
        try:
            prompt = build_prompt(inputs)
            report = _generate_llm(prompt)
            model_used = settings.openai_model
            logger.info("LLM(%s)로 리포트 생성 완료", model_used)
        except Exception as exc:
            logger.warning("LLM 생성 실패 (%r) → 룰 기반 fallback.", exc)
            error = repr(exc)
            report = _generate_rule_based(inputs)
            used_fallback = True
    else:
        reason = "force_fallback" if force_fallback else "OPENAI 미설정"
        logger.info("LLM 미사용(%s) → 룰 기반 fallback.", reason)
        report = _generate_rule_based(inputs)
        used_fallback = True

    elapsed = round(time.perf_counter() - t0, 3)

    if save:
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text(report, encoding="utf-8")
        logger.info("리포트 저장: %s", REPORT_PATH)
        _log_performance(used_fallback, elapsed, model_used, error)

    return {"report": report, "used_fallback": used_fallback,
            "elapsed_sec": elapsed, "model": model_used}


def _log_performance(used_fallback, elapsed, model, error) -> None:
    """LLM 호출 시간/ fallback 여부를 performance_log.json 에 기록(append)."""
    existing = {}
    if PERF_LOG_PATH.exists():
        try:
            with open(PERF_LOG_PATH, encoding="utf-8") as f:
                existing = json.load(f)
        except Exception:
            existing = {}
    runs = existing.get("report_generation_runs", [])
    runs.append({
        "timestamp": _now_iso(),
        "used_fallback": used_fallback,
        "elapsed_sec": elapsed,
        "model": model,
        "error": error,
    })
    existing["report_generation_runs"] = runs
    existing["report_generation_last"] = runs[-1]
    PERF_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(PERF_LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)
    logger.info("성능 로그 기록: %s (fallback=%s, %.3fs)", PERF_LOG_PATH, used_fallback, elapsed)


def _gather_inputs_from_outputs() -> dict:
    """outputs/ 의 분석 결과를 모아 리포트 입력 dict 를 구성한다(데모용)."""
    import pandas as pd

    inputs: dict = {"kpi": {}, "sentiment_dist": {}, "top_complaints": {},
                    "topic_summary": [], "benchmark": [], "rag_evidence": []}

    metrics_path = PATHS.outputs_dir / "metrics.json"
    if metrics_path.exists():
        m = json.loads(metrics_path.read_text(encoding="utf-8"))
        bench = m.get("benchmark", {})
        inputs["kpi"] = {
            "our_app": bench.get("our_app", "슈퍼SOL"),
            "total_reviews": m.get("total_reviews"),
            "our_avg_rating": bench.get("our_avg_rating"),
            "our_negative_ratio": bench.get("our_negative_ratio"),
            "our_rating_rank": bench.get("our_rating_rank"),
        }

    # 감성 분포
    sent_csv = PATHS.processed_dir / "review_clean.csv"
    if sent_csv.exists():
        from src.train_sentiment import make_weak_label
        df = pd.read_csv(sent_csv)
        df["label"] = pd.to_numeric(df["rating"], errors="coerce").dropna().astype(int).map(make_weak_label)
        inputs["sentiment_dist"] = df["label"].value_counts().to_dict()

    # 벤치마킹 + 불만 유형
    bench_csv = PATHS.outputs_dir / "benchmark_summary.csv"
    if bench_csv.exists():
        b = pd.read_csv(bench_csv)
        inputs["benchmark"] = b[["app_name", "avg_rating", "negative_ratio",
                                 "review_count"]].to_dict(orient="records")
        our = inputs["kpi"].get("our_app", "슈퍼SOL")
        row = b[b["app_name"] == our]
        if not row.empty and "top_complaint_types" in b.columns:
            raw = str(row.iloc[0]["top_complaint_types"])
            parsed = []
            for part in raw.split("|"):
                if ":" in part:
                    t, c = part.rsplit(":", 1)
                    try:
                        parsed.append((t.strip(), int(c)))
                    except ValueError:
                        pass
            inputs["top_complaints"] = {our: parsed}

    # 토픽 요약
    topic_csv = PATHS.outputs_dir / "topic_summary.csv"
    if topic_csv.exists():
        t = pd.read_csv(topic_csv)
        inputs["topic_summary"] = t[["topic_id", "size", "auto_label"]].to_dict(orient="records")

    # RAG 근거: 부정 대표 리뷰 3건 (RAG 단계 전 데모용)
    if sent_csv.exists():
        neg = pd.read_csv(sent_csv)
        neg = neg[pd.to_numeric(neg["rating"], errors="coerce") <= 2]
        for _, r in neg.head(3).iterrows():
            inputs["rag_evidence"].append({
                "source": f"{r['app_name']} 리뷰",
                "text": str(r["clean_text"])[:120],
            })
    return inputs


if __name__ == "__main__":
    inputs = _gather_inputs_from_outputs()
    result = generate_report(inputs)
    print("\n=== AI 리포트 생성 ===")
    print(f"fallback 사용: {result['used_fallback']} / 모델: {result['model']} "
          f"/ 소요: {result['elapsed_sec']}s")
    print(f"저장: {REPORT_PATH}")
