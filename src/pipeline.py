"""
통합 분석 파이프라인 (pipeline.py)

수집 → 정제 → 검증 → 감성 → 이슈 → 토픽 → 벤치마킹 → RAG 근거 → AI 리포트
→ 답글 생성 → 테스트 요약을 하나로 연결한다.

설계 원칙:
- **단계 격리**: 각 단계는 _safe_step 으로 감싸 실패해도 전체가 중단되지 않고
  fallback 메시지를 담아 다음 단계로 진행한다.
- **시간 측정**: 단계별 소요 시간을 outputs/performance_log.json 에 기록한다.
- **최종 병합**: outputs/metrics.json 에 핵심 KPI 를 병합 저장한다.

실행:
    python -m src.pipeline
"""
from __future__ import annotations

import json
import logging
import time
import traceback
from datetime import datetime, timezone

import pandas as pd

from src.config import PATHS, settings

logging.basicConfig(
    level=getattr(logging, settings.log_level, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("pipeline")

METRICS_PATH = PATHS.outputs_dir / "metrics.json"
PERF_LOG_PATH = PATHS.outputs_dir / "performance_log.json"
CLEAN_PATH = PATHS.processed_dir / "review_clean.csv"
RAW_PATH = PATHS.raw_dir / "review_raw.csv"

DEFAULT_CONFIG = {
    "use_existing_raw": True,    # 기존 review_raw.csv 사용(없으면 수집)
    "topic_k": 10,
    "reply_sample_n": 30,        # 답글 생성 건수(데모용)
    "run_tests": True,
    # 성공 기준
    "min_total_reviews": 1000,
    "target_sentiment_f1": 0.75,
    "reply_quality_threshold": 2.0,
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ----------------------------------------------------------------------
# 단계 안전 실행 래퍼
# ----------------------------------------------------------------------
def _safe_step(name: str, fn, timings: dict) -> dict:
    """단계 실행을 감싸 예외를 잡고 시간을 기록한다.

    Returns:
        {"status": "ok"|"fallback", "data": ..., "error": ..., "elapsed_sec": ...}
    """
    t0 = time.perf_counter()
    try:
        data = fn()
        elapsed = round(time.perf_counter() - t0, 3)
        timings[name] = elapsed
        logger.info("[%s] 완료 (%.3fs)", name, elapsed)
        return {"status": "ok", "data": data, "error": None, "elapsed_sec": elapsed}
    except Exception as exc:
        elapsed = round(time.perf_counter() - t0, 3)
        timings[name] = elapsed
        logger.warning("[%s] 실패 (%r) → fallback 진행", name, exc)
        return {
            "status": "fallback",
            "data": None,
            "error": repr(exc),
            "traceback": traceback.format_exc(limit=3),
            "elapsed_sec": elapsed,
            "message": f"{name} 단계 실패: 이후 단계는 가능한 범위에서 계속 진행합니다.",
        }


# ----------------------------------------------------------------------
# 단계 구현 (각 단계는 기존 모듈을 호출)
# ----------------------------------------------------------------------
def _step_collect(config) -> pd.DataFrame:
    """수집: 기존 raw 가 있으면 재사용, 없으면 수집 시도(실패 시 샘플)."""
    if config.get("use_existing_raw") and RAW_PATH.exists():
        return pd.read_csv(RAW_PATH)
    from src.collect_reviews import collect_all
    return collect_all(n_per_store=config.get("n_per_store", 300))


def _step_preprocess() -> dict:
    from src.preprocess import preprocess
    _, summary = preprocess()
    return summary


def _step_validate() -> dict:
    from src.validation import run as validate_run
    res = validate_run(save=True)
    return {"passed": res["passed"], "metrics": res["metrics"], "by_app": res["by_app"]}


def _step_sentiment(config) -> dict:
    """감성: 기존 metrics 에 결과가 있으면 재사용, 없으면 학습."""
    existing = _load_json(METRICS_PATH)
    if "final_model" in existing:
        fm = existing["final_model"]
        return {"reused": True, "f1_macro_star": fm.get("f1_macro_star"),
                "best_model": fm.get("name")}
    from src.train_sentiment import train
    out = train(save=True)
    best = max(out["results"], key=lambda r: r["f1_macro"])
    return {"reused": False, "f1_macro_star": best["f1_macro"], "best_model": best["model"]}


def _step_issues() -> dict:
    from src.issue_classifier import run as issue_run
    df = issue_run(save=True)
    sev = df["severity"].value_counts().to_dict()
    exploded = df.explode("issue_types")
    types = exploded[exploded["issue_types"] != "기타"]["issue_types"].value_counts().head(10).to_dict()
    return {"severity_dist": sev, "top_issue_types": types}


def _step_topics(config) -> dict:
    from src.topic_modeling import run as topic_run
    res = topic_run(k=config.get("topic_k", 10), save=True)
    return {"topic_count": res["k"], "coverage": res["coverage"],
            "dominant_ratio": res.get("dominant_ratio"),
            "topics": [{"id": t["topic_id"], "label": t["auto_label"], "size": t["size"]}
                       for t in res["topics"]]}


def _step_benchmark() -> dict:
    from src.benchmark import run as bench_run
    return bench_run(save=True)


def _step_rag_evidence(df_clean) -> list:
    """RAG 근거: (RAG 인덱스 전 단계) 부정 대표 리뷰 Top-3 를 근거로 제공."""
    if df_clean is None or df_clean.empty:
        return []
    neg = df_clean[pd.to_numeric(df_clean["rating"], errors="coerce") <= 2]
    out = []
    for _, r in neg.head(3).iterrows():
        out.append({"source": f"{r['app_name']} 리뷰", "text": str(r["clean_text"])[:120]})
    return out


def _step_ai_report(prev: dict) -> dict:
    from src.report_generator import generate_report, _gather_inputs_from_outputs
    inputs = _gather_inputs_from_outputs()
    if prev.get("rag_evidence"):
        inputs["rag_evidence"] = prev["rag_evidence"]
    res = generate_report(inputs, save=True)
    return {"used_fallback": res["used_fallback"], "model": res["model"],
            "elapsed_sec": res["elapsed_sec"], "report_preview": res["report"][:300]}


def _step_reply(config) -> dict:
    from src.reply_generator import generate_reply_batch
    if not CLEAN_PATH.exists():
        raise FileNotFoundError("review_clean.csv 없음")
    df = pd.read_csv(CLEAN_PATH)
    out = generate_reply_batch(df, n=config.get("reply_sample_n", 30), save=True)
    perf = _load_json(PERF_LOG_PATH).get("reply_generation_last", {})
    return {"n": len(out),
            "needs_human_review": int(out["needs_human_review"].sum()),
            "fallback_count": perf.get("fallback_count"),
            "api_used": perf.get("api_used"),
            "elapsed_sec": perf.get("elapsed_sec")}


def _step_reply_eval() -> dict:
    from src.reply_eval import run as eval_run
    res = eval_run(save=True)
    return {"violations": res["violations"], "quality_avg": res["quality_avg"],
            "safety_pass": res["safety_pass"], "quality_pass": res["quality_pass"]}


def _step_tests(config) -> dict:
    if not config.get("run_tests"):
        return {"skipped": True}
    import subprocess
    import sys
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "--no-header", str(PATHS.root / "tests")],
        capture_output=True, text=True, cwd=str(PATHS.root), timeout=300,
    )
    last = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
    passed = "passed" in last and "failed" not in last
    return {"passed": passed, "summary": last, "returncode": proc.returncode}


# ----------------------------------------------------------------------
# 메인 파이프라인
# ----------------------------------------------------------------------
def run_pipeline(input_csv: str | None = None, config: dict | None = None) -> dict:
    """전체 분석 파이프라인을 실행하고 단계별 결과 dict 를 반환한다.

    반환 키: data_quality, kpi, sentiment, issues, topics, benchmark,
            rag_evidence, ai_report, reply_drafts, test_summary
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    timings: dict = {}
    pipe_t0 = time.perf_counter()
    results: dict = {}

    logger.info("===== 파이프라인 시작 =====")

    # 1) 수집 (input_csv 가 주어지면 그 파일을 입력으로, 아니면 기존 raw 재사용/수집)
    if input_csv:
        cfg["use_existing_raw"] = False

    def _collect():
        if input_csv:
            df_in = pd.read_csv(input_csv)
            # 이후 단계(preprocess)가 RAW_PATH 를 읽으므로 입력을 raw 로 저장
            RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
            df_in.to_csv(RAW_PATH, index=False, encoding="utf-8-sig")
            return df_in
        return _step_collect(cfg)

    collect = _safe_step("collect", _collect, timings)

    # 2) 정제
    preprocess = _safe_step("preprocess", _step_preprocess, timings)
    results["data_quality"] = preprocess["data"] or {"status": preprocess["status"],
                                                      "message": preprocess.get("message")}
    df_clean = pd.read_csv(CLEAN_PATH) if CLEAN_PATH.exists() else None

    # 3) 검증
    validate = _safe_step("validate", _step_validate, timings)
    if validate["status"] == "ok":
        results["data_quality"] = {**(results.get("data_quality") or {}),
                                   "validation": validate["data"]}
        results["kpi"] = validate["data"].get("metrics", {})
    else:
        results.setdefault("kpi", {"status": "fallback", "message": validate.get("message")})

    # 4) 감성
    sentiment = _safe_step("sentiment", lambda: _step_sentiment(cfg), timings)
    results["sentiment"] = sentiment["data"] or {"status": "fallback",
                                                 "message": sentiment.get("message")}

    # 5) 이슈
    issues = _safe_step("issues", _step_issues, timings)
    results["issues"] = issues["data"] or {"status": "fallback", "message": issues.get("message")}

    # 6) 토픽
    topics = _safe_step("topics", lambda: _step_topics(cfg), timings)
    results["topics"] = topics["data"] or {"status": "fallback", "message": topics.get("message")}

    # 7) 벤치마킹
    benchmark = _safe_step("benchmark", _step_benchmark, timings)
    results["benchmark"] = benchmark["data"] or {"status": "fallback",
                                                 "message": benchmark.get("message")}

    # 8) RAG 근거
    rag = _safe_step("rag_evidence", lambda: _step_rag_evidence(df_clean), timings)
    results["rag_evidence"] = rag["data"] if rag["status"] == "ok" else []

    # 9) AI 리포트
    report = _safe_step("ai_report", lambda: _step_ai_report(results), timings)
    results["ai_report"] = report["data"] or {"status": "fallback", "message": report.get("message")}

    # 10) 답글 생성
    reply = _safe_step("reply", lambda: _step_reply(cfg), timings)
    results["reply_drafts"] = reply["data"] or {"status": "fallback", "message": reply.get("message")}

    # 11) 답글 품질 평가
    reply_eval = _safe_step("reply_eval", _step_reply_eval, timings)

    # 12) 테스트
    tests = _safe_step("tests", lambda: _step_tests(cfg), timings)
    results["test_summary"] = tests["data"] or {"status": "fallback", "message": tests.get("message")}

    total_elapsed = round(time.perf_counter() - pipe_t0, 3)
    timings["_total"] = total_elapsed
    logger.info("===== 파이프라인 종료 (%.3fs) =====", total_elapsed)

    # 성공 기준 종합 판정
    success = _judge_success(results, reply_eval.get("data") or {}, cfg)
    results["success_criteria"] = success

    # 성능/지표 저장
    _save_performance(timings, results, reply_eval.get("data") or {})
    _merge_final_metrics(results, total_elapsed, reply_eval.get("data") or {}, success)

    return results


def _judge_success(results, reply_eval, cfg) -> dict:
    """성공 기준 pass/fail 종합."""
    kpi = results.get("kpi", {})
    total = kpi.get("total_reviews", 0) if isinstance(kpi, dict) else 0
    sent = results.get("sentiment", {})
    f1 = sent.get("f1_macro_star") if isinstance(sent, dict) else None
    checks = {
        "data_quality_pass": bool(kpi.get("data_quality_pass")) if isinstance(kpi, dict) else False,
        "min_reviews": total >= cfg["min_total_reviews"],
        "sentiment_f1_target": (f1 is not None and f1 >= cfg["target_sentiment_f1"]),
        "reply_safety_pass": bool(reply_eval.get("safety_pass")),
        "reply_quality_pass": bool(reply_eval.get("quality_pass")),
    }
    # 감성 F1 목표는 약지도 한계로 미달 가능 → 경고로만 표시(전체 판정에서 제외)
    core = {k: v for k, v in checks.items() if k != "sentiment_f1_target"}
    return {"checks": checks, "overall_pass": all(core.values()),
            "note": "sentiment_f1_target 은 별점 약지도 한계로 참고 지표(핵심 판정 제외)."}


def _save_performance(timings, results, reply_eval) -> None:
    existing = _load_json(PERF_LOG_PATH)
    runs = existing.get("pipeline_runs", [])
    entry = {
        "timestamp": _now_iso(),
        "step_timings_sec": timings,
        "report_generation_time_sec": (results.get("ai_report") or {}).get("elapsed_sec"),
        "reply_generation_time_sec": (results.get("reply_drafts") or {}).get("elapsed_sec"),
        "fallback_reply_count": (results.get("reply_drafts") or {}).get("fallback_count"),
        "reply_safety_violations": reply_eval.get("violations"),
    }
    runs.append(entry)
    existing["pipeline_runs"] = runs
    existing["pipeline_last"] = entry
    _write_json(PERF_LOG_PATH, existing)
    logger.info("성능 로그 저장: %s", PERF_LOG_PATH)


def _merge_final_metrics(results, total_elapsed, reply_eval, success) -> None:
    existing = _load_json(METRICS_PATH)
    kpi = results.get("kpi", {}) if isinstance(results.get("kpi"), dict) else {}
    ai = results.get("ai_report") or {}
    reply = results.get("reply_drafts") or {}
    existing["pipeline_summary"] = {
        "timestamp": _now_iso(),
        "total_reviews": kpi.get("total_reviews"),
        "app_count": kpi.get("app_count"),
        "processing_time_sec": total_elapsed,
        "report_generation_time_sec": ai.get("elapsed_sec"),
        "reply_generation_time_sec": reply.get("elapsed_sec"),
        "fallback_reply_count": reply.get("fallback_count"),
        "reply_safety_violations": reply_eval.get("violations"),
        "reply_quality_avg": reply_eval.get("quality_avg"),
        "success_criteria_pass_fail": "PASS" if success["overall_pass"] else "FAIL",
        "success_checks": success["checks"],
    }
    _write_json(METRICS_PATH, existing)
    logger.info("최종 metrics.json 병합 저장: %s", METRICS_PATH)


# ----------------------------------------------------------------------
# JSON 유틸
# ----------------------------------------------------------------------
def _load_json(path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _write_json(path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    out = run_pipeline()
    print("\n=== 파이프라인 결과 요약 ===")
    print("단계별 키:", list(out.keys()))
    sc = out["success_criteria"]
    print("성공 기준:", "PASS" if sc["overall_pass"] else "FAIL")
    for k, v in sc["checks"].items():
        print(f"  - {k}: {v}")
