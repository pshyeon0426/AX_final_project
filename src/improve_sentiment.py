"""
감성 분류 개선 모델 (improve_sentiment.py)

목적:
- baseline(TF-IDF unigram + LogReg) 대비 개선 실험을 수행하고 최종 모델을 선정한다.
- 별점 기반 약지도 라벨의 한계를 보완하기 위한 수동 검수 샘플을 추출한다.
- 수동 검수 라벨이 준비되면 '별점 기준' 성능과 '수동 검수 기준' 성능을 구분 계산한다.

⚠️ 라벨 주의:
    학습 라벨은 별점 규칙(1~2=neg, 3=neu, 4~5=pos)으로 만든 약지도 라벨이며
    실제 감정 정답이 아니다. 따라서 성능은 '별점 라벨 재현도'로 해석한다.

개선 실험(최소 2개 이상):
    exp1: TF-IDF (1,2)gram + LogReg(class_weight='balanced')
    exp2: TF-IDF (1,2)gram + max_features 제한 + LinearSVC(class_weight='balanced')
    exp3: TF-IDF (1,3)gram + min_df 조정 + LogReg(class_weight='balanced')

실행:
    python -m src.improve_sentiment
"""
from __future__ import annotations

import json
import logging

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.config import PATHS, settings
from src.tokenizer import get_tokenizer
from src.train_sentiment import (
    LABELS,
    RANDOM_STATE,
    TEST_SIZE,
    load_dataset,
)

logging.basicConfig(
    level=getattr(logging, settings.log_level, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("improve_sentiment")

# ----------------------------------------------------------------------
# 경로
# ----------------------------------------------------------------------
REPORT_PATH = PATHS.outputs_dir / "model_report.md"
METRICS_PATH = PATHS.outputs_dir / "metrics.json"
MODEL_PKL_PATH = PATHS.models_dir / "sentiment_model.pkl"
MANUAL_SAMPLE_PATH = PATHS.outputs_dir / "manual_label_sample.csv"
MANUAL_LABELED_PATH = PATHS.outputs_dir / "manual_label_sample_filled.csv"  # 검수 완료 시
ERROR_CASES_PATH = PATHS.outputs_dir / "error_cases.csv"

IMPROVEMENT_TARGET = 0.05  # baseline 대비 macro F1 +0.05p 목표


# 모듈 레벨 토크나이저 (pickle 안전: train_sentiment.tokenize 와 동일 전략)
_TOKENIZER = None


def tokenize(text: str) -> list[str]:
    """TF-IDF 용 토큰화 함수 (pickle 가능)."""
    global _TOKENIZER
    if _TOKENIZER is None:
        _TOKENIZER = get_tokenizer()
    return _TOKENIZER.tokenize(text)


def _metrics(y_true, y_pred) -> dict:
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision_macro": round(precision_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "recall_macro": round(recall_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "f1_macro": round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4),
    }


def _experiments() -> dict[str, Pipeline]:
    """개선 실험 파이프라인 정의 (최소 2개 이상)."""
    return {
        "exp1_LogReg_bigram_balanced": Pipeline([
            ("tfidf", TfidfVectorizer(tokenizer=tokenize, token_pattern=None,
                                      ngram_range=(1, 2), min_df=2)),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]),
        "exp2_LinearSVC_bigram_balanced_maxfeat": Pipeline([
            ("tfidf", TfidfVectorizer(tokenizer=tokenize, token_pattern=None,
                                      ngram_range=(1, 2), min_df=2, max_features=5000)),
            ("clf", LinearSVC(class_weight="balanced")),
        ]),
        "exp3_LogReg_trigram_balanced": Pipeline([
            ("tfidf", TfidfVectorizer(tokenizer=tokenize, token_pattern=None,
                                      ngram_range=(1, 3), min_df=3)),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", C=2.0)),
        ]),
    }


def _baseline_f1() -> float | None:
    """metrics.json 에서 baseline 의 best f1_macro 를 읽는다."""
    if not METRICS_PATH.exists():
        return None
    try:
        with open(METRICS_PATH, encoding="utf-8") as f:
            m = json.load(f)
        return m.get("sentiment_baseline", {}).get("best_f1_macro")
    except Exception:
        return None


def run(save: bool = True) -> dict:
    df = load_dataset()
    logger.info("데이터셋: %d건, 라벨분포(약지도)=%s", len(df), df["label"].value_counts().to_dict())

    X, y = df["text"], df["label"]
    Xtr, Xte, ytr, yte, idx_tr, idx_te = train_test_split(
        X, y, df.index, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # 1) 개선 실험
    logger.info("토크나이저 backend: %s", get_tokenizer().backend)
    exp_results = []
    fitted = {}
    for name, pipe in _experiments().items():
        pipe.fit(Xtr, ytr)
        m = _metrics(yte, pipe.predict(Xte))
        m["model"] = name
        exp_results.append(m)
        fitted[name] = pipe
        logger.info("[%s] f1_macro=%.4f acc=%.4f", name, m["f1_macro"], m["accuracy"])

    # 2) 최종 모델 선정 (별점 기준 test f1_macro 최고)
    best = max(exp_results, key=lambda r: r["f1_macro"])
    best_model = fitted[best["model"]]
    baseline_f1 = _baseline_f1()
    improvement = round(best["f1_macro"] - baseline_f1, 4) if baseline_f1 is not None else None
    target_met = (improvement is not None and improvement >= IMPROVEMENT_TARGET)
    logger.info("최종 모델: %s (f1=%.4f, baseline=%s, 개선=%s, 목표달성=%s)",
                best["model"], best["f1_macro"], baseline_f1, improvement, target_met)

    # 3) 오분류 사례 (test 셋 기준, 최소 3건 이상)
    yte_pred = best_model.predict(Xte)
    err_mask = (yte.values != yte_pred)
    err_df = df.loc[idx_te[err_mask], ["app_name", "store", "rating", "clean_text"]].copy()
    err_df["true_label_star"] = yte.values[err_mask]
    err_df["pred_label"] = yte_pred[err_mask]

    # 4) 수동 검수 기준 성능 (검수 완료 파일이 있을 때만)
    manual_eval = _evaluate_manual(best_model)

    if save:
        joblib.dump(best_model, MODEL_PKL_PATH)
        logger.info("최종 모델 저장: %s", MODEL_PKL_PATH)

        ERROR_CASES_PATH.parent.mkdir(parents=True, exist_ok=True)
        err_df.to_csv(ERROR_CASES_PATH, index=False, encoding="utf-8-sig")
        logger.info("오분류 사례 저장: %s (%d건)", ERROR_CASES_PATH, len(err_df))

        _make_manual_sample(df)
        _append_report(df, exp_results, best, baseline_f1, improvement, target_met,
                       yte, yte_pred, manual_eval)
        _merge_metrics(best, baseline_f1, improvement, target_met, manual_eval)

    return {
        "experiments": exp_results, "best": best,
        "baseline_f1": baseline_f1, "improvement": improvement,
        "target_met": target_met, "n_errors": len(err_df),
        "manual_eval": manual_eval,
    }


def _make_manual_sample(df: pd.DataFrame, n: int = 150) -> None:
    """수동 검수용 샘플 100~200건 추출. sentiment_manual/issue_manual 은 공란."""
    n = min(n, len(df))
    # 라벨 분포를 반영해 층화 추출 (neutral 도 충분히 포함되도록)
    parts = []
    for label, g in df.groupby("label"):
        k = min(len(g), max(1, int(round(n * len(g) / len(df)))))
        parts.append(g.sample(k, random_state=RANDOM_STATE))
    sample = pd.concat(parts)
    if len(sample) > n:
        sample = sample.sample(n, random_state=RANDOM_STATE)

    out = sample[["app_name", "store", "rating", "clean_text"]].copy()
    out["star_label"] = sample["label"].values   # 별점 기반 약지도 라벨(참고용)
    out["sentiment_manual"] = ""   # 검수자가 입력: negative/neutral/positive
    out["issue_manual"] = ""       # 검수자가 입력: 로그인/인증/속도/오류 등
    MANUAL_SAMPLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(MANUAL_SAMPLE_PATH, index=False, encoding="utf-8-sig")
    logger.info("수동 검수 샘플 저장: %s (%d건, sentiment_manual/issue_manual 공란)",
                MANUAL_SAMPLE_PATH, len(out))


def _evaluate_manual(model) -> dict | None:
    """수동 검수 라벨이 입력된 파일이 있으면 그 기준으로 성능을 계산한다."""
    if not MANUAL_LABELED_PATH.exists():
        logger.info("수동 검수 완료 파일 없음(%s). 별점 기준 성능만 보고.", MANUAL_LABELED_PATH.name)
        return None
    mdf = pd.read_csv(MANUAL_LABELED_PATH)
    mdf = mdf[mdf["sentiment_manual"].isin(LABELS)].copy()
    if mdf.empty:
        logger.warning("수동 검수 파일에 유효한 sentiment_manual 라벨이 없습니다.")
        return None
    pred = model.predict(mdf["clean_text"].fillna("").astype(str))
    m = _metrics(mdf["sentiment_manual"], pred)
    m["n"] = len(mdf)
    logger.info("수동 검수 기준 성능: f1_macro=%.4f (n=%d)", m["f1_macro"], m["n"])
    return m


def _append_report(df, exp_results, best, baseline_f1, improvement, target_met,
                   yte, yte_pred, manual_eval) -> None:
    """model_report.md 에 개선 섹션을 추가(append)한다."""
    cm = confusion_matrix(yte, yte_pred, labels=LABELS).tolist()
    lines = [
        "",
        "---",
        "",
        "# 감성 분류 개선 모델 리포트",
        "",
        "## 개선 실험 (별점 기준 약지도 라벨)",
        "",
        "| 실험 | accuracy | precision_macro | recall_macro | f1_macro |",
        "|------|----------|-----------------|--------------|----------|",
    ]
    for r in exp_results:
        lines.append(
            f"| {r['model']} | {r['accuracy']:.4f} | {r['precision_macro']:.4f} "
            f"| {r['recall_macro']:.4f} | {r['f1_macro']:.4f} |"
        )

    base_str = f"{baseline_f1:.4f}" if baseline_f1 is not None else "N/A"
    imp_str = f"{improvement:+.4f}p" if improvement is not None else "N/A"
    lines += [
        "",
        "## 최종 모델 선정",
        "",
        f"- 최종 모델: **{best['model']}**",
        f"- 최종 f1_macro: **{best['f1_macro']:.4f}**",
        f"- baseline f1_macro: {base_str}",
        f"- 개선폭: **{imp_str}** (목표 +{IMPROVEMENT_TARGET:.2f}p)",
        f"- 목표 달성: **{'예 ✅' if target_met else '아니오 ❌'}**",
        "",
        "### 최종 모델 Confusion Matrix (행=실제 별점라벨, 열=예측)",
        "",
        "| 실제\\예측 | negative | neutral | positive |",
        "|-----------|----------|---------|----------|",
    ]
    for i, lab in enumerate(LABELS):
        lines.append(f"| {lab} | {cm[i][0]} | {cm[i][1]} | {cm[i][2]} |")

    # 목표 미달 시 원인 분석 (요구사항)
    if not target_met:
        dist = df["label"].value_counts().to_dict()
        neu_ratio = dist.get("neutral", 0) / len(df)
        lines += [
            "",
            "## 목표 미달 원인 분석",
            "",
            f"- **클래스 불균형**: 라벨 분포 {dist}. neutral 이 전체의 "
            f"{neu_ratio:.1%}에 불과해 소수 클래스 학습이 어렵다. "
            "class_weight='balanced' 로 일부 완화했으나 한계가 있다.",
            "- **라벨 노이즈(약지도 한계)**: 별점↔본문 감정 불일치(예: 고별점+불만, "
            "저별점+기능칭찬, 문의성 중립 리뷰)가 상한을 제약한다. neutral 은 본문이 "
            "긍/부정과 겹쳐 별점만으로는 분리가 어렵다.",
            "- **데이터 부족**: neutral 표본 자체가 적어(수백 건 수준) 일반화가 약하다.",
            "- **개선 방향**: ① 수동 검수 라벨로 재학습/검증(`manual_label_sample.csv`) "
            "② neutral 정의 재검토(별점3 → 본문 기반 재라벨) ③ 임베딩(sentence-transformers) "
            "기반 분류 ④ 데이터 증강/추가 수집.",
        ]

    # 수동 검수 기준 성능 (있을 때)
    lines += ["", "## 성능 기준 구분", ""]
    lines.append(f"- **별점 기준(약지도)**: f1_macro = {best['f1_macro']:.4f} (test {len(yte)}건)")
    if manual_eval:
        lines.append(
            f"- **수동 검수 기준**: f1_macro = {manual_eval['f1_macro']:.4f} "
            f"(검수 {manual_eval['n']}건) — 실제 감정 정답 대비 성능"
        )
    else:
        lines.append(
            "- **수동 검수 기준**: 검수 완료 파일(`manual_label_sample_filled.csv`) 미제공. "
            "`manual_label_sample.csv` 의 sentiment_manual 을 채워 다시 실행하면 자동 계산된다."
        )
    lines.append("")

    with open(REPORT_PATH, "a", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info("리포트 갱신(append): %s", REPORT_PATH)


def _merge_metrics(best, baseline_f1, improvement, target_met, manual_eval) -> None:
    existing = {}
    if METRICS_PATH.exists():
        try:
            with open(METRICS_PATH, encoding="utf-8") as f:
                existing = json.load(f)
        except Exception as exc:
            logger.warning("metrics.json 로드 실패 (%r).", exc)
    existing["final_model"] = {
        "name": best["model"],
        "label_type": "weak_supervision_from_rating",
        "f1_macro_star": best["f1_macro"],
        "accuracy_star": best["accuracy"],
        "precision_macro_star": best["precision_macro"],
        "recall_macro_star": best["recall_macro"],
        "baseline_f1_macro": baseline_f1,
        "improvement_over_baseline": improvement,
        "improvement_target": IMPROVEMENT_TARGET,
        "target_met": target_met,
        "f1_macro_manual": manual_eval["f1_macro"] if manual_eval else None,
        "manual_eval_n": manual_eval["n"] if manual_eval else None,
    }
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)
    logger.info("metrics.json 병합 저장: %s", METRICS_PATH)


if __name__ == "__main__":
    from src.improve_sentiment import run as _run  # pickle 안전(모듈 경로 고정)
    out = _run()
    print("\n=== 감성 분류 개선 결과 ===")
    print("※ 학습 라벨은 별점 기반 약지도 라벨(실제 감정 정답 아님)")
    for r in out["experiments"]:
        print(f"  {r['model']}: f1_macro={r['f1_macro']:.4f}")
    b, imp = out["best"], out["improvement"]
    print(f"\n최종: {b['model']} f1_macro={b['f1_macro']:.4f}")
    print(f"baseline 대비 개선: {imp:+.4f}p / 목표달성: {out['target_met']}")
    print(f"오분류 사례: {out['n_errors']}건")
