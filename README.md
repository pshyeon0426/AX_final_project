# 슈퍼SOL 고객 반응 분석 · 경쟁 벤치마킹 · 리뷰 답글 초안 생성

신한은행 AX 전문가 Intensive 과정 최종 프로젝트 (4조 · 댓글부대)

슈퍼SOL 앱 출시 초기의 **고객 반응(앱 리뷰)** 을 수집·분석하고, **경쟁 금융 앱**과
비교 벤치마킹하며, **리뷰 답글 초안을 생성**하는 데이터 분석 서비스입니다.

3개의 축으로 구성됩니다.

1. **고객 반응(VOC) 분석** — 스토어 리뷰 수집 → 감성/토픽/이슈 분석으로 초기 반응 진단
2. **경쟁 앱 벤치마킹** — 경쟁 금융 앱 리뷰와 비교하여 강·약점·개선 기회 도출
3. **리뷰 답글 초안 생성** — LLM 기반 답글 **초안**(담당자 검수용, 자동 게시 아님)

> ⚠️ **보안 원칙**
> - 실제 API 키·고객정보·내부 비공개 데이터는 코드/문서/제출물에 **절대 하드코딩하지 않습니다.**
> - 모든 비밀 값은 `.env` 로만 주입하며, `.env` 는 git에 커밋하지 않습니다 (`.env.example` 만 공유).
> - 답글은 **자동 게시가 아니라 담당자 검수 후 활용하는 초안**으로만 제공합니다.

---

## 1. 요구 환경

- Python **3.10 이상**
- 한국어 형태소 분석은 `kiwipiepy` 권장. 설치/로딩 실패 시 **공백 기반 토큰화로 자동 fallback**
  (`src/tokenizer.py`, 환경변수 `TOKENIZER=kiwi|whitespace`).
- App Store 리뷰는 별도 라이브러리 없이 **Apple 공개 iTunes RSS 피드**로 수집합니다.
  (`app-store-scraper` 는 Python 3.13 비호환이라 사용하지 않습니다.)

## 2. 폴더 구조

```
team_project/
├── app.py                      # Streamlit 데모 대시보드 (진입점)
├── data/
│   ├── raw/                    # 수집 원본 review_raw.csv (git 제외)
│   └── processed/              # 정제 결과 review_clean.csv (git 제외)
├── src/
│   ├── config.py               # 중앙 설정(.env 로딩, 경로)
│   ├── theme.py                # 공용 디자인 팔레트/차트 테마(단일 진실원본)
│   ├── tokenizer.py            # 한국어 토크나이저 (kiwi→whitespace fallback)
│   ├── source_manifest.py      # 데이터 출처 매니페스트
│   ├── app_registry.py         # 수집 대상 앱 레지스트리 + 스토어 수집
│   ├── collect_reviews.py      # Google Play / App Store(RSS) 수집
│   ├── preprocess.py           # 정제·PII 마스킹·품질 요약
│   ├── validation.py           # 데이터 품질 자동 검증
│   ├── train_sentiment.py      # 감성 baseline (Dummy / TF-IDF+LogReg)
│   ├── improve_sentiment.py    # 감성 개선 모델 + 수동검수 샘플
│   ├── issue_classifier.py     # VOC 이슈 유형/심각도 (룰 기반)
│   ├── topic_modeling.py       # TF-IDF + KMeans 토픽
│   ├── benchmark.py            # 경쟁 앱 벤치마킹
│   ├── report_generator.py     # AI 요약 리포트 (LLM/룰 fallback)
│   ├── reply_generator.py      # 리뷰 답글 초안 생성
│   ├── reply_eval.py           # 답글 품질/안전성 평가
│   └── pipeline.py             # 전체 파이프라인 오케스트레이션
├── config/
│   ├── apps.yaml               # 수집 대상 앱(이름·구글/앱스토어 ID)
│   ├── issue_keywords.yaml      # 이슈 유형 키워드 사전
│   └── reply_policy.yaml        # 답글 정책(유형·금지원칙·고객센터 문구)
├── prompts/
│   ├── reply_prompt.md         # 답글 생성 프롬프트
│   └── report_prompt.md        # 리포트 생성 프롬프트
├── docs/
│   ├── PRD.md                  # 제품 요구사항(as-built)
│   ├── design_guide.md         # 색·스타일 가이드
│   ├── qa_scenario.md          # 품질 검토 시나리오
│   └── competitors/            # 경쟁 앱 공개 문서
├── models/                     # 학습 모델 (git 제외)
├── notebooks/                  # EDA 노트북
├── outputs/                    # 리포트·지표·결과 파일
├── tests/                      # pytest
├── .streamlit/config.toml      # Streamlit 테마(브랜드 블루)
├── requirements.txt
├── .env.example
└── README.md
```

## 3. 설치

```bash
python -m venv .venv
# Windows (PowerShell): .venv\Scripts\Activate.ps1
# macOS / Linux       : source .venv/bin/activate
pip install -r requirements.txt
```

## 4. API 키 설정

```bash
# Windows (PowerShell): Copy-Item .env.example .env
# macOS / Linux       : cp .env.example .env
```

`.env` 에 실제 값 입력:

| 변수 | 설명 |
|------|------|
| `OPENAI_API_KEY` | OpenAI API 키 (답글/AI 리포트) |
| `OPENAI_MODEL`   | 모델명 (예: `gpt-4o-mini`, 교체 가능) |
| `GOOGLE_PLAY_APP_ID` | 슈퍼SOL Google Play 패키지명 |
| `APP_STORE_APP_ID` / `APP_STORE_COUNTRY` | App Store 앱 ID / 국가 |
| `TOKENIZER` | `kiwi` 또는 `whitespace` |

> `OPENAI_API_KEY`/`OPENAI_MODEL` 미설정 시 답글·리포트는 **룰 기반 fallback** 으로 동작합니다.
> ⚠️ 키는 절대 코드/채팅/문서에 노출하지 말고 `.env` 에만 두세요. 노출된 키는 즉시 폐기·재발급.

## 5. 실행

### 5-1. 데모 대시보드 (권장)
```bash
streamlit run app.py
```
- 사이드바 **`1·데이터 소스 → 🗂️ 리뷰 수집`** 에서 앱별 스토어 수집(미리보기 후 적용) / 수동 입력
- **`2·분석 필터`** 기준 앱·경쟁 앱·기간 (기간 종료 기본=오늘)
- **`3·실행`** 전체 파이프라인 실행 / 새로고침
- 탭: 🤖 AI 리포트 · 🏆 경쟁 벤치마킹 · 🧩 토픽 분석 · 💬 답글 생성 · ✅ 검증 · 📄 원본 리뷰

### 5-2. 단계별 / 일괄 실행 (CLI)
```bash
python -m src.config              # 설정 점검
python -m src.collect_reviews     # 수집 → data/raw/review_raw.csv
python -m src.preprocess          # 정제 → data/processed/review_clean.csv
python -m src.validation          # 데이터 품질 검증
python -m src.train_sentiment     # 감성 baseline
python -m src.improve_sentiment   # 감성 개선 모델
python -m src.issue_classifier    # VOC 이슈/심각도 (룰 기반)
python -m src.topic_modeling --k 10   # 토픽
python -m src.benchmark           # 경쟁 벤치마킹
python -m src.report_generator    # AI 요약 리포트
python -m src.reply_generator     # 답글 초안 배치
python -m src.reply_eval          # 답글 품질 평가
python -m src.pipeline            # 위 전 과정 일괄 + 성공기준 판정
```

> ⚠️ **분석 단계 구분**: 감성 분류는 **ML**(TF-IDF+LogReg), VOC 이슈 유형 분류는 **룰(키워드 사전)** 기반입니다.
> 이슈 키워드는 `config/issue_keywords.yaml` 로 분리되어 코드 수정 없이 갱신 가능합니다.

## 6. 테스트
```bash
pytest
```

## 7. 개발 규칙
- 비밀 값은 `.env` 에만, 코드에서는 `src.config.settings` 로 접근.
- 원본 데이터(`data/raw`)는 수정하지 않고, 가공 결과는 `data/processed` 에 저장.
- 색·스타일은 `src/theme.py` 의 `PALETTE` 단일 진실원본을 사용(→ `docs/design_guide.md`).
- 새 기능에는 대응 테스트를 `tests/` 에 추가.

## 8. 제출물 체크리스트 (공지문 제출물 5종)

- [x] **① 문제 정의서** — `docs/PRD.md`
- [ ] **② 분석 노트북** — 수집/전처리/감성·토픽 분석 과정 (`notebooks/`)
- [x] **③ 코드** — 수집·분석·답글·대시보드 (`src/`, `app.py`, `tests/`)
- [x] **④ 데모 앱(Streamlit)** — `app.py`
- [ ] **⑤ 발표자료** — 최종 결과·인사이트 (색/스타일은 `docs/design_guide.md` 참고)

## 9. 리뷰 답글 생성 정책

> ⚠️ **자동 게시하지 않습니다.** 생성 답글은 **담당자 검수 후 복사·활용하는 초안**입니다.
> 정책은 `config/reply_policy.yaml`, 프롬프트는 `prompts/reply_prompt.md` 에서 관리(코드 수정 없이 갱신).

- **답글 유형(6종)**: `thanks_positive` · `apology_and_guidance` · `feature_request_ack` ·
  `how_to_guidance` · `security_sensitive` · `fallback_general`
- **형식**: 한국어, 공손한 존댓말, 2~4문장
- **금지 원칙**: ① 개인정보·계좌·비번·인증번호 입력 요청 금지 ② 보상·원인·결과 확정 약속 금지
  ③ 고객별 거래내역 아는 듯한 표현 금지 ④ 투자 조언·확정 추천 금지 ⑤ 경쟁사 비방 금지
- **검수 필수(`needs_human_review=True`)**: 심각도 `high` 또는 보안/송금/오류 등 민감 이슈
- **운영자 지침**: 대시보드 답글 탭의 🧭 입력칸(또는 `reply_policy.yaml`의 `default_llm_guidance`)에
  적은 내용이 프롬프트에 **우선 반영**됩니다(단, 금지 원칙과 충돌 시 금지 원칙 우선).
  모델은 번호 등 사실을 지어내지 않으므로 **실제 값**을 입력해야 합니다.
- 생성 후 `safety_check_reply` 가 금칙어·민감정보 요청·확정 약속·투자 조언을 재검증합니다.

## 10. 데이터 수집 원칙

- **직접 수집** 근거를 `outputs/source_manifest.csv` 에 출처·방법·약관 메모로 기록.
- **공개 출처 + 검증용 샘플만** 사용. 실제 고객정보·내부정보·키 미포함, 작성자 비식별화.
- **약관 준수/fallback**: 스크래핑 제한 시 무리한 우회 대신 수동 CSV 등으로 대체하고
  `terms_note` 에 `manual csv fallback` 등으로 명시.
- App Store 는 공개 iTunes RSS(앱당 ~100~500건 상한). 부족하면 수동 CSV 보강.

매니페스트 컬럼: `source_name, source_type, app_name, url_or_file, collection_method, collected_at, terms_note, status`
- `source_type`: review | app_description | release_note | news | sample
- `collection_method`: google_play_scraper | app_store_manual_csv | manual_download | sample_generated

## 11. 산출물(outputs/)

| 파일 | 내용 |
|------|------|
| `source_manifest.csv` | 데이터 출처 근거 |
| `data_quality_summary.json` | 정제 품질 지표 |
| `data_validation_report.md` | 데이터 검증 리포트 |
| `model_comparison.csv` / `model_report.md` | 감성 모델 비교/리포트 |
| `manual_label_sample.csv` / `error_cases.csv` | 수동 검수 샘플 / 오분류 |
| `topic_summary.csv` / `topic_validation_sample.md` | 토픽 결과/검증 |
| `benchmark_summary.csv` / `competitor_gap_analysis.md` | 벤치마킹 |
| `ai_report.md` | AI 요약 리포트 |
| `reply_drafts.csv` / `reply_eval_report.md` / `reply_risk_cases.csv` | 답글 초안/품질/리스크 |
| `metrics.json` / `performance_log.json` | 통합 지표 / 성능·시간 로그 |

## 12. 샘플 데이터 설명
- 실제 수집(Google Play / App Store RSS)이 **0건이거나 네트워크가 없을 때**도 데모가
  되도록 `generate_sample_reviews()` 가 **합성 리뷰**를 만듭니다.
- 11개 이슈 카테고리(로그인/인증/속도/혜택/송금/투자/UX/오류/보안/문의/긍정)를 포함해
  1,000건 이상 생성 가능하며, `store="sample"`, `source_url="sample_generated"` 로 표기되어
  **실데이터와 명확히 구분**됩니다(매니페스트에도 "합성 데이터, 실제 고객정보 아님" 기록).

## 13. ML / RAG / LLM / 답글 생성 적용 위치
| 기법 | 위치 | 내용 |
|------|------|------|
| **ML(지도)** | `train_sentiment` · `improve_sentiment` | 감성 분류 — TF-IDF + LogReg/LinearSVC, 별점 약지도 라벨 |
| **ML(비지도)** | `topic_modeling` | 토픽 — TF-IDF + KMeans |
| **룰 기반** | `issue_classifier` | 이슈 유형/심각도 — 키워드 사전(ML 아님) |
| **RAG(경량)** | `report_generator` | 근거 리뷰 Top-3 를 리포트 프롬프트에 주입(근거 기반 생성). 벡터 인덱스(ChromaDB)는 향후 |
| **LLM** | `report_generator` · `reply_generator` | OpenAI 호출, 키 없으면 룰 fallback |
| **답글 생성** | `reply_generator` (+`reply_policy.yaml`/`reply_prompt.md`) | 유형 선택 → 프롬프트 → 안전성 재검증 |

## 14. 검증 결과 요약 (수집 시점 기준 스냅샷)
- **데이터 품질**: 원본 약 3,400 → 정제 약 2,976건(6개 앱). 결측률 0% · 날짜 파싱 100% · 중복 0%(정제 후) · rating 1~5 · 총 ≥1,000 → **전 항목 PASS**.
- **감성 모델(별점 약지도 기준)**: Dummy 0.23 → baseline(TF-IDF+LogReg) **0.547** → 개선 **0.576** (macro F1).
- **토픽**: **10개**(최대 토픽 비율 ~35%, 짧은 리뷰 필터로 쏠림 완화).
- **답글 품질/안전성**: 안전성 위반 **0건**, 자동 품질 평균 **~2.9/3.0**.
- **파이프라인 종합 판정**: **PASS** (전체 ≈ 2~3분).

## 15. 실패 / 오분류 / 부적절 답글 케이스
- **감성 오분류**(`outputs/error_cases.csv`): 별점↔본문 불일치가 주원인. 예) "★5인데 본문은 불만",
  "★1인데 기능 칭찬". neutral(별점3)이 본문상 긍/부정과 겹쳐 분리가 어려움 → **약지도 라벨의 구조적 한계**.
- **이슈 오탐**(룰 기반): 키워드 다의어로 오탐 가능 → `config/issue_keywords.yaml` 보정.
- **부적절 답글 후보**(`outputs/reply_risk_cases.csv`): 금칙어·민감정보 요청·확정 약속·투자 조언을
  `safety_check_reply` 가 탐지해 분리(현재 0건). 심각/보안/금전 이슈는 `needs_human_review=True` 로 검수 강제.

## 16. 한계 및 고도화 방향
- 감성 라벨이 별점 기반 **약지도** → 수동 검수 라벨 확보 시 신뢰도↑(데모상 수동 기준 F1↑ 확인).
- App Store RSS 수집량 상한 + 일부 앱 데이터 시점 편중(특정 월 쏠림).
- 토픽/상세 벤치마크는 파이프라인 산출물(전체 데이터) 기준 — 실시간 기간 재계산 미적용.
- 고도화: **벡터 RAG(ChromaDB) 근거 강화**, 임베딩 기반 감성 모델, 정기 자동 수집, 답글 검수 워크플로우 연동.

## 17. 라이선스 / 주의
- 스토어 이용약관·로봇 배제 정책 준수. 분석 결과는 내부 의사결정 참고용.
- 개인정보 비식별화. 생성 답글은 **초안**이며 담당자 검수 후에만 활용.
