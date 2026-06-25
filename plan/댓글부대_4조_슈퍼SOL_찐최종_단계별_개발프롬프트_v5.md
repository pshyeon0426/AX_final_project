# 단계별 개발 프롬프트 찐최종본 v5

# 슈퍼SOL 앱 출시 초기 고객 반응·경쟁 앱 벤치마킹·리뷰 답글 자동 생성 서비스

- 과정: 신한은행 AX 전문가 Intensive 과정 최종 프로젝트
- 팀명 / 조: 댓글부대 / 4조
- 팀원: 허재호 프로, 박성현 프로, 배미혜 프로, 김재훈 프로
- 작성일: 2026. 6. 20.
- 문서 목적: 공지문 필수 요건과 추가 요건인 OpenAI API 기반 리뷰 답글 자동 생성기를 포함한 최종 개발 지시서

---

## 0. 사용 원칙

- 아래 프롬프트는 순서대로 사용한다.
- 각 단계 결과물을 파일로 저장한 뒤 다음 단계에 넘긴다.
- 실제 고객정보와 내부 비공개 데이터는 사용하지 않는다.
- 공개 리뷰, 공개 문서 또는 샘플 CSV만 사용한다.
- 스크래핑이 막히면 수동 다운로드/샘플 CSV 방식으로 전환한다.
- App Store는 수동 CSV 또는 샘플 CSV fallback을 기본으로 둔다.
- OpenAI API 키는 `OPENAI_API_KEY` 환경변수 또는 `.env`로만 사용하고, 실제 키는 코드/README/제출 zip에 포함하지 않는다.
- 답글 생성 기능은 실제 앱스토어 자동 게시가 아니라 담당자 검수용 답글 초안 생성으로 제한한다.
- 각 단계 코드에는 실행 방법, 예외 처리, 로그, 테스트 가능 구조를 포함한다.
- 최종 발표에서는 정상 케이스뿐 아니라 실패/오분류/부적절 답글 케이스 3개 이상을 반드시 보여준다.
- 30초 성능 기준은 사전 학습 모델과 사전 생성 RAG 인덱스가 준비된 상태에서 CSV 업로드 후 정제/분석/리포트 생성까지의 사용자 체감 시간을 기준으로 한다. 모델 재학습과 최초 RAG 인덱싱 시간은 별도 측정한다.

---

## 1단계. 프로젝트 뼈대 생성

```text
역할: 너는 Python 데이터 분석 프로젝트의 시니어 개발자다.
목표: MVP 폴더 구조와 기본 파일을 생성한다.

요구사항:
- Python 3.10 이상 기준으로 작성한다.
- 폴더는 data/raw, data/processed, docs/competitors, models, notebooks, src, tests, outputs, prompts, config, chroma_db 로 구성한다.
- requirements.txt에는 pandas, numpy, scikit-learn, streamlit, plotly, beautifulsoup4, google-play-scraper, app-store-scraper, kiwipiepy, sentence-transformers, langchain, chromadb, python-dotenv, pytest, pyyaml, joblib, openai를 포함한다.
- .gitignore에는 .env, __pycache__, .pytest_cache, chroma_db 임시 파일, 대용량 캐시를 포함한다.
- .env.example에는 OPENAI_API_KEY=your_key_here, OPENAI_MODEL=your_model_here 예시를 포함한다.
- kiwipiepy 설치 실패 시 공백 기반 토큰화 fallback을 제공할 수 있게 설계한다.
- README.md 초안을 작성하고 설치/실행/테스트/API 키 설정 순서를 명시한다.
- 공지문 제출물 5종을 README 체크리스트에 포함한다.
- 실제 API 키나 내부정보는 절대 하드코딩하지 않는다.

산출물:
- 폴더 구조
- requirements.txt
- README.md 초안
- .env.example
- .gitignore
- src/config.py
```

---

## 2단계. 데이터 수집 계획 및 출처 관리 파일 생성

```text
역할: 너는 금융 데이터 수집 계획 담당자다.
목표: 공개 리뷰와 경쟁 앱 공개 문서를 직접 수집했다는 근거를 남길 수 있는 source_manifest 구조를 만든다.

요구사항:
- outputs/source_manifest.csv를 생성하는 함수를 작성한다.
- 필수 컬럼은 source_name, source_type, app_name, url_or_file, collection_method, collected_at, terms_note, status 로 한다.
- source_type은 review, app_description, release_note, news, sample 중 하나로 한다.
- collection_method는 google_play_scraper, app_store_manual_csv, manual_download, sample_generated 중 하나로 한다.
- 스크래핑 제한 또는 약관 이슈가 있으면 terms_note에 'manual csv fallback' 등으로 기록한다.
- README에 데이터 직접 수집 원칙과 실제 고객정보/내부정보 미사용 원칙을 적는다.

산출물:
- src/source_manifest.py
- outputs/source_manifest.csv
- README 데이터 수집 원칙 섹션
```

---

## 3단계. 샘플 리뷰 데이터 생성/수집 모듈

```text
역할: 너는 공개 앱 리뷰 데이터 수집 담당자다.
목표: 실제 수집이 실패해도 데모 가능한 샘플 리뷰 CSV와 선택형 수집 모듈을 만든다.

요구사항:
- CSV 필수 컬럼은 app_name, store, rating, date, review_text, version, source_url 로 한다.
- 우리 앱은 '슈퍼SOL', 경쟁 앱은 3~5개로 설정 가능하게 한다.
- Google Play는 google-play-scraper 기반 collect_google_play_reviews(app_id, app_name, n)를 작성한다.
- App Store는 자동 수집 실패 가능성이 있으므로 collect_app_store_reviews는 선택 기능으로 작성하고, 실패 시 수동 CSV 또는 샘플 CSV로 전환한다.
- store 컬럼에는 google_play, app_store, sample 중 하나를 기록한다.
- 수집 실패 시 경고 로그를 출력하고 generate_sample_reviews()를 실행한다.
- 샘플 데이터는 로그인, 인증, 속도, 혜택, 송금, 투자, UX, 오류, 보안 우려, 문의성 리뷰, 긍정 리뷰 등 다양한 이슈를 포함해 1,000건 이상 생성 가능해야 한다.
- source_manifest.csv에도 수집/생성 결과를 기록한다.
- 결과는 data/raw/review_raw.csv로 저장한다.

산출물:
- src/collect_reviews.py
- data/raw/review_raw.csv
- outputs/source_manifest.csv 업데이트
```

---

## 4단계. 데이터 정제 및 개인정보성 패턴 마스킹

```text
역할: 너는 금융권 데이터 전처리 담당자다.
목표: 공개 리뷰 데이터를 분석 가능한 형태로 정제하고 개인정보성 패턴을 마스킹한다.

요구사항:
- 입력은 data/raw/review_raw.csv, 출력은 data/processed/review_clean.csv로 한다.
- 필수 컬럼 누락 여부를 검사한다.
- review_text 결측, 중복, 너무 짧은 리뷰를 처리한다.
- 전화번호, 이메일, 계좌번호, 생년월일, 인증번호처럼 보이는 패턴은 [MASKED]로 치환한다.
- date를 datetime으로 표준화하고 실패 건수는 로그로 남긴다.
- clean_text 컬럼을 생성하고 이모지/특수문자/반복 공백을 정리한다.
- 리뷰 길이 분포를 계산하고 IQR 또는 분위수 기준으로 너무 짧거나 긴 리뷰를 이상치 후보로 표시한다.
- 답글 자동 생성 대상에서 [MASKED]가 포함된 리뷰는 needs_human_review=True 후보로 표시한다.
- 정제 전후 건수, 결측률, 중복률, 날짜 파싱 실패율, 이상치 후보 수를 outputs/data_quality_summary.json에 저장한다.

산출물:
- src/preprocess.py
- data/processed/review_clean.csv
- outputs/data_quality_summary.json
```

---

## 5단계. 데이터 품질 검증 스크립트

```text
역할: 너는 QA 엔지니어다.
목표: 데이터 품질 기준을 자동 검증하는 스크립트와 pytest 테스트를 작성한다.

검증 기준:
- 필수 컬럼 100% 존재
- 리뷰 본문 결측률 5% 이하
- 날짜 파싱 성공률 95% 이상
- 중복률 3% 이하
- rating은 1~5 범위
- 총 리뷰 수 1,000건 이상
- 슈퍼SOL + 경쟁 앱 3개 이상 포함

요구사항:
- validate_review_data(df) 함수를 작성해 pass/fail과 상세 사유를 반환한다.
- app_name별 리뷰 수를 출력한다.
- tests/test_data_validation.py에 정상/실패 케이스를 작성한다.
- 검증 결과를 outputs/data_validation_report.md로 저장한다.
- outputs/metrics.json에 total_reviews, app_count, data_quality_pass를 병합 저장한다.

산출물:
- src/validation.py
- tests/test_data_validation.py
- outputs/data_validation_report.md
- outputs/metrics.json
```

---

## 6단계. Pandas EDA 및 전처리 판단 근거 노트북

```text
역할: 너는 앱 VOC 분석가다.
목표: 공지문 제출물인 분석 노트북을 만들고, EDA + 전처리 판단 근거를 주석과 마크다운으로 설명한다.

요구사항:
- notebooks/01_EDA_preprocessing.ipynb를 작성한다.
- 전체 리뷰 수, 앱별 리뷰 수, 평균 평점, 부정 리뷰 비율, 최근 7일/30일 변화량을 계산한다.
- 기간별 리뷰 수, 평균 평점, 부정률 추이를 시각화한다.
- 리뷰 길이, 중복, 결측, 날짜 파싱 실패, rating 이상값 등 데이터 품질 이슈를 분석한다.
- 결측·이상치·중복을 어떻게 처리했는지 마크다운으로 근거를 설명한다.
- 앱별 부정 리뷰 TOP 키워드를 계산한다.
- 답글 생성 우선순위 후보를 선별한다. 예: 부정 리뷰, high severity, 인증/오류/보안 이슈.
- plotly 또는 matplotlib 시각화를 생성한다.
- outputs/eda_summary.json과 outputs/eda_charts.html을 생성한다.
- 데이터의 한계와 편향을 노트북 마지막에 정리한다.

산출물:
- notebooks/01_EDA_preprocessing.ipynb
- src/eda.py
- outputs/eda_summary.json
- outputs/eda_charts.html
```

---

## 7단계. 감성 라벨링과 ML 베이스라인 구축

```text
역할: 너는 머신러닝 엔지니어다.
목표: 리뷰의 긍정/중립/부정을 분류하는 baseline 모델을 만든다.

요구사항:
- 별점 기반 약지도 라벨을 생성한다: 1~2점=negative, 3점=neutral, 4~5점=positive.
- 별점 기반 라벨은 실제 감정 정답이 아니라 약지도 라벨임을 코드 주석과 리포트에 명시한다.
- Baseline 1: DummyClassifier 또는 최빈 클래스 예측 모델을 만든다.
- Baseline 2: TF-IDF unigram + LogisticRegression 기본값 모델을 만든다.
- train/test split 후 accuracy, precision_macro, recall_macro, f1_macro, confusion matrix를 계산한다.
- 결과를 outputs/model_comparison.csv와 outputs/model_report.md에 저장한다.
- outputs/metrics.json에 baseline 성능을 병합 저장한다.

산출물:
- src/train_sentiment.py
- outputs/model_comparison.csv
- outputs/model_report.md
- outputs/metrics.json
```

---

## 8단계. 감성분석 개선 모델 및 수동 검수 샘플

```text
역할: 너는 데이터 사이언스 모델 개선 담당자다.
목표: 베이스라인 대비 개선 모델을 만들고, 라벨 품질 한계를 보완하기 위한 수동 검수 샘플을 생성한다.

요구사항:
- 개선 모델은 TF-IDF ngram + LogisticRegression 또는 LinearSVC를 사용한다.
- class_weight='balanced', ngram_range, max_features 등 최소 2개 이상의 개선 실험을 수행한다.
- baseline 대비 macro F1-score +0.05p 이상 개선을 목표로 한다.
- 미달 시 class imbalance, 라벨 노이즈, 데이터 부족 등 원인을 model_report.md에 작성한다.
- 최종 모델은 models/sentiment_model.pkl로 저장한다.
- 리뷰 샘플 100~200건을 outputs/manual_label_sample.csv로 추출하고 sentiment_manual, issue_manual 컬럼을 비워둔다.
- 수동 검수 라벨이 입력된 파일이 있으면 별점 기반 성능과 수동 검수 샘플 기준 성능을 구분해 계산한다.
- 오분류 사례 3개 이상을 outputs/error_cases.csv에 저장한다.
- outputs/metrics.json에 final_model 성능과 baseline 대비 개선폭을 병합 저장한다.

산출물:
- models/sentiment_model.pkl
- outputs/model_report.md 업데이트
- outputs/manual_label_sample.csv
- outputs/error_cases.csv
- outputs/metrics.json 업데이트
```

---

## 9단계. 불만 유형 분류기 및 심각도 분류

```text
역할: 너는 금융 앱 VOC 카테고리 설계자다.
목표: 리뷰를 로그인, 인증, 속도, 오류, 혜택, 송금, 투자, UX, 보안, 기타 유형으로 분류하고 답글 우선순위용 심각도를 부여한다.

요구사항:
- MVP에서는 키워드 사전 기반 classify_issue_type(text) 함수를 만든다.
- 이 단계는 ML이 아니라 룰 기반 분류임을 README와 발표 자료에 명시한다.
- 한 리뷰가 여러 유형에 해당할 수 있으므로 list 형태로 반환한다.
- 유형별 키워드는 config/issue_keywords.yaml로 분리한다.
- classify_severity(text, rating, issue_types) 함수를 작성한다.
- severity는 high, medium, low, positive 중 하나로 한다.
- high 예시는 로그인 불가, 이체/결제 실패, 보안 우려, 오류 반복 등이다.
- 앱별/기간별 불만 유형 TOP N을 집계한다.
- 샘플 100건에 대해 사람이 검수할 수 있는 outputs/issue_review_sample.csv를 생성한다.
- tests/test_issue_classifier.py에 대표 문장 테스트를 작성한다.

산출물:
- src/issue_classifier.py
- config/issue_keywords.yaml
- outputs/issue_review_sample.csv
- tests/test_issue_classifier.py
```

---

## 10단계. 토픽/키워드 분석

```text
역할: 너는 한국어 텍스트 마이닝 분석가다.
목표: 앱별 주요 토픽 5개 이상을 도출한다.

요구사항:
- clean_text를 기반으로 불용어 제거와 토큰화를 수행한다.
- kiwipiepy를 기본으로 쓰고 실패 시 공백 기반 토큰화 fallback을 제공한다.
- TF-IDF 기반 키워드 추출과 KMeans 기반 토픽 군집화를 구현한다.
- 각 토픽별 대표 키워드 10개와 대표 리뷰 5개를 저장한다.
- 토픽 수는 기본 5개로 하되 파라미터로 조정 가능하게 한다.
- 토픽 품질 검증을 위해 outputs/topic_validation_sample.md를 생성한다.
- 토픽명이 애매하면 사람이 수정할 수 있도록 topic_label 컬럼을 분리한다.
- outputs/metrics.json에 topic_count, topic_review_coverage를 병합 저장한다.

산출물:
- src/topic_modeling.py
- outputs/topic_summary.csv
- outputs/topic_validation_sample.md
- outputs/metrics.json 업데이트
```

---

## 11단계. 경쟁 앱 문서 수집/RAG 인덱싱

```text
역할: 너는 RAG 파이프라인 개발자다.
목표: 경쟁 앱 기능 설명, 업데이트 내역, 공개 기사/공지 문서를 근거 검색용으로 인덱싱한다.

요구사항:
- docs/competitors 폴더의 txt/md/pdf 문서를 로드한다.
- 문서 메타데이터는 app_name, doc_type, source, date를 포함한다.
- RecursiveCharacterTextSplitter로 chunk_size 500~800, overlap 80~100 기준 청크를 만든다.
- sentence-transformers 기반 임베딩 또는 간단한 TF-IDF 검색 대체 모드를 제공한다.
- ChromaDB 로컬 인덱스를 chroma_db에 저장한다.
- search_evidence(query, top_k=3) 함수를 구현하고 검색 결과에 source와 snippet을 포함한다.
- RAG 검증용 테스트 질문 5개를 만들고 각 질문별 Top-3 문서의 source, snippet, app_name을 outputs/rag_eval_sample.csv에 저장한다.
- rag_eval_sample.csv에는 human_relevance_score 컬럼을 두고 사람이 0/1/2점으로 관련성을 평가할 수 있게 한다.
- 근거 문서에 없는 질문은 추정하지 않고 '공개 근거 문서에서 확인되지 않음'으로 응답하도록 한다.

산출물:
- src/rag_index.py
- src/rag_search.py
- chroma_db
- outputs/rag_eval_sample.csv
```

---

## 12단계. 경쟁 앱 벤치마킹 지표 생성

```text
역할: 너는 경쟁 앱 분석 컨설턴트 겸 데이터 분석가다.
목표: 슈퍼SOL과 경쟁 앱 3~5개의 리뷰/평점/불만 유형/토픽/공개 기능 정보를 비교해 벤치마킹 테이블을 생성한다.

요구사항:
- 앱별 평균 평점, 부정 리뷰 비율, 리뷰 수, 최근 30일 평점 변화량을 비교한다.
- 앱별 주요 불만 유형 TOP5와 주요 토픽 TOP5를 비교한다.
- 경쟁 앱 기능 설명/RAG 문서에서 혜택, 인증, 송금, 투자, UI/UX, 이벤트 관련 키워드를 추출한다.
- 슈퍼SOL 대비 경쟁 앱의 강점, 약점, 개선 기회를 정리한다.
- 근거 없이 경쟁사 내부 전략을 추정하지 않고 공개 리뷰와 공개 문서 기준으로만 작성한다.
- 결과는 outputs/benchmark_summary.csv와 outputs/competitor_gap_analysis.md로 저장한다.
- Streamlit 대시보드와 AI 리포트 입력값으로 사용할 수 있는 dict 형태 결과를 반환한다.

산출물:
- src/benchmark.py
- outputs/benchmark_summary.csv
- outputs/competitor_gap_analysis.md
```

---

## 13단계. AI 요약 리포트 생성

```text
역할: 너는 금융 앱 CX 리포트 작성 AI 프롬프트 엔지니어다.
목표: 분석 지표와 RAG 근거를 바탕으로 근거 기반 요약 리포트를 생성하는 함수를 만든다.

요구사항:
- 입력은 KPI, 감성 분포, 불만 유형 TOP5, 토픽 요약, 경쟁 앱 비교표, RAG 근거 Top-3로 한다.
- 출력 섹션은 핵심 요약, 부정 반응 원인, 개선 우선순위, 경쟁 앱 벤치마킹 포인트, 근거 리뷰/문서, 데이터 한계로 구성한다.
- 프롬프트에는 '근거 없는 추정 금지', '내부 전략 추정 금지', '공개 데이터 기준' 문구를 포함한다.
- LLM API가 없을 경우 rule-based summary fallback을 제공한다.
- 결과는 outputs/ai_report.md로 저장한다.
- LLM 호출 시간과 fallback 사용 여부를 outputs/performance_log.json에 기록한다.

산출물:
- src/report_generator.py
- prompts/report_prompt.md
- outputs/ai_report.md
- outputs/performance_log.json 업데이트
```

---

## 14단계. 리뷰 답글 생성 정책 파일 작성

```text
역할: 너는 금융 앱 고객응대 정책 설계자다.
목표: 실제 리뷰 답글 초안을 안전하게 만들기 위한 정책 파일과 프롬프트를 작성한다.

요구사항:
- config/reply_policy.yaml을 만든다.
- 답글 유형은 thanks_positive, apology_and_guidance, feature_request_ack, how_to_guidance, security_sensitive, fallback_general로 정의한다.
- 금지 원칙을 포함한다.
  1) 개인정보, 계좌번호, 비밀번호, 인증번호 입력 요청 금지
  2) 보상, 장애 원인, 처리 결과 확정 약속 금지
  3) 고객별 계좌/거래 내역을 알고 있는 것처럼 표현 금지
  4) 금융상품 투자 조언 또는 확정적 추천 금지
  5) 경쟁사 비방 금지
- 심각한 오류/보안/금전 관련 이슈는 고객센터 또는 앱 내 문의로 안내하고 needs_human_review=True로 표시한다.
- prompts/reply_prompt.md를 작성한다.
- 프롬프트는 입력 리뷰, 감성, 이슈 유형, 심각도, 앱명, 톤, 금칙어를 받아 JSON 형식으로 답글 초안을 반환하게 설계한다.
- 답글은 한국어, 공손한 톤, 2~4문장 이내를 기본으로 한다.
- 자동 게시가 아닌 담당자 검수용 초안임을 주석과 README에 명시한다.

산출물:
- config/reply_policy.yaml
- prompts/reply_prompt.md
- README 답글 생성 정책 섹션
```

---

## 15단계. OpenAI API 기반 리뷰 답글 생성기 구현

```text
역할: 너는 OpenAI API를 활용하는 Python 백엔드 개발자다.
목표: OPENAI_API_KEY를 이용해 리뷰별 고객 답글 초안을 생성하는 모듈을 만든다.

요구사항:
- src/reply_generator.py를 작성한다.
- python-dotenv로 .env를 로드하되, 실제 키는 코드에 하드코딩하지 않는다.
- 환경변수 OPENAI_API_KEY가 없으면 LLM 호출을 시도하지 않고 rule_based_reply() fallback을 사용한다.
- OPENAI_MODEL 환경변수로 모델명을 설정 가능하게 한다.
- generate_reply(review_text, rating, sentiment, issue_types, severity, app_name) 함수를 작성한다.
- OpenAI API 호출은 별도 함수 call_openai_reply(prompt, model)로 분리한다.
- 결과는 반드시 dict로 반환한다: reply_draft, reply_type, safety_flags, needs_human_review, reason.
- API 오류, 타임아웃, JSON 파싱 실패 시 fallback 답글과 오류 로그를 반환한다.
- safety_check_reply(reply_text) 함수를 작성해 민감정보 요청, 확정 약속, 금융 조언 표현을 탐지한다.
- 부적절 가능성이 있으면 needs_human_review=True로 표시한다.
- 여러 리뷰를 일괄 처리해 outputs/reply_drafts.csv로 저장하는 generate_reply_batch(df, n=None) 함수를 만든다.
- outputs/performance_log.json에 답글 생성 시간, API 사용 여부, fallback 건수를 기록한다.

산출물:
- src/reply_generator.py
- outputs/reply_drafts.csv
- outputs/performance_log.json 업데이트
```

---

## 16단계. 답글 생성 단위 테스트 및 품질 검증

```text
역할: 너는 고객응대 AI 품질 검증 담당자다.
목표: 답글 생성 결과가 안전하고 발표 가능한 수준인지 검증한다.

테스트 케이스:
1. 긍정 리뷰: 감사 답글 생성
2. 로그인 오류 리뷰: 사과 + 고객센터/앱 내 문의 안내
3. 보안 우려 리뷰: 민감정보 요청 금지 + human review 표시
4. 개인정보가 포함된 리뷰: [MASKED] 처리 확인 + human review 표시
5. 기능 개선 요청: 의견 감사 + 개선 참고 답글
6. OpenAI API 키 누락: fallback 답글 정상 생성
7. API 오류/JSON 파싱 실패: 앱 중단 없이 fallback 동작

요구사항:
- tests/test_reply_generator.py를 작성한다.
- 답글 20건 이상을 outputs/reply_eval_sample.csv로 추출한다.
- reply_eval_sample.csv에는 tone_score, safety_score, helpfulness_score, human_comment 컬럼을 둔다.
- outputs/reply_eval_report.md를 생성한다.
- 안전성 위반 후보가 있으면 outputs/reply_risk_cases.csv에 기록한다.
- 성공 기준: 안전성 위반 0건 목표, 품질 평균 3점 만점 2점 이상. 미달 시 원인과 개선 방안을 작성한다.

산출물:
- tests/test_reply_generator.py
- outputs/reply_eval_sample.csv
- outputs/reply_eval_report.md
- outputs/reply_risk_cases.csv
```

---

## 17단계. Streamlit 대시보드 통합

```text
역할: 너는 Streamlit 앱 개발자다.
목표: 데이터 업로드부터 분석 리포트와 리뷰 답글 생성까지 한 화면에서 시연 가능한 앱을 만든다.

화면 구성:
- 사이드바: CSV 업로드, 샘플 데이터 사용, 앱 선택, 기간 선택, 경쟁 앱 선택, OpenAI API 사용 여부 표시
- 메인 KPI 카드: 리뷰 수, 평균 평점, 부정률, 주요 토픽 수, 리포트 생성 시간
- 차트: 리뷰 추이, 감성 분포, 불만 유형 TOP N, 앱별 비교
- 테이블: 부정 리뷰 원문, 토픽별 대표 리뷰, 경쟁 앱 비교표
- AI 리포트 탭: 요약 리포트와 근거 문서 표시
- 답글 생성 탭: 리뷰 선택, 감성/이슈/심각도 표시, 답글 생성, 재생성, 수정, 복사, CSV 다운로드
- 검증 탭: 데이터 검증 결과, 모델 성능, RAG 평가 결과, 답글 품질 평가, 테스트 결과, 실패 케이스

요구사항:
- 사용자에게 30초 기준이 '사전 학습 모델/사전 생성 인덱스 준비 후 분석 요청 기준'임을 안내한다.
- 답글 생성은 자동 게시가 아니라 검수용 초안임을 화면에 명확히 표시한다.
- OPENAI_API_KEY가 없으면 API 미사용 상태와 fallback 사용 안내를 보여준다.
- 네트워크가 없어도 사전 생성된 reply_drafts.csv로 데모 가능하게 한다.

산출물:
- app.py
- Streamlit 실행 명령어
- 화면별 함수 구조
```

---

## 18단계. 통합 파이프라인 작성

```text
역할: 너는 Python 애플리케이션 아키텍트다.
목표: 수집/정제/검증/분석/벤치마킹/리포트/답글 생성을 하나의 파이프라인으로 연결한다.

요구사항:
- run_pipeline(input_csv, config) 함수를 작성한다.
- 단계별 결과를 dict로 반환한다: data_quality, kpi, sentiment, issues, topics, benchmark, rag_evidence, ai_report, reply_drafts, test_summary.
- 각 단계는 실패해도 전체 앱이 중단되지 않도록 예외 처리하고 fallback 메시지를 반환한다.
- 처리 시간을 단계별로 측정해 outputs/performance_log.json에 저장한다.
- Streamlit에서는 캐싱을 적용하되, 데이터 변경 시 갱신되게 한다.
- 전체 파이프라인 종료 시 outputs/metrics.json을 최종 병합 저장한다.
- metrics.json에는 total_reviews, app_count, processing_time_sec, report_generation_time_sec, reply_generation_time_sec, fallback_reply_count, success_criteria_pass_fail을 포함한다.

산출물:
- src/pipeline.py
- outputs/performance_log.json
- outputs/metrics.json
- Streamlit 연동 코드
```

---

## 19단계. 단위 테스트 작성

```text
역할: 너는 테스트 자동화 엔지니어다.
목표: 핵심 함수의 회귀 오류를 막기 위한 pytest 테스트를 작성한다.

테스트 대상:
- 데이터 검증: 필수 컬럼 누락, rating 범위 오류, 날짜 파싱 오류
- 전처리: 개인정보성 패턴 마스킹, 중복 제거
- 감성 라벨링: 별점 라벨 생성 규칙
- 불만 유형/심각도 분류: 대표 문장이 올바른 유형과 심각도로 분류되는지
- RAG 검색: 테스트 문서에서 관련 키워드가 포함된 청크를 반환하는지
- 벤치마킹: benchmark_summary.csv에 앱별 비교 지표가 누락 없이 생성되는지
- 리포트 생성: 필수 섹션이 모두 포함되는지
- 답글 생성: API 키 누락 fallback, 금칙어 탐지, human review 플래그

요구사항:
- tests 폴더에 테스트를 작성한다.
- pytest 실행 결과 예시를 outputs/test_result.txt에 저장한다.

산출물:
- tests/*.py
- outputs/test_result.txt
```

---

## 20단계. 모델/분석/답글 검증 리포트

```text
역할: 너는 데이터 사이언스 및 고객응대 AI 검증 담당자다.
목표: 성공 기준 달성 여부와 실패 원인을 정리하는 검증 리포트를 작성한다.

검증 항목:
- 리뷰 1,000건 이상 여부
- 앱별/기간별 EDA 완료 여부
- 감성분석 F1-score 0.75 이상 여부
- baseline 대비 개선 여부
- 별점 기반 약지도 성능과 수동 검수 샘플 기준 성능 구분 여부
- 불만 유형 분류가 룰 기반이면 F1-score 대상이 아님을 명시했는지 여부
- 주요 토픽 5개 이상 도출 여부
- 경쟁 앱 벤치마킹 테이블과 개선 기회 생성 여부
- RAG 근거 평가 샘플 생성 여부
- 분석 요청 후 요약 리포트 30초 이내 생성 여부
- 답글 20건 이상 생성 및 품질 검증 여부
- 답글 안전성 위반 후보와 대응 여부
- 실패/오분류/부적절 답글 케이스 3개 이상 분석 여부

요구사항:
- metrics.json과 outputs/*.csv를 읽어 outputs/validation_report.md를 생성한다.
- 목표 미달 항목은 '미달 사유'와 '개선 방안'을 반드시 작성한다.
- 답글 관련 검증 결과는 reply_eval_report.md와 연계한다.

산출물:
- src/make_validation_report.py
- outputs/validation_report.md
```

---

## 21단계. 통합 테스트/UAT 시나리오

```text
역할: 너는 발표 전 QA 리더다.
목표: 실제 발표에서 시연할 정상/예외/실패 케이스를 만든다.

시나리오:
1. 정상: 슈퍼SOL과 경쟁 앱 3개, 최근 3개월 리뷰 분석 후 AI 리포트 생성
2. 정상: 부정 리뷰 1건을 선택해 답글 초안 생성 후 복사/수정
3. 예외: CSV 필수 컬럼 누락 시 사용자 안내 메시지 확인
4. 예외: 빈 리뷰 파일 업로드 시 분석 중단 및 원인 안내
5. 모델 실패: 별점은 높지만 본문은 부정적인 리뷰의 오분류 사례 확인
6. RAG 실패: 경쟁 앱 문서에 없는 질문을 했을 때 근거 부족 안내 확인
7. 벤치마킹 실패: 경쟁 앱 문서가 없는 경우 공개 리뷰 기준 비교만 수행하고 문서 근거 부족 표시
8. 답글 실패: OpenAI API 키가 없거나 호출 실패했을 때 fallback 답글 생성 확인
9. 답글 안전성: 개인정보 포함 리뷰 또는 보안 우려 리뷰가 human review로 표시되는지 확인
10. 성능: 사전 학습 모델과 사전 RAG 인덱스 준비 상태에서 1,000건 기준 리포트 30초 이내, 답글 1건 10초 이내 처리 여부 측정

요구사항:
- 각 시나리오의 입력, 예상 결과, 실제 결과, 조치사항을 표로 정리한다.
- outputs/uat_report.md를 작성한다.
- 발표용 데모 체크리스트를 작성한다.

산출물:
- outputs/uat_report.md
- demo_checklist.md
```

---

## 22단계. README, 발표자료, Q&A 준비

```text
역할: 너는 교육 과정 최종 발표 준비 담당자다.
목표: 프로젝트를 10~15분 안에 설명하고 시연할 수 있도록 README와 데모 스크립트를 작성한다.

README 포함 내용:
- 프로젝트 개요와 문제 정의
- 설치 방법과 실행 명령어
- OPENAI_API_KEY 설정 방법과 실제 키 제출 금지 안내
- 폴더 구조
- 데이터 수집 방법과 source_manifest 설명
- 샘플 데이터 설명
- 주요 기능
- ML/RAG/LLM/답글 생성 적용 위치
- 검증 결과 요약
- 실패/오분류/부적절 답글 케이스
- 한계와 고도화 방향

발표자료 구성:
1. 문제 정의와 금융 업무 활용 가능성
2. 데이터 수집 및 EDA/전처리 판단 근거
3. 시스템 아키텍처
4. ML baseline 대비 개선 결과
5. 토픽/불만 유형 분석 결과
6. 경쟁 앱 벤치마킹 결과
7. RAG/LLM 리포트 생성 구조
8. 리뷰 답글 자동 생성 구조와 안전장치
9. Streamlit 라이브 데모
10. 검증/오답/실패 케이스
11. 한계 및 고도화 방향

Q&A 예상 문답:
- 왜 이 주제를 선택했는가?
- 머신러닝은 어디에 썼는가?
- F1-score는 어떤 라벨 기준인가?
- baseline 대비 어떻게 개선했는가?
- RAG가 왜 필요한가?
- 답글 자동 생성은 안전한가?
- OpenAI API 키는 어떻게 관리했는가?
- 실제 고객 응대에 바로 자동 게시해도 되는가?
- 앱스토어 리뷰의 대표성 한계는 무엇인가?
- 30초 성능 기준은 무엇인가?
- 내부 데이터 없이 실무 활용 가능성이 있는가?

산출물:
- README.md
- demo_script.md
- presentation_outline.md 또는 PPT
- qna.md
```

---

## 23단계. 최종 제출 패키징

```text
역할: 너는 최종 제출 담당자다.
목표: 공지문 제출물 5종과 README를 누락 없이 제출 가능한 구조로 묶는다.

요구사항:
- 아래 산출물이 존재하는지 확인한다.
  1) 문제 정의서 1p
  2) 분석 노트북
  3) 모델/시스템 코드
  4) Streamlit 데모 앱
  5) 발표 자료
  6) README.md
  7) 검증 리포트
  8) 답글 생성 관련 정책/프롬프트/검증 결과
- README에 실행 방법을 반드시 적는다.
- 코드 실행에 필요한 샘플 데이터는 data/raw 또는 sample_data 폴더에 포함한다.
- API 키가 필요한 경우 .env.example만 제공하고 실제 키는 포함하지 않는다.
- 최종 zip 파일을 만들되, 불필요한 캐시, 개인정보, 실제 API 키, 대용량 임시 파일은 제외한다.

산출물:
- final_submission.zip
- final_checklist.md
```

---

## 최종 완료 기준

| 구분 | 완료 기준 |
|---|---|
| 기능 | CSV 업로드 또는 샘플 데이터 기반으로 전체 분석 흐름이 실행된다. |
| 데이터 | 직접 수집/수동 다운로드/샘플 생성 출처가 `source_manifest.csv`에 기록된다. |
| EDA | 전처리와 이상치 처리 판단 근거가 노트북에 설명되어 있다. |
| ML | 감성분석 baseline 대비 개선 결과와 평가지표가 제시된다. |
| RAG/LLM | 경쟁 앱 공개 문서 근거 기반 AI 리포트가 생성된다. |
| 답글 생성 | OpenAI API 또는 fallback으로 리뷰 답글 초안이 생성되고, 안전장치와 human review 플래그가 동작한다. |
| 데모 | Streamlit 앱에서 분석 결과, 검증 결과, 답글 생성 결과를 시연할 수 있다. |
| 검증 | 데이터 품질, 모델 성능, RAG 적합성, 답글 품질, 통합 테스트 결과가 문서화된다. |
| 실패 분석 | 오분류/실패/부적절 답글 케이스 3개 이상을 제시한다. |
| 발표 | 팀원이 구현 이유와 한계를 설명할 수 있다. |

---

교육용 MVP 문서 - 실제 고객정보 및 내부 비공개 데이터 미사용 / 답글은 자동 게시가 아닌 검수용 초안
