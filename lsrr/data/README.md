# `lsrr/data/` — 데이터

## 역할
제안서 §6.1의 데이터셋들을 통일된 스키마로 공급하고, 프롬프트 규약·배치·채점·캐시를 소유한다.

## 경계
- **한다:** 데이터 획득·검증, 스키마 통일, 질문 렌더링·토크나이즈, 배치·패딩·라벨 마스킹, 데이터셋별 정답 정규화, H 캐시.
- **하지 않는다:** 모델을 모른다. 평가 지표 집계는 `metrics`가 한다(여기는 **한 샘플의 정오 판정**까지).

## 모듈

| 모듈 | 역할 | 상태 |
|---|---|---|
| `schema.py` | `DataSample{question, answer, cot_steps, meta}`와 split 규약 | 계획 |
| `prompting.py` | 질문 렌더링·토크나이즈 계약 — **정답 비노출 강제** (I6) | 계획 |
| `collate.py` | 배치·패딩·`labels` 마스킹 | 계획 |
| `scoring.py` | 데이터셋별 정답 정규화 + exact match | 계획 |
| `cache.py` | 샤딩 safetensors H 캐시 + manifest + 캐시 키 (보조 경로, ADR-002) | 계획 |
| `download.py` | 데이터 획득 + SHA-256 검증 | 계획 |

`datasets/`는 별도 README 참조.

## 핵심 계약

### 프롬프트 규약 (`prompting.py`) — 누출 방지의 최전선
```
render_question(sample) -> str          # 정답·CoT 절대 미포함
tokenize_question(text) -> input_ids    # 우측 패딩
render_answer(sample)   -> str          # 손실·생성 대상
```
문맥 인코딩에 들어가는 것은 **질문뿐**이다 (I6). 이 계약이 깨지면 전 실험이 무효다.

### 채점 (`scoring.py`)
최종답 exact match. 데이터셋별 정규화(수치 파싱, 공백·구두점, 선택지 라벨)를 데이터셋 모듈이 소유하고, 프로토콜(greedy 디코딩, 샘플별 EOS 절단, **최종답만 채점**)은 `metrics/accuracy.py`가 소유한다. 이 분리는 잠재 추론 베이스라인들의 평가 규칙과 정렬하기 위한 것이다 (§6.2 — 공표치 직접 비교의 전제).

### 캐시 (`cache.py`) — 보조 경로임에 유의
캐시 키: `backbone_id, dataset_id, split, tokenizer_hash, position_rule, precision, include_embedding` 해시.
**캐시 경로에서는 연속 디코딩이 불가하다** (KV 캐시가 없다). 캐시로 학습을 시도하면 명시적 오류를 던진다. 캐시의 용도는 엔진 전용 프로파일링과 어댑터 단독 실험뿐이다 (ADR-002).

## 의존
`core`. (L1.)

## 레거시 참조
전부 **이식** 가능하며 잘 만들어져 있다.
- `v1.0:lsrr/data/{schema,collate,answer_scoring,cache}.py`
- `v1.0:scripts/download_data.py` — SHA-256 검증과 upstream 커밋 고정, GSM8K-Aug의 coconut 스키마 변환. 로직을 `download.py`로 옮기고 스크립트는 얇게.
- `v1.0:tests/{test_cache,test_gsm8k,test_download_data,test_target_padding}.py`

## 상태
**검증(부분)** — `schema`·`prompting`(M2)·`collate`(M3) 완료. `scoring`·`cache`·`download`는 M5~M6.
