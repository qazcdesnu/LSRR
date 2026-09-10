# `lsrr/data/datasets/` — 데이터셋 모듈

## 역할
데이터셋별 로딩·스키마 변환·정답 정규화. 각 모듈은 `BaseDataModule`을 구현하고 `DATA` 레지스트리에 등록한다.

## 모듈

| 모듈 | 역할 | 벤치마크 지위 (§6.1) | 상태 |
|---|---|---|---|
| `gsm8k_aug.py` | GSM8K-Aug (train 385,620 / valid 500 / test 1,319) | **주 학습·ID 평가** | 계획 |
| `math_ood.py` | GSM-Hard · MultiArith · SVAMP | OOD 평가 전용 | 계획 |
| `prosqa.py` | ProsQA (train 17,886 / valid 300 / test 500), 홉 수 메타 포함 | 홉 수 통제 — **메커니즘 분석 주 무대** | 계획 |
| `prontoqa.py` | ProntoQA | 홉 수 통제 보조 | 계획 |
| `multiplication.py` | 다자리 곱셈 합성 (시드 생성, 다운로드 불필요) | 용량 확장 검증 · **Phase 0** | 계획 |
| `commonsenseqa.py` | CommonsenseQA (CODI 공개 CoT 데이터) | 비수학 일반성 | 계획 |
| `closed_book_multihop.py` | closed-book 다중 홉 QA 1종 | **범위 검증 — 이득의 부재를 예측** | 계획 |

## 핵심 계약

```
get_split(split) -> list[DataSample]
score(prediction, target, meta) -> bool
```

### `meta`에 반드시 담을 것
- `hop_count` (ProsQA/ProntoQA) — `analysis/hop_convergence.py`의 층화 축. **"홉이 많을수록 오래 생각한다"**는 단조 증가 예측을 검증하는 데이터이므로, 없으면 §7 메커니즘 분석 (i)을 수행할 수 없다.
- `digits` (곱셈) — 용량 확장 곡선의 축.
- `answer_type` — 채점 정규화 분기.

### `closed_book_multihop.py`의 특별한 지위
이 데이터셋은 **이득이 없기를 예측하는** 실험 대상이다 (§6.1, §9). 본 방법은 필요한 정보가 문맥과 1회 인코딩으로 확보되는 **결합-병목(composition-bound)** 추론을 겨냥하며, 파라미터 지식의 스텝별 재인출이 필요한 **인출-병목(retrieval-bound)** 과제는 범위 밖이다. 이득의 부재가 곧 메커니즘 이해의 증거이므로, 여기서 결과가 나쁘다고 데이터셋을 교체하거나 하이퍼파라미터를 더 튜닝해서는 안 된다 — 반증 가능한 예측을 사후에 구제하는 것은 실험이 아니다.

## 레거시 참조
`Legacy_LSRR/lsrr/data/{gsm8k,prosqa,multiplication}.py` — **이식**(파일 이동). 나머지는 신규.

## 상태
계획 — `multiplication`·`prosqa`는 M3, `gsm8k_aug`는 M6, 나머지는 M6~M7
