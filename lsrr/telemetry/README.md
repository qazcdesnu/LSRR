# `lsrr/telemetry/` — 실행 기록

## 역할
런의 모든 것을 디스크에 남긴다. 설정 스냅샷, 지표 시계열, **사이클별 진단 트레이스**, 런 대장.

## 경계
- **한다:** 쓰기(write). 런 디렉터리 구조와 파일 형식을 소유한다.
- **하지 않는다:** 지표를 계산하지 않는다(→ `metrics`). 분석하지 않는다(→ `analysis`). 표를 만들지 않는다(→ `reporting`).

## 모듈

| 모듈 | 역할 | 상태 |
|---|---|---|
| `tracker.py` | `ExperimentTracker` — 런 디렉터리 생성, 설정 스냅샷, 지표 기록 | 계획 |
| `traces.py` | 사이클 진단 트레이스 스키마와 영속화 | 계획 |
| `run_index.py` | `runs/runs_index.csv` 대장 | 계획 |
| `console.py` | 진행 표시·요약 출력 | 계획 |

## 핵심 계약

### 런 디렉터리 구조
```
runs/<run_id>/
  config.yaml          # 해석 완료 설정 전체 (런 신원)
  meta.json            # 백본 해시, 학습 파라미터 수·비율, 활성 안정화 단,
                       # 실제 배치 크기, I2 예외 플래그, 라이브러리 버전
  metrics.jsonl        # 스텝별 지표
  diagnostics.jsonl    # 사이클별 트레이스
  checkpoints/
  eval/                # 평가 산출물 (anytime 곡선, 비용 보고, 파레토 점)
```

`run_id = <exp>_<engine>_<backbone>_<YYYYMMDD_HHMMSS>_s<seed>` (레거시 규약 계승).

### 사이클 트레이스 (`traces.py`) — 분석의 원재료
사이클마다 기록한다: `m`, `delta_state`, `kl_div`, `entropy`, `stop_mask`, `alpha`(레이어별), 필요 시 `R^(m)` 요약 통계(노름·분산·랭크). 원본 `R^(m)` 전체는 기본적으로 저장하지 않되, **분석 대상 부분집합에 대해서만** 저장 플래그로 남긴다(용량 때문).

이 트레이스가 없으면 §7 메커니즘 분석 전체를 재실행 없이 수행할 수 없다. 무엇을 기록할지는 분석 계획에서 역산해 정한다:
| 분석 | 필요한 트레이스 필드 |
|---|---|
| Δ 궤적·거동 분류 | `delta_state` 전 사이클 |
| 홉별 수렴 사이클 | `stopping_cycle` + `meta.hop_count` |
| α 분포 | `alpha` |
| logit-lens 궤적 | `R^(m)` 원본 (부분집합) |
| anytime 곡선 | 사이클별 판독 정오 |

### 런 대장 (`run_index.py`)
모든 런의 한 줄 요약(설정 해시, 주요 노브, 최종 지표, 비용)을 CSV로 축적한다. `reporting/tables.py`의 입력이다.

## 의존
`core`. (L5.)

## 레거시 참조
`v1.0:lsrr/utils/logging.py` — **개작**. `ExperimentTracker`와 런 ID 규약, `diagnostics.jsonl`은 이식하고 트레이스 스키마를 별도 모듈로 분리한다.

## 상태
**검증(부분)** — `tracker`·`traces` 완료 (M3). `run_index`·`console`은 M5~M6.
