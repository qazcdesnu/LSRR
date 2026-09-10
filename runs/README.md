# `runs/` — 실험 산출물

버전 관리 대상이 아니다. `lsrr/telemetry/tracker.py`가 여기에 쓴다.

## 구조
```
runs/
  runs_index.csv         # 전체 런 대장 (reporting/tables.py의 입력)
  <run_id>/
    config.yaml          # 해석 완료 설정 (런 신원)
    meta.json            # 백본 해시, 학습 파라미터 수·비율, 활성 안정화 단,
                         # 실제 배치 크기, I2 예외 플래그, 라이브러리 버전
    metrics.jsonl        # 스텝별 지표
    diagnostics.jsonl    # 사이클별 트레이스
    checkpoints/         # 학습 파라미터만 (백본 가중치 저장 금지)
    eval/                # anytime 곡선, 비용 보고, 파레토 점
```

`run_id = <exp>_<engine>_<backbone>_<YYYYMMDD_HHMMSS>_s<seed>`

## 주의
`Legacy_LSRR/runs/`의 산출물과 **섞어 보고하지 않는다.** 디코딩 경로가 다르므로(ADR-001) 비교 불가다.
