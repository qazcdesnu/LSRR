# `lsrr/runtime/` — 구동

## 역할
학습·평가를 **돌리는** 일. 옵티마이저, 스케줄, 체크포인트, 시드·결정성, 디바이스, OOM 대응, 재개.

## 경계
- **한다:** 루프 구동과 상태 관리.
- **하지 않는다:** 지표를 정의하지 않는다(→ `metrics`). 기록 형식을 정하지 않는다(→ `telemetry`). 모델을 조립하지 않는다(→ `lsrr/builder.py`).

## 모듈

| 모듈 | 역할 | 상태 |
|---|---|---|
| `trainer.py` | 학습 루프, 옵티마이저·LR 스케줄, grad clip, AMP | 계획 |
| `evaluator.py` | 평가 구동 (채점·비용은 `metrics`에 위임) | 계획 |
| `checkpoint.py` | 저장·로드 — **학습 파라미터만** (백본 가중치 저장 금지) | 계획 |
| `resume.py` | 중단 재개 (옵티마이저·스케줄·데이터 순서 복원) | 계획 |
| `seeding.py` | 시드·결정성 설정 | 계획 |
| `device.py` | 디바이스·dtype·AMP 정책 | 계획 |
| `oom.py` | OOM 감지·배치 축소 대응 | 계획 |

## 핵심 계약

### 학습 파라미터 범위
옵티마이저에 넘기는 파라미터는 **레이어 어댑터 + 엔진 + 융합 헤드**뿐이다 (§5). `trainer.py`는 시작 시 학습 대상 파라미터 수를 백본 대비 비율과 함께 로그에 남긴다 — §5의 "3% 이내" 주장이 매 런에서 확인되어야 한다.

체크포인트에 백본 가중치를 저장하지 않는다(용량 낭비이자 I1의 혼동 원인). 대신 백본 **식별자와 가중치 해시**를 저장해 로드 시 대조한다.

### 학습 스텝의 형태
```
1. backbone.encode(question)          # 무그래디언트, 1회 (I2)
2. memory.pipeline(bundle)     -> R0
3. recurrence.run_train(R0, hooks)  -> R*, per-cycle readouts
4. readout.readout(R*, h_ctx)  -> logits          # 훅이 쓴 것과 동일 인스턴스 (I3)
5. objectives.composite(trace, batch) -> loss
6. backward / clip / step             # 백본에는 그래디언트 없음 (I1, I4)
```

### 결정성
시드 고정 + `torch.use_deterministic_algorithms`. 재현성은 회귀 테스트로 강제한다(레거시 `test_reproducibility.py` 계승 — 비트 단위 동일성).

### OOM 대응
KV 캐시가 상주하므로 레거시보다 메모리 압박이 크다 (ADR-002). 배치 축소 후 재시도하되, **축소된 배치 크기를 런 메타에 기록**한다 — 배치 크기가 조용히 달라진 런을 같은 셀에서 비교하지 않기 위해서다.

## 의존
`core`, `config`, `objectives`, `telemetry`, `metrics`, 조립된 모델. (L5.)

## 레거시 참조
`Legacy_LSRR/lsrr/training/trainer.py` — **개작**. 1-cycle half-cosine LR 스케줄(Mamba-2 레시피), grad clip, 체크포인트, 재개는 이식. 판독이 백본 경유로 바뀌었으므로 스텝 루프 자체는 재작성한다.
`Legacy_LSRR/lsrr/utils/{seed,oom}.py` — **이식**.
`Legacy_LSRR/tests/{test_reproducibility,test_logging_and_resume,test_oom}.py` — 이식.

## 상태
**검증** — M3 완료. `resume.py`만 남았다(체크포인트 왕복은 동작).
