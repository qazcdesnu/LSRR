# `lsrr/objectives/` — 학습 목표

## 역할
제안서 §5의 손실 함수들.

$$L = L_{NLL} + \lambda_{ds}L_{DeepSup} + \lambda_{reg}L_{VarReg}\ (+\ \lambda_{KD}L_{KD}:\text{ablation 전용})$$

학습 대상은 **레이어 어댑터 + SSM 엔진 + 풀링/융합 헤드**뿐이다 (백본 대비 약 3% 이내). 백본은 완전 동결하며 LoRA도 쓰지 않는다.

## 경계
- **한다:** 손실 항 계산, 타깃 마스킹, 가중 합성, 항별 지표 분리 보고.
- **하지 않는다:** 모델 모듈을 호출하지 않는다. **손실은 값만 받는다** — 사이클별 로짓이 필요하면 훅이 미리 계산해 트레이스에 넣어 준다 (ADR-005).

## 모듈

| 모듈 | 역할 | 상태 |
|---|---|---|
| `targets.py` | 정답 구간 마스킹, `IGNORE_INDEX` 규약 (I6) | 계획 |
| `answer_nll.py` | `L_NLL` — `h_fusion` 조건 하 답 토큰 NLL | 계획 |
| `deep_supervision.py` | `L_DeepSup` — 답-앵커형, γ 가중 (ADR-006) | 계획 |
| `variance_reg.py` | `L_VarReg` — 상태 분산 하한 (자명해·상수 붕괴 방지, 소계수) | 계획 |
| `distillation.py` | `L_KD` — **ablation 전용** | 계획 |
| `composite.py` | 가중 합성, 항별 지표를 개별로도 보고 | 계획 |

## 핵심 계약

```
BaseObjective.forward(trace: ReasoningTrace, batch) -> dict[str, Tensor]  # 최소 "loss"
```

### 타깃 규약 (`targets.py`) — 가장 흔한 사고 지점
모든 답 손실의 타깃은 **정답 토큰 one-hot에 대한 표준 cross-entropy(NLL)**로 통일한다 (§5). 두 가지를 구분한다:
- `target_ids` — 디코더 **입력**. 패딩이 실제 토큰 id로 채워진다.
- `labels` — 손실 **타깃**. 패딩이 `IGNORE_INDEX(-100)`로 마스킹된다.

이 둘을 혼동하면 패딩을 학습한다. 레거시가 주석으로 명시했던 함정이며(`losses/composite.py:_loss_targets`), 여기서도 같은 책임을 진다. 정답이 없는 배치는 **건너뛰지 않고** `MissingTargetError`로 던진다 — 조용히 넘기면 자유형 데이터셋에서 100%, 수치형에서 0%가 보고된다.

### 깊은 감독 (`deep_supervision.py`) — §5의 정확한 명세

$$L_{DeepSup} = \sum_{m\in S} w_m \cdot \mathrm{NLL}(\text{answer} \mid h_{fusion}^{(m)})$$

- **감독 집합 `S`:** truncated BPTT 윈도 **안에서** 1~2개 사이클 균등 샘플링. 윈도 밖은 detach되어 감독 효과가 없다 (ADR-006).
- **가중:** `w_m = γ^(M−m)`, γ≈0.85 — 후반 사이클을 강조해 '점진 개선 + 후반 완성' 압력을 만든다.
- **판독 경로는 전 사이클 공유.** 사이클별 헤드 금지 (I3) — anytime 성질의 원천이다.

이는 잠재 반복이 감독 없이 표현 동질화로 붕괴한다는 진단에 대한 **교사-불요 처방**이다. `m`번째 사이클을 `m`번째 추론 스텝에 정렬하는 **스텝-앵커형** 감독은 이 손실이 아니라 `L_KD` ablation으로 분리한다 — 본 구조의 사이클은 스텝 방출이 아니라 전체 메모리 정제이기 때문이다.

**위험:** 과잉 감독은 반복을 형해화한다(모든 사이클이 이미 답을 알면 반복이 무의미해진다). Phase 0 게이트 ②(M 증가 시 정확도 증가)가 이를 검출하며, 처방은 `λ_ds`·`γ` 하향이다.

### 증류 (`distillation.py`) — ablation 전용
기본 주장('교사 CoT 없이 성립')과 **분리해 부스터로만** 검증한다. 기본 손실 목록에 들어가서는 안 된다. 설계 시 반영할 것: 유계 발산(JS), stop-gradient 교사, 정답 조건부 가중.

## 의존
`core`, `recurrence`(TBPTT 윈도 경계 조회). (L3.)

## 레거시 참조
`Legacy_LSRR/lsrr/losses/composite.py` — **개작(파일 분할)**. `AnswerNLLLoss`와 `StateVarianceRegLoss`는 이식. `DeepSupervisionLoss`는 (i) 윈도 경계 인식 (ii) γ 가중 (iii) 모듈 객체 대신 사전 계산된 사이클별 로짓 수신으로 개작한다. `_loss_targets`의 `labels`/`target_ids` 구분은 `targets.py`로 승격.

## 상태
**검증(부분)** — `targets`·`answer_nll`·`composite` 완료 (M3). `deep_supervision`·`variance_reg`는 M5.
