# LSRR 시스템 아키텍처

> **문서 지위:** 이 문서는 LSRR_01 코드베이스의 최상위 설계 문서다. `documents/Research_Proposal.md`(무엇을 주장하는가)와 각 폴더의 `README.md`(각 부품이 무엇을 하는가) 사이를 잇는다.
> 충돌 시 우선순위: `Research_Proposal.md` > `ARCHITECTURE.md` > 폴더별 `README.md` > 코드.
> **최종 갱신:** 2026-09-10 · **상태:** 설계 확정, 구현 미착수

---

## 0. 문서 지도

| 문서 | 답하는 질문 |
|---|---|
| `documents/Research_Proposal.md` | 무엇을 주장하고 무엇을 증명해야 하는가 |
| `documents/ARCHITECTURE.md` (본 문서) | 시스템 전체가 어떻게 조립되는가 |
| `documents/MODULE_INDEX.md` | 어떤 모듈이 어디에 있고 지금 어떤 상태인가 (지속 갱신 보드) |
| `documents/DESIGN_DECISIONS.md` | 왜 이렇게 결정했는가 (ADR, 특히 레거시와 갈라지는 지점) |
| `documents/FINDINGS.md` | 구현 중 **무엇이 측정되었는가** (실측·위험·교훈) |
| `documents/CONVENTIONS.md` | 컴포넌트를 어떤 규약으로 쓰고 문서를 어떻게 유지하는가 |
| `documents/LEGACY_MAP.md` | `Legacy_LSRR`의 무엇을 이식/개작/폐기하는가 |
| `documents/ROADMAP.md` | 어떤 순서로 구현하고 어느 게이트에서 멈추는가 |
| `lsrr/<pkg>/README.md` | 이 폴더의 역할·경계·모듈 목록·계약 |

---

## 1. 설계 원칙 — 제안서의 주장을 코드 제약으로 번역

제안서의 핵심 주장은 다섯 문장으로 요약되며, 각각이 코드베이스의 강제 가능한 제약이 된다.

| # | 제안서의 주장 | 코드 제약 |
|---|---|---|
| **P1** | 트랜스포머는 **1회 실행되는 문맥 인코더**이고 완전 동결된다 (§2, §5) | 백본은 학습 파라미터를 갖지 않으며, 사이클 루프는 백본에 재진입할 수 없다 → 불변식 I1·I2 |
| **P2** | 스캔 축은 **레이어(깊이)**, 반복 축은 **격자 밖 사이클 m** (§4.2) | `engine`(레이어 축 연산자)과 `recurrence`(사이클 축 제어)는 **서로 다른 패키지**로 분리하며 상호 참조하지 않는다 |
| **P3** | 판독 경로(풀링·융합·주입)는 **전 사이클 공유**, 사이클별 헤드 금지 (§5) | 판독 경로는 단일 객체(`ReadoutPath`)로 구현하고, 모든 사이클이 동일 인스턴스를 호출한다 → 불변식 I3 |
| **P4** | 수렴은 **보장이 아니라 유도**되며 `M_max` 폴백은 필수 명세 (§4.3) | 모든 종료 규칙은 `m = M_max`에서 반드시 정지한다. 안정화 장치는 사다리(ladder)로 분리해 실패 시 단계적으로 켠다 → 불변식 I5 |
| **P5** | 답변 생성 능력은 **끝까지 백본 소유**, 별도 학습 디코더 없음 (§4.4) | 학습 가능한 디코더 슬롯을 두지 않는다. 판독은 `h_fusion`을 백본 입력 임베딩 공간에 주입하는 통로뿐이다 → 불변식 I8 |

부수 원칙 하나: **모든 정확도 셀은 비용과 함께 보고된다** (§6.2) — FLOPs·평균 사이클·지연은 선택 지표가 아니라 평가 프로토콜의 일부이며 `metrics` 패키지가 소유한다.

---

## 2. 엔드투엔드 데이터 플로우

```
 (question, answer)
        │
        ▼
 ┌──────────────────┐
 │ data             │  DataSample{question, answer, cot_steps, meta}
 │                  │  · 질문만 인코딩 (정답 비노출, I6)
 └────────┬─────────┘
          ▼
 ┌────────────────────────────────────────────────────────┐
 │ backbone  ── 동결 백본 순전파 **단 1회** (I1, I2)       │
 │                                                        │
 │   ContextBundle {                                      │
 │     H_last   [B, L, d]   질문 마지막 토큰의 수직 1열     │
 │     H_pool   [B, L, d]   질문 전체 어텐션 풀링 (§4.1)   │
 │     kv_cache             질문 구간 KV — 연속 디코딩용    │
 │     h_ctx    [B, d]      백본 원본 h^(L) — 융합 잔차 앵커 │
 │     meta     {q_len, mask, backbone_id, ...}           │
 │   }                                                    │
 └────────┬───────────────────────────────────────────────┘
          ▼
 ┌──────────────────┐
 │ memory           │  compose(H_last, H_pool) → scope(레이어 구간, Ablation A)
 │                  │  → per-layer affine + norm + layer pos emb  (§4.1)
 │                  │  R⁰ ∈ R^{B×L×d_model}
 └────────┬─────────┘
          ▼
 ┌────────────────────────────────────────────────────────────────┐
 │ recurrence  ── 사이클 축 m 제어 (학습: M 샘플링 + TBPTT /       │
 │                평가: 동적 종료)                                 │
 │                                                                │
 │   for m in 0..M-1:                                             │
 │     ┌──────────────┐                                           │
 │     │ engine       │  R^(m+1) = (1-α)R^(m) + α·S_φ(R^(m),R⁰,m) │
 │     │              │  레이어 축 양방향 스캔 (quasiseparable)     │
 │     └──────┬───────┘  + 감쇠 / R⁰ 재주입 / 사이클 임베딩         │
 │            │                                                   │
 │     ┌──────▼───────┐   ┌──────────────┐                        │
 │     │ termination  │   │ stability    │ (선택적 사다리)          │
 │     │ Δ<ε / KL / H │   │ jacobian ··· │                        │
 │     └──────┬───────┘   └──────────────┘                        │
 │            │                                                   │
 │     ┌──────▼───────┐                                           │
 │     │ hooks        │ → 사이클별 진단 기록 / 깊은 감독 판독 /      │
 │     └──────────────┘   anytime 곡선                            │
 └────────┬───────────────────────────────────────────────────────┘
          ▼  R* = R^(M)
 ┌────────────────────────────────────────────────────────┐
 │ readout  ── 전 사이클 공유 단일 경로 (I3)                │
 │   α_l = softmax(wᵀr_l*),  h_SSM = Σ α_l r_l*           │
 │   h_fusion = h_ctx + W_r·h_SSM       (백본 임베딩 공간, I8) │
 └────────┬───────────────────────────────────────────────┘
          ▼
 ┌────────────────────────────────────────────────────────┐
 │ backbone.continuation  ── 백본 연속 디코딩 (§4.4)        │
 │   h_fusion을 질문 다음 위치 입력 벡터로 1회 주입          │
 │   + 질문 KV 캐시 재사용 → 동결 백본이 답 토큰 자기회귀 생성 │
 │   역전파는 h_fusion 위치를 통해서만 (I4)                  │
 └────────┬───────────────────────────────────────────────┘
          ▼
 ┌──────────────────┐
 │ objectives       │  L_NLL + λ_ds·L_DeepSup + λ_reg·L_VarReg (+ λ_KD·L_KD)
 └────────┬─────────┘
          ▼
   runtime(학습/평가 구동) → telemetry(기록) → metrics(채점·비용)
          → analysis(메커니즘) → reporting(표·그림) → gates(단계 판정)
```

**읽는 법:** 위에서 아래로 흐르는 경로가 순전파다. `engine`은 **오직 레이어 축**만 본다(사이클 인덱스는 조건 입력일 뿐이다). `recurrence`는 **오직 사이클 축**만 본다(엔진 내부를 모른다). 이 두 축의 분리가 제안서 §4.2의 "축의 구분"을 코드 구조로 못 박은 것이며, 본 아키텍처에서 가장 중요한 경계다.

---

## 3. 슬롯 맵 — 교체 가능한 컴포넌트 목록

각 슬롯은 `core/interfaces.py`의 추상 클래스 하나와 레지스트리 하나에 대응한다. 실험은 **코드 수정이 아니라 설정 파일의 `type` 교체**로 수행된다.

| 슬롯 | 패키지 | 인터페이스 | 레지스트리 | 기본 구현 | 교체 대상(실험) |
|---|---|---|---|---|---|
| 문맥 인코더 | `backbone` | `BaseContextEncoder` | `BACKBONE` | `hf_frozen_causal` | GPT-2 / Llama-3.2-1B / Mistral-7B / 3B / 8B |
| 문맥 풀링 | `backbone` | `BaseContextPooler` | `POOLER` | `attn_pool` | last-only / mean / attn (§4.1 보완항) |
| 메모리 구성 | `memory` | `BaseMemoryComposer` | `COMPOSER` | `last_plus_pool_gate` | add / concat / gate / last-only |
| 메모리 범위 | `memory` | `BaseMemoryScope` | `SCOPE` | `all_layers` | **Ablation A**: h(L)단독 / 전체 / 중간구간 / 후반구간 |
| 레이어 어댑터 | `memory` | `BaseLayerAdapter` | `ADAPTER` | `per_layer_affine+rmsnorm` | shared_affine / identity |
| 사고 엔진 | `engine` | `BaseRefinementEngine` | `ENGINE` | `hydra_qs` | **Ablation C**: mlp_onepass / attn_block / mamba_up / mamba_down / bidir_add / hydra_qs |
| 재귀 적응 | `engine` | (`EngineWrapper` 고정) | — | 감쇠+재주입+사이클emb | **Ablation D**: α 스윕 / 재주입 유무 / 사이클emb 유무 |
| 사이클 스케줄 | `recurrence` | `BaseCycleSchedule` | `SCHEDULE` | `lognormal` | fixed / uniform / lognormal |
| 종료 규칙 | `termination` | `BaseTerminationRule` | `TERMINATION` | `delta_state` | **Ablation B**: fixed_m / delta_state / kl_output / entropy_output |
| 안정화 | `stability` | `BaseStabilityConstraint` | `STABILITY` | `none` | jacobian_reg / spectral_norm / monotone (사다리) |
| 융합 헤드 | `readout` | `BaseFusionHead` | `FUSION` | `attention_pooling` | residual / gate / concat |
| 판독 경로 | `readout` | `BaseReadoutPath` | `READOUT` | `backbone_continuation` | (대안 없음 — P5에 의해 고정) |
| 손실 | `objectives` | `BaseObjective` | `OBJECTIVE` | `answer_nll` 외 | deep_supervision / variance_reg / distillation(ablation) |
| 데이터 | `data` | `BaseDataModule` | `DATA` | — | gsm8k_aug / prosqa / prontoqa / … / closed_book_multihop |
| 분석 | `analysis` | `BaseAnalysisPlugin` | `ANALYSIS` | — | logit_lens / alpha_profile / hop_convergence / backpatch |
| 게이트 | `gates` | `BasePhaseGate` | `GATE` | — | phase0 ①~④ / phase1 |

> **내부 베이스라인은 별도 코드가 아니다.** 제안서 §6.2의 내부 베이스라인 3종은 모두 슬롯 교체로 표현된다 — "H+1회 통과 MLP" = `engine=mlp_onepass`, "h(L)+동일 SSM" = `scope=final_only`, "동FLOPs 어텐션 엔진" = `engine=attn_block` + `engine.budget_match=hydra_qs`. 베이스라인이 본 모델과 **같은 코드 경로**를 지나는 것이 공정 비교의 전제다.

---

## 4. 불변식 (Invariants)

아키텍처가 무너지면 실험 결과 전체가 무효가 되는 조건들이다. 각 불변식은 **실행 시 단언(`core/invariants.py`)과 테스트(`tests/contracts/`) 양쪽**으로 강제한다.

| # | 불변식 | 왜 치명적인가 | 강제 지점 |
|---|---|---|---|
| **I1** | 백본 파라미터는 `requires_grad=False`이며 학습 전후 가중치 해시가 동일하다 | 백본이 조금이라도 학습되면 "동결 백본에 사후 장착" 주장이 붕괴 | `backbone/session.py` 로드 시 + `tests/contracts/test_frozen_backbone.py` |
| **I2** | 샘플당 질문 인코딩 백본 순전파는 **정확히 1회**. 사이클 루프 안에서 백본 재진입 금지 | 재진입은 곧 Coconut형 반복 호출 — 본 연구의 효율 주장 자체가 사라짐 (재인코딩 외부 루프는 Phase 3 예비 실험으로만 격리) | `core/invariants.py`의 인코딩 카운터 + `tests/contracts/test_single_encode.py` |
| **I3** | 모든 사이클이 **동일한** `ReadoutPath` 인스턴스를 호출한다 (사이클별 헤드 금지) | anytime 성질의 원천. 사이클별 헤드는 깊은 감독을 무의미하게 만듦 (§5) | `readout/path.py` 인스턴스 동일성 검사 + `tests/contracts/test_shared_readout.py` |
| **I4** | 엔진으로의 역전파는 `h_fusion` 주입 위치를 통해서만 흐른다 | 다른 통로가 생기면 "백본은 판독만 담당" 구조가 아님 | `backbone/continuation.py` + `tests/contracts/test_gradient_path.py` |
| **I5** | 모든 종료 규칙은 `m = M_max`에서 반드시 정지한다 | 진동/드리프트 샘플에서 무한 루프. 제안서 §4.3이 필수 명세로 지정 | `termination/rules.py` 공통 기반 + `tests/contracts/test_termination_fallback.py` |
| **I6** | 문맥 인코딩 입력에 정답 토큰이 포함되지 않으며, 손실 타깃은 정답 구간만(패딩은 `IGNORE_INDEX`) | 누출 시 전 실험 무효 | `data/prompting.py`, `objectives/targets.py` + `tests/contracts/test_no_leakage.py` |
| **I7** | 보고되는 FLOPs·지연은 백본 1회 순전파를 **포함**한다. 측정 불가 시 `flops_includes_backbone: false`로 명시 플래그 | 백본 비용을 빼면 효율 비교가 무의미 | `metrics/cost.py` + `tests/contracts/test_cost_accounting.py` |
| **I8** | `h_fusion`은 **백본 입력 임베딩 공간**의 벡터다 (`h_ctx`는 어댑터 통과 전 원본 `h^(L)`) | 주입 통로가 요구하는 공간 정합. 어댑터 공간 벡터를 주입하면 매니폴드 불일치 | `readout/fusion.py`, `readout/injection.py` + `tests/contracts/test_injection_space.py` |

---

## 5. 패키지 레이어와 의존 규칙

**규칙: 상위 레이어는 하위 레이어만 import한다. 같은 레이어끼리는 import하지 않는다. 역방향 import는 금지.**

```
 L7  gates          reporting
 L6  analysis
 L5  runtime        metrics        telemetry
 L4  lsrr/model.py  lsrr/builder.py            ← 조립 루트 (유일하게 전 슬롯을 안다)
 L3  objectives
 L2  recurrence     readout
 L1  memory   engine   termination   stability   backbone   data
 L0  core     config
```

- **L0 `core`는 어떤 것도 import하지 않는다.** 계약(인터페이스·타입·레지스트리)만 소유한다.
- **L1의 형제들은 서로 모른다.** `engine`은 `termination`을 모르고, `memory`는 `backbone`을 모른다 (`ContextBundle`이라는 `core` 타입으로만 대화한다).
- **조립은 L4에서만.** 슬롯을 골라 연결하는 지식은 `lsrr/builder.py` 한 곳에만 존재한다. 이것이 "어떤 부품도 다른 부품의 구체 타입을 모른다"는 컴포넌트화의 실질이다.
- 순환 의존이 필요해 보이면 그것은 `core/types.py`에 들어갈 데이터 타입이 빠졌다는 신호다.

---

## 6. 폴더별 역할 (최상위)

| 경로 | 역할 한 줄 | 상세 |
|---|---|---|
| `documents/` | 연구 계획·아키텍처·결정 기록·로드맵 | — |
| `lsrr/core/` | 모든 컴포넌트가 지키는 계약: 인터페이스·데이터 타입·레지스트리·불변식 단언 | `lsrr/core/README.md` |
| `lsrr/config/` | YAML 설정의 스키마·해석·검증·스윕 전개·스냅샷 | `lsrr/config/README.md` |
| `lsrr/backbone/` | 동결 백본과 닿는 모든 것: 1회 인코딩, 레이어 추출, 질문 풀링, 연속 디코딩, 개입 훅 | `lsrr/backbone/README.md` |
| `lsrr/memory/` | H → R⁰ 변환: 구성·범위 선택(Ablation A)·레이어 어댑터·레이어 위치 임베딩 | `lsrr/memory/README.md` |
| `lsrr/engine/` | **레이어 축** 정제 연산자들과 §4.2 갱신식을 소유하는 래퍼 (Ablation C·D) | `lsrr/engine/README.md` |
| `lsrr/recurrence/` | **사이클 축** 제어: M 샘플링, TBPTT, 조기 종료 래칭, 사이클 훅 | `lsrr/recurrence/README.md` |
| `lsrr/termination/` | 수렴 신호 계산과 종료 규칙, ε 보정, 수렴 거동 분류 (Ablation B) | `lsrr/termination/README.md` |
| `lsrr/readout/` | 전 사이클 공유 판독 경로: 어텐션 풀링 융합 → 주입 → 백본 연속 디코딩 | `lsrr/readout/README.md` |
| `lsrr/stability/` | 수렴 유도 실패 시의 안정화 사다리 (Jacobian → 스펙트럴 → monotone) | `lsrr/stability/README.md` |
| `lsrr/objectives/` | 손실 항들과 합성: NLL, 깊은 감독, 분산 정규화, 증류(ablation 전용) | `lsrr/objectives/README.md` |
| `lsrr/data/` | 데이터셋 모듈·프롬프트 규약·콜레이트·정답 채점·H 캐시·다운로드 | `lsrr/data/README.md` |
| `lsrr/runtime/` | 학습/평가 구동, 체크포인트, 시드·결정성, OOM 대응, 재개 | `lsrr/runtime/README.md` |
| `lsrr/metrics/` | 채점 프로토콜과 비용 회계: exact match, anytime 곡선, FLOPs·지연, 파레토 | `lsrr/metrics/README.md` |
| `lsrr/telemetry/` | 실행 기록: 런 디렉터리, 사이클 트레이스, 런 인덱스, 콘솔 출력 | `lsrr/telemetry/README.md` |
| `lsrr/analysis/` | 사후 메커니즘 분석 플러그인 (logit lens, α 분포, 홉-수렴, back-patching 회수율) | `lsrr/analysis/README.md` |
| `lsrr/reporting/` | 런 산출물 + 공표치 → 논문용 표·그림, 출처 이원 표기(†/‡) | `lsrr/reporting/README.md` |
| `lsrr/gates/` | 단계별 게이트 판정과 킬 스위치 신호 | `lsrr/gates/README.md` |
| `configs/` | 실행 가능한 설정 트리 (실험 = 설정 조합) | `configs/README.md` |
| `scripts/` | CLI 진입점 (얇은 래퍼, 로직 금지) | `scripts/README.md` |
| `tests/` | 계약·단위·통합·회귀 테스트 | `tests/README.md` |
| `runs/` | 실험 산출물 (버전 관리 제외) | `runs/README.md` |
| `caches/` | 추출 은닉 상태 캐시 (버전 관리 제외) | `caches/README.md` |
| `Legacy_LSRR/` | **읽기 전용 참조 구현. 절대 수정 금지.** | `documents/LEGACY_MAP.md` |

---

## 7. 레거시 대비 구조적 변경점 (요약)

상세 근거는 `documents/DESIGN_DECISIONS.md`. 여기서는 왜 폴더 구조가 달라졌는지만 짚는다.

| 변경 | 레거시 | 신규 | 이유 |
|---|---|---|---|
| **디코딩 경로** | `decoders/light_decoder.py` — 학습 가능한 경량 트랜스포머 디코더 | 학습 디코더 슬롯 **폐지**. `readout` + `backbone/continuation.py` | 개정 제안서 §4.4가 백본 연속 디코딩을 명시하고 "별도의 학습 디코더는 두지 않는다"를 원칙으로 선언 (ADR-001) |
| **백본의 지위** | 캐시 추출용 오프라인 도구 (학습 중 미로드) | 학습 루프의 상주 컴포넌트 (KV 캐시 + 판독 담당) | 연속 디코딩이 질문 KV를 요구 (ADR-002) |
| **융합 잔차 앵커** | `R0[:, -1, :]` (어댑터 통과 후) | `h_ctx` = 백본 원본 `h^(L)` (어댑터 통과 전) | 주입 공간 정합 I8 (ADR-003) |
| **문맥 보완항** | 없음 (last_token 단독) | `H_pool` — 질문 전체 어텐션 풀링을 결합 | 제안서 §4.1의 명시 요구사항이 레거시에 미구현 (ADR-004) |
| **사이클 축 제어** | `iteration/controller.py` 한 파일이 학습·평가·종료·판독을 모두 소유 | `recurrence`(축 제어) / `termination`(규칙) / `readout`(판독) 3분할 | 판독 객체를 `model_outputs` 딕셔너리로 실어 나르던 결합을 제거, I3를 구조로 보장 (ADR-005) |
| **깊은 감독** | 중간 상태 랜덤 2개 샘플링, 균등 가중 | TBPTT 윈도 내 샘플링 + `w_m = γ^(M−m)` 가중 | 개정 §5의 답-앵커형 명세 (ADR-006) |
| **안정화 장치** | 없음 | `stability` 패키지 (사다리) | 개정 §4.3의 실패 대비 명세 (ADR-007) |
| **게이트/분석/보고** | 스크립트에 산재 | `analysis` / `reporting` / `gates` 패키지 | Phase 0 킬 스위치를 실행 가능한 판정으로 (ADR-008) |

---

## 8. 실험 축 ↔ 설정 축 대응

제안서 §7의 절제 실험표가 그대로 설정 스윕이 되어야 한다. 새 실험을 위해 새 파일을 만드는 일이 생긴다면 슬롯 설계가 틀린 것이다.

| 실험 | 설정 노브 | 설정 파일 |
|---|---|---|
| **A. 메모리 범위** | `memory.scope.type ∈ {final_only, all_layers, mid_band, late_band}` | `configs/ablation/A_memory_scope.yaml` |
| **B. 종료 규칙** | `termination.type ∈ {fixed_m, delta_state, kl_output, entropy_output}` × ε 스윕 | `configs/ablation/B_termination.yaml` |
| **C. 사고 엔진** | `engine.type ∈ {mlp_onepass, attn_block, mamba_up, mamba_down, bidir_add, hydra_qs}` + `engine.budget_match` | `configs/ablation/C_engine.yaml` |
| **D. 재귀 적응·감독** | `engine.damping_alpha` 스윕, `engine.reinject_r0`, `engine.cycle_embedding`, `objective.deep_supervision.{enabled,gamma}` | `configs/ablation/D_recurrence.yaml` |
| **KD 부스터** | `objective.distillation.enabled=true` (기본 손실에서 분리) | `configs/ablation/E_distillation.yaml` |
| **범위 검증** | `data=closed_book_multihop` — 이득의 **부재**를 예측하는 반증 실험 | `configs/exp/scope_check.yaml` |
| **스케일 곡선** | `backbone ∈ {gpt2, llama32_1b, llama32_3b, llama31_8b}` | `configs/exp/scale_curve.yaml` |

---

## 9. Phase 로드맵 대응

`documents/ROADMAP.md`가 구현 순서를 다루고, 여기서는 어느 패키지가 어느 게이트를 지탱하는지만 표시한다.

| Phase | 게이트 | 필요 패키지 (최소) |
|---|---|---|
| **Phase 0** (GPT-2 + 곱셈·ProsQA) | ① 자명해 붕괴 없음 ② M↑ → 정확도↑ ③ H+MLP 1회 통과 대비 유의한 우위 **(실패 시 킬 스위치)** ④ Δ⁽ᵐ⁾ 거시적 감소 | core, config, data, backbone, memory, engine, recurrence, termination, readout, objectives, runtime, metrics, telemetry, gates |
| **Phase 1** (GSM8k-Aug 본 실험) | 종료 규칙 비교, 파레토 | + `termination/calibration.py`, `metrics/pareto.py`, `reporting` |
| **Phase 2** (메커니즘·층화·전이) | 홉별 수렴, α 분포, logit lens, back-patching 회수율 | + `analysis` 전체, `backbone/interventions.py` |
| **Phase 3** (선택: 스케일·재인코딩 외부 루프) | — | + 백본 확장 설정. 재인코딩 루프는 **I2의 명시적 예외 플래그** 하에서만 허용 |

---

## 10. 문서 유지 규약

1. **폴더를 만들면 그 폴더의 `README.md`를 같은 커밋에서 만든다.** README 없는 폴더는 존재하지 않는 것으로 간주한다.
2. **모듈을 추가/삭제/이동하면** 해당 폴더 `README.md`의 모듈 표와 `documents/MODULE_INDEX.md`의 상태를 같이 갱신한다.
3. **설계가 바뀌면** 코드보다 `DESIGN_DECISIONS.md`에 ADR을 먼저 추가한다. ADR 번호를 코드 주석에서 인용한다 (`# ADR-003`).
3-1. **측정이 설계를 바꿨거나 위험을 드러냈으면** `documents/FINDINGS.md`에 수치·조건·재현 방법과 함께 기록한다. ADR은 결정을, FINDINGS는 사실을 담는다 — 결정은 뒤집혀도 측정은 남는다.
4. **불변식을 추가하려면** 본 문서 §4 표에 행을 추가하고, 같은 커밋에 `tests/contracts/` 테스트를 추가한다.
5. `Legacy_LSRR/`은 **읽기 전용**이다. 참조한 내용은 `LEGACY_MAP.md`에 이식/개작/폐기 판정과 함께 기록한다.
