# `lsrr/core/` — 계약

## 역할
모든 컴포넌트가 지키는 **계약**을 소유한다. 인터페이스, 컴포넌트 간에 오가는 데이터 타입, 레지스트리, 불변식 단언, 예외 타입.

## 경계
- **한다:** 추상 정의와 그것을 지키게 만드는 장치.
- **하지 않는다:** 어떤 구체 구현도, 어떤 텐서 연산도 하지 않는다. `torch.nn` 기반 클래스 선언 외에 torch 연산이 들어가면 잘못된 것이다.
- **import 규칙:** `core`는 이 패키지 밖의 어떤 것도 import하지 않는다 (레이어 L0).

## 모듈

| 모듈 | 역할 | 상태 |
|---|---|---|
| `interfaces.py` | 전 슬롯의 추상 기반 클래스 | 계획 |
| `types.py` | 컴포넌트 간 데이터 타입 (dataclass) | 계획 |
| `registry.py` | 범용 레지스트리 + 슬롯별 인스턴스 | 계획 |
| `invariants.py` | I1–I8 실행 시 단언 헬퍼 | 계획 |
| `errors.py` | 타입 예외 | 계획 |

## 핵심 계약

### 인터페이스 (`interfaces.py`)

| 추상 클래스 | 계약 |
|---|---|
| `BaseContextEncoder` | `encode(input_ids, attention_mask) -> ContextBundle` — 백본 순전파 1회 |
| `BaseContextPooler` | `pool(hidden_stack, mask) -> [B, L, d_in]` — 레이어별 질문 풀링 |
| `BaseMemoryComposer` | `compose(H_last, H_pool) -> [B, L, d_in]` |
| `BaseMemoryScope` | `select(H) -> [B, L', d_in]` + `layer_indices` 노출 (분석용) |
| `BaseLayerAdapter` | `forward(H) -> R0 [B, L', d_model]` |
| `BaseRefinementEngine` | `forward_step(R_m, R0, m) -> R_next` — **레이어 축만 본다** |
| `BaseCycleSchedule` | `sample_M() -> int`, `weights(M) -> list[float]` |
| `BaseTerminationRule` | `reset(B, device)`, `should_stop(signals, m) -> (stop_mask, CycleDiagnostics)` |
| `BaseStabilityConstraint` | `apply(engine)`, `penalty(R_m, R_next) -> Tensor` |
| `BaseFusionHead` | `forward(R_star, h_ctx) -> (h_fusion, alpha)` |
| `BaseReadoutPath` | `readout(R, h_ctx, context) -> ReadoutResult` — **전 사이클 공유 (I3)** |
| `BaseObjective` | `forward(trace, batch) -> dict[str, Tensor]` (최소 `"loss"` 포함) |
| `BaseDataModule` | `get_split(split) -> list[DataSample]`, `score(pred, target, meta) -> bool` |
| `BaseAnalysisPlugin` | `analyze(traces, run_dir) -> dict` |
| `BasePhaseGate` | `evaluate(run_records) -> GateVerdict` |

### 데이터 타입 (`types.py`)

| 타입 | 담는 것 |
|---|---|
| `DataSample` | `question, answer, cot_steps, meta` |
| `ContextBundle` | `H_last [B,L,d_in]`, `H_pool [B,L,d_in]`, `h_ctx [B,d_in]`, `kv_cache`, `attention_mask`, `meta` |
| `CycleDiagnostics` | `delta_state, kl_div, entropy, extra` — 사이클 1회분 |
| `ReadoutResult` | `logits, h_fusion, alpha` |
| `ReasoningTrace` | 위 전부 + `R0, R_star, stopping_cycles, per_cycle: list[CycleDiagnostics]`, `per_cycle_readout: list[ReadoutResult]` |
| `CostReport` | `flops_backbone, flops_adapter, flops_engine, flops_continuation, latency_ms, avg_cycles, flops_includes_backbone: bool` |

`ContextBundle`이 `backbone`과 `memory`가 서로를 모른 채 대화하는 유일한 통로다. 새 순환 의존이 필요해 보이면 여기에 필드가 빠진 것이다.

### 불변식 단언 (`invariants.py`)
`ARCHITECTURE.md` §4의 I1–I8을 실행 시 검사하는 헬퍼. 위반은 **경고가 아니라 예외**다.
- `EncodeCounter` — 샘플 배치당 백본 인코딩 횟수 추적 (I2)
- `assert_frozen(module)` / `weight_hash(module)` (I1)
- `assert_same_instance(readout_calls)` (I3)
- `assert_embedding_space(h_fusion, d_in)` (I8)

### 레지스트리 (`registry.py`)
`BACKBONE, POOLER, COMPOSER, SCOPE, ADAPTER, ENGINE, SCHEDULE, TERMINATION, STABILITY, FUSION, READOUT, OBJECTIVE, DATA, ANALYSIS, GATE`

## 의존
없음 (L0).

## 레거시 참조
- `Legacy_LSRR/lsrr/registry.py` — **이식**. 시그니처 기반 kwargs 필터링과 별칭 등록이 잘 설계되어 있다.
- `Legacy_LSRR/lsrr/interfaces.py` — **개작**. 슬롯 재편(디코더 폐지, ReadoutPath·Stability·Gate 신설) 및 dataclass 분리.

## 상태
**검증** — M1 완료. 5개 모듈, 인터페이스 18종·불변식 헬퍼 14종·레지스트리 15종.
