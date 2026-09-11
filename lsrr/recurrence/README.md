# `lsrr/recurrence/` — 사이클 축 제어

## 역할
격자 밖 정제 사이클 축 `m`을 소유한다. **몇 번 돌지, 어디까지 그래디언트를 흘릴지, 언제 멈출지, 매 사이클 무엇을 기록할지.**

## 경계
- **한다:** M 샘플링, 절단 BPTT 윈도 관리, 샘플별 조기 종료 래칭, 사이클 훅 디스패치.
- **하지 않는다:** 엔진 내부를 모른다(`forward_step`만 호출). 종료 **규칙**을 모른다(`termination`에 위임). 판독을 직접 하지 않는다(훅으로 `readout`을 호출할 뿐).

> `recurrence`는 반복 축만, `engine`은 스캔 축만 본다. 이 분리가 본 아키텍처의 최상위 경계다 (`ARCHITECTURE.md` P2).

## 모듈

| 모듈 | 역할 | 상태 |
|---|---|---|
| `runner.py` | `CycleRunner.run_train(R0, hooks)` / `run_eval(R0, hooks)` | 계획 |
| `schedules.py` | M 샘플링: `fixed` / `uniform` / `lognormal`(기본). 깊은 감독 γ 가중 산출 | 계획 |
| `tbptt.py` | 절단 BPTT 윈도 계산과 detach 정책. **윈도 경계를 공개 API로 노출** | 계획 |
| `state.py` | 샘플별 조기 종료 래칭, 실행 마스크, 정지 사이클 기록 | 계획 |
| `hooks.py` | 사이클 콜백 프로토콜 | 계획 |

## 핵심 계약

### 학습 (`run_train`)
1. `schedules.sample_M()` — 로그정규 샘플링. 임의 깊이에서 잘려도 정답을 유지해야 하므로 정상 상태가 유리해진다 (§4.3의 수렴 압력 ①).
2. `tbptt.window(M, k)` — 마지막 `k` 사이클만 그래디언트를 흘리고 이전은 detach.
3. 매 사이클: `engine.forward_step` → 진단 기록 → 훅 디스패치.

### 평가 (`run_eval`)
1. 매 사이클 `engine.forward_step` (무그래디언트).
2. `termination.should_stop(signals, m)` → 정지 마스크.
3. `state.latch()` — 새로 정지한 샘플의 상태를 고정하고 정지 사이클을 기록. 이미 정지한 샘플은 더 갱신하지 않는다.
4. `m = M_max`에서 반드시 종료 (I5).

### 훅 프로토콜 (`hooks.py`)

```
CycleHook.on_cycle(m, R_m, R_next, diagnostics) -> None
```

구현체: `DiagnosticsRecorder`(트레이스 축적), `DeepSupervisionHook`(TBPTT 윈도 내 사이클에서 판독 경로 호출), `AnytimeCurveHook`(평가 시 사이클별 정확도).

**핵심:** 깊은 감독은 손실 함수가 모듈 객체를 꺼내 쓰는 방식이 아니라, 훅이 **미리 판독해 값으로 넣어 주는** 방식이다 (ADR-005). 훅은 조립 루트에서 주입된 **단일** `ReadoutPath` 인스턴스를 쓴다 (I3).

### TBPTT 윈도가 공개 API인 이유
깊은 감독은 윈도 **안에서만** 샘플링해야 한다 — 윈도 밖 상태는 detach되어 있어 감독을 걸어도 엔진에 그래디언트가 가지 않고, 계산만 낭비하며 손실 스케일만 흔든다 (ADR-006). 따라서 `objectives`가 윈도 경계를 조회할 수 있어야 한다.

## 의존
`core`, `engine`, `termination`. (L2.)

## 레거시 참조
`v1.0:lsrr/iteration/controller.py` — **개작(3분할)**. 조기 종료 래칭 로직(`controller.py:129-141`: `newly_stopped` 마스크로 `R_final`을 고정하고 `stopping_cycles`를 기록)은 의미 그대로 `state.py`로 옮긴다. 반면 `run_eval`이 `fusion_head`·`decoder`를 인자로 받던 구조는 훅으로 뒤집는다 (ADR-005).

## 상태
**검증** — M3 완료. 러너·스케줄·TBPTT·래칭·훅 전부 구현.
