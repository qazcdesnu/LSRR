# `lsrr/termination/` — 수렴 기반 자율 종료

## 역할
제안서 §4.3. 매 사이클 메모리 변화량을 측정해 **추가 반복이 의미 있는 변화를 만들지 못할 때** 사고를 종료한다.

$$\Delta^{(m)} = \frac{1}{L}\sum_l \lVert r_l^{(m+1)} - r_l^{(m)}\rVert_2 < \varepsilon \quad (\text{폴백: } m = M_{max})$$

## 경계
- **한다:** 종료 신호 계산, 규칙 판정, ε 보정, 수렴 거동 분류.
- **하지 않는다:** 루프를 돌지 않는다(→ `recurrence`). 수렴을 *강제*하지 않는다(→ `stability`). 이 패키지는 **관측하고 판정할 뿐**이다.

> 본 설계의 정확한 지위: **'수렴을 보장하는 구조'가 아니라 '수렴을 유도하는 학습 체계'**(§4.3). 유도의 재료는 M 무작위 샘플링(`recurrence`), 깊은 감독(`objectives`), 감쇠 갱신(`engine`)이며 — 이 패키지는 그 결과를 측정한다. 유도가 실패했을 때의 처방은 `stability`가 갖는다.

## 모듈

| 모듈 | 역할 | 상태 |
|---|---|---|
| `base.py` | 규칙 기반 클래스 — **`M_max` 폴백을 여기서 강제** (I5). 하위 클래스는 폴백을 우회할 수 없다 | 계획 |
| `signals.py` | 신호 계산 공통화: 상태 Δ(L2), 출력 KL, 출력 엔트로피 | 계획 |
| `rules.py` | `fixed_m` / `delta_state`(기본) / `kl_output` / `entropy_output` — **Ablation B** | 계획 |
| `calibration.py` | ε 스윕, 검증셋 기반 임계값 선택, 파레토 지원 | 계획 |
| `behavior.py` | 수렴 거동 분류: `converged` / `oscillating` / `drifting` — **게이트 ④** | 검증 |

## 핵심 계약

```
should_stop(signals: TerminationSignals, m: int) -> (stop_mask [B], CycleDiagnostics)
```

- 판정은 **샘플별**이다. 배치 평균으로 정지를 결정하면 "문제 난이도에 따른 적응적 계산"이라는 주장이 성립하지 않는다.
- 규칙이 출력 공간 신호(KL·엔트로피)를 쓰려면 사이클별 로짓이 필요하다 → `recurrence`의 판독 훅이 켜져 있어야 하며, `config/validate.py`가 이 조합을 검증한다.
- **모든 규칙은 `m + 1 >= M_max`에서 정지한다.** 이는 `base.py`가 하위 클래스의 판정 결과에 OR로 합성하므로 우회 불가다.

### Ablation B가 묻는 것
종료 신호를 어느 공간에서 재야 하는가 — 상태 공간 Δ / 출력 공간 KL / 엔트로피 / 고정 M. 정확도-연산 파레토에서 수렴 종료가 우위인지를 판정한다.

### `behavior.py` — 문헌의 공백을 메우는 실험
루프 모델에서 일부 상태가 수렴 대신 **진동(궤도)**하는 사례가 보고되어 있다. 거동별로 종료 시점 민감도를 층화 분석하는 것이 제안서 §7이 명시한 기여 실험이다. 분류 기준:
- `converged` — Δ 궤적이 단조 감소하고 ε 아래로 안착
- `oscillating` — Δ가 주기적으로 반등, 자기상관 유의
- `drifting` — Δ가 ε 위에서 감소하지 않고 상태 노름이 계속 변함

이 라벨은 `analysis`와 `reporting`에서 층화 축으로 쓰인다.

## 의존
`core`. (L1 — `recurrence`가 이 패키지를 호출하지, 그 반대가 아니다.)

## 레거시 참조
`Legacy_LSRR/lsrr/termination/rules.py` — **개작**. `fixed_m`·`delta_state`·`kl_output` 구현은 이식하되 (i) 신호 계산을 `signals.py`로 공통화하고 (ii) `M_max` 폴백을 각 규칙에 흩어 두는 대신 `base.py`로 승격한다(레거시는 규칙마다 `fallback = (m + 1 >= self.m_max)`를 반복했다 — 새 규칙을 추가할 때 빠뜨리기 쉬운 형태다). `entropy_output`과 `behavior.py`는 신규.

## 상태
**검증(부분)** — 규칙 4종·신호·M_max 폴백 기반 완료 (M3). `calibration.py`·`behavior.py`는 M6.
