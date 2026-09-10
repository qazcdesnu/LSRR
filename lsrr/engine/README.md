# `lsrr/engine/` — 레이어 축 정제 연산자

## 역할
제안서 §4.2의 **사고 엔진**. 사고 메모리를 한 사이클만큼 정제한다.

$$R^{(m+1)} = (1-\alpha)\,R^{(m)} + \alpha\,S_\phi(R^{(m)}, R^{(0)}, m)$$

## 경계
- **한다:** 레이어 축 스캔(양방향 quasiseparable 기본), 감쇠 갱신, `R⁰` 재주입, 사이클 임베딩 조건화, 절제용 대안 연산자, 예산 정합.
- **하지 않는다:** 반복 자체를 돌리지 않는다. 몇 번 도는지, 언제 멈추는지, 언제 detach하는지는 전부 `recurrence`와 `termination`의 일이다. **엔진은 `m`을 조건 입력으로만 받고 루프를 소유하지 않는다.**

> 이 경계가 제안서 §4.2 "축의 구분"의 코드적 표현이다. 엔진이 루프를 갖는 순간 스캔 축과 반복 축이 한 파일에서 섞이고, 본 연구가 Huginn류와 다르다는 주장이 코드에서 보이지 않게 된다.

## 모듈

| 모듈 | 역할 | 상태 |
|---|---|---|
| `wrapper.py` | `EngineWrapper` — **위 갱신식의 유일한 소유자**. 감쇠·재주입·사이클 임베딩·pre-norm. 모든 코어를 감싼다 | 계획 |
| `core_hydra.py` | quasiseparable 양방향 스캔 (Hydra) — **기본 구현** | 계획 |
| `core_mamba.py` | `mamba_up`(하→상) / `mamba_down`(상→하) / `bidir_add`(휴리스틱 양방향) | 계획 |
| `core_attention.py` | 동FLOPs 어텐션 블록 — 내부 베이스라인 | 계획 |
| `core_mlp.py` | 1회 통과 MLP — Phase 0 게이트 ③의 비교 대상 | 계획 |
| `budget.py` | 파라미터·FLOPs 예산 정합 (5% 이내) | 계획 |

## 핵심 계약

```
EngineWrapper.forward_step(R_m, R0, m) -> R_next   # [B, L, d_model] 유지
```

모든 코어는 `[B, L, d_model] -> [B, L, d_model]` 순수 함수이며, 감쇠·재주입·사이클 조건화는 **코어가 아니라 래퍼가** 한다. 따라서 Ablation C(엔진 구조)와 Ablation D(재귀 적응 장치)가 서로 오염되지 않는다 — 코어를 바꿔도 갱신식은 동일하고, 갱신식을 바꿔도 코어는 동일하다.

### Ablation C의 공정성
비교 대상은 **믹서 행렬 클래스**다: dense(어텐션) / semiseparable(단방향 SSM) / quasiseparable(양방향 Hydra) / 대각(MLP, 사실상 믹싱 없음). 이 비교가 성립하려면 파라미터·FLOPs가 정합되어야 한다 (ADR-009). `budget.py`가 이를 보증하고, 정합 실패는 설정 로드 시점에 잡는다.

### Ablation D의 노브
`damping_alpha`(진동 억제 ↔ 표현력), `reinject_r0`(`none`/`add`/`concat`/`gate`), `cycle_embedding`(on/off). 앞의 둘은 대규모 루프 모델에서 검증된 레시피의 이식이다(§4.2).

### 양방향의 근거 (구현 시 유의)
레이어 축은 **완성된 궤적**이므로 인과 마스킹이 불필요하다. 정제는 필터링이 아니라 평활화(smoothing) 문제이며, 역방향 패스를 원리적으로 요구한다(칼만 평활기의 forward-backward와 동형). 그러므로 `bidir_add`(휴리스틱 결합)가 아니라 `hydra_qs`(quasiseparable 행렬 클래스)가 기본이다 — 휴리스틱 양방향의 열세가 보고되어 있다.

## 의존
`core`. (L1.)

## 레거시 참조
전부 **이식** 대상이며, 레거시에서 가장 완성도 높은 영역이다.
- `Legacy_LSRR/lsrr/engines/wrapper.py` — 감쇠·사이클 임베딩·게이트 재주입·pre-norm이 §4.2의 "재귀 적응 장치" 3종에 정확히 대응한다.
- `Legacy_LSRR/lsrr/engines/{hydra_qs,mamba_up_down,attn_block,mlp_onepass}.py`
- `Legacy_LSRR/tests/{test_hydra_quasiseparable,test_engine_equiv,test_param_matching}.py` — HydraQS 순방향 분기가 Mamba-Up과 수치적으로 같아야 한다는 검사는 quasiseparable 구현의 정확성 근거다. 반드시 함께 이식한다.

## 상태
**검증(부분)** — `wrapper.py`·`core_mlp.py` 완료 (M3). Hydra/Mamba/Attention 코어와 `budget.py`는 M4.
