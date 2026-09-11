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
| `wrapper.py` | `EngineWrapper` — **위 갱신식의 유일한 소유자**. 감쇠·재주입·사이클 임베딩·pre-norm. 모든 코어를 감싼다 | 검증 |
| `scan.py` | 선택적 스캔 프리미티브와 quasiseparable shift — SSM 계열 코어의 공용 부품 | 검증 |
| `core_hydra.py` | quasiseparable 양방향 스캔 (Hydra) — **기본 구현** | 검증 |
| `core_mamba.py` | `mamba_up`(하→상) / `mamba_down`(상→하) / `bidir_add`(휴리스틱 양방향) | 검증 |
| `core_attention.py` | 동예산 어텐션 블록 — 내부 베이스라인 | 검증 |
| `core_mlp.py` | 1회 통과 MLP — Phase 0 게이트 ③의 비교 대상 | 검증 |
| `budget.py` | 파라미터 예산 정합 (5% 이내) | 검증 |

> `scan.py`는 원래 계획에 없던 모듈이다. `core_hydra`와 `core_mamba`가 같은 스캔을
> 쓰는데, 코어끼리 import하면 Ablation C의 비교 대상들이 서로 의존하게 되므로
> 공용 부품으로 분리했다. 레거시 `ssm_core.py`의 이식이다.

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
- `v1.0:lsrr/engines/wrapper.py` — 감쇠·사이클 임베딩·게이트 재주입·pre-norm이 §4.2의 "재귀 적응 장치" 3종에 정확히 대응한다.
- `v1.0:lsrr/engines/{hydra_qs,mamba_up_down,attn_block,mlp_onepass}.py`
- `v1.0:tests/{test_hydra_quasiseparable,test_engine_equiv,test_param_matching}.py` — HydraQS 순방향 분기가 Mamba-Up과 수치적으로 같아야 한다는 검사는 quasiseparable 구현의 정확성 근거다. 반드시 함께 이식한다.

## 상태
**검증** — 전 모듈 완료 (M4). Ablation C 스윕의 6개 엔진이 모두 등록되어 있고
(`hydra_qs`·`mamba_up`·`mamba_down`·`bidir_add`·`attn_block`·`mlp_onepass`),
이식 충실도는 `tests/unit/test_hydra_port_fidelity.py`가 고정한다.

FLOPs 정합은 아직 파라미터 정합만 구현되어 있다 — 어텐션의 이차 항은 L=12에서
파라미터 항에 묻히지만, 스케일 확장(M8) 시 `budget.py`에 FLOPs 축을 추가해야 한다.
