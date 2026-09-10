# 코드베이스 규약

> 컴포넌트화의 목적은 **재사용이 아니라 모듈화**다 — 부품을 갈아 끼우며 가설을 판정하기 위해, 그리고 한 부품의 실패가 다른 부품의 결과를 오염시키지 않게 하기 위해서다. 아래 규약은 전부 그 목적에 종속된다.

**최종 갱신:** 2026-09-10

---

## 1. 컴포넌트 규약

### 1.1 하나의 컴포넌트 = 하나의 인터페이스 + 하나의 레지스트리 키

```python
# lsrr/engine/core_hydra.py
from lsrr.core.interfaces import BaseRefinementEngine
from lsrr.core.registry import ENGINE_REGISTRY

@ENGINE_REGISTRY.register("hydra_qs")
class HydraQSEngine(BaseRefinementEngine):
    ...
```

- 등록 키는 **설정 파일에 쓰는 이름**이며, 클래스 이름과 독립적으로 안정적이어야 한다. 한 번 논문 표에 실린 키는 바꾸지 않는다.
- 별칭 등록은 허용하되(`@ENGINE_REGISTRY.register("hydra")`), 정본 키를 폴더 README에 명시한다.

### 1.2 컴포넌트는 자기보다 상위 레이어를 모른다

`ARCHITECTURE.md` §5의 레이어 규칙이 절대 기준이다. 구체 타입을 알아야 할 것 같으면 `core/types.py`의 데이터 타입으로 대화한다.

**금지 패턴:** 다른 컴포넌트 객체를 반환 딕셔너리에 실어 나르기(레거시 `model_outputs["fusion_head"]`). 필요한 것은 객체가 아니라 그 객체가 이미 계산한 값이며, 값을 넘길 수 없다면 훅(`recurrence/hooks.py`)으로 뒤집는다.

### 1.3 모든 슬롯은 무력화 구현을 갖는다

`identity` 어댑터, `none` 안정화, `mlp_onepass` 엔진처럼 "이 부품이 없을 때"를 같은 인터페이스로 표현할 수 있어야 절제 실험이 코드 분기 없이 성립한다 (ADR-009).

### 1.4 생성자는 설정만 받는다

컴포넌트 생성자는 스칼라·문자열·중첩 dict만 받는다. 다른 컴포넌트 인스턴스를 생성자로 받는 것은 조립 루트(`lsrr/builder.py`)와 명시적 래퍼(`engine/wrapper.py`)에만 허용한다.

### 1.5 배치 차원 규약

| 기호 | 의미 |
|---|---|
| `B` | 배치 |
| `L` | 레이어 축 길이 (백본 레이어 수, 범위 절제 시 부분집합) |
| `T` | 토큰 축 길이 |
| `d_in` | 백본 은닉 폭 |
| `d_model` | 사고 메모리 폭 (엔진 폭) |
| `M` | 정제 사이클 수 |
| `m` | 사이클 인덱스 (0-indexed) |

텐서 모양은 docstring 첫 줄에 `[B, L, d_model]` 형식으로 반드시 적는다.

---

## 2. 설정 규약

- 실험은 **설정 조합**이다. 새 실험을 위해 새 파이썬 파일을 만들어야 한다면 슬롯 설계가 틀린 것이다 (ADR-009).
- `configs/base.yaml` → 도메인별 조각(`backbone/`, `engine/`, …) → `exp/` 또는 `ablation/` 순으로 병합하고, CLI dotlist가 최우선이다.
- 모든 런은 **해석 완료된 설정 전체**를 `runs/<run_id>/config.yaml`에 스냅샷한다. 스냅샷의 해시가 런 신원이다.
- 설정 키는 슬롯 이름과 일치한다: `memory.scope.type`, `engine.type`, `termination.type`, `objective.deep_supervision.gamma`.
- **검증은 로드 시점에.** `config/validate.py`가 교차 제약(예: `M_max ≥ recurrence.schedule.max`, `readout.fusion_type=residual`이면 `W_r` 출력 폭 = `d_in`)을 확인하고 즉시 실패시킨다. 학습 3시간 뒤에 터지는 설정 오류는 없어야 한다.

---

## 3. 실패 규약

- **조용한 실패 금지.** 값이 없으면 추정하지 말고 던진다. 레거시가 잘한 지점을 계승한다 — 정답 없는 배치는 건너뛰지 않고 `MissingTargetError`, 백본 FLOPs를 못 재면 0으로 두지 않고 `flops_includes_backbone: false` 플래그.
- 예외는 `core/errors.py`의 타입을 쓴다. 일반 `ValueError`는 인자 검증에만.
- 불변식 위반은 경고가 아니라 예외다.

---

## 4. 테스트 규약

| 디렉터리 | 대상 | 규칙 |
|---|---|---|
| `tests/contracts/` | 불변식 I1–I8 | 불변식 1개당 최소 1개 테스트. `ARCHITECTURE.md` §4 표의 행과 1:1 대응 |
| `tests/unit/` | 개별 모듈 | 외부 의존 없이, 작은 텐서로 |
| `tests/integration/` | 슬롯 조합 | 조립 루트를 통과하는 최소 학습 스텝 |
| `tests/regression/` | 수치 재현성 | 시드 고정 비트 단위 동일성, 캐시 왕복 오차 |

- 새 불변식을 추가할 때는 **같은 커밋에** 계약 테스트를 넣는다.
- 테스트는 GPU 없이 CPU에서 통과해야 한다(작은 백본 스텁 허용).

---

## 5. 문서 규약

1. 폴더를 만들면 같은 커밋에서 그 폴더의 `README.md`를 만든다.
2. 폴더 README는 다음 절을 갖는다: **역할 / 경계(하는 일·하지 않는 일) / 모듈 목록 / 핵심 계약 / 의존 / 레거시 참조 / 상태**.
3. 모듈 표의 `상태` 열은 `계획` · `구현중` · `구현` · `검증`(테스트 통과) 중 하나다. 이 값은 `documents/MODULE_INDEX.md`와 항상 일치해야 한다.
4. 설계 변경은 코드보다 ADR이 먼저다. 코드에서 `# ADR-00N`으로 인용한다.
5. **측정 결과가 설계를 바꿨거나 위험을 드러냈으면 `documents/FINDINGS.md`에 남긴다.** 수치·측정 조건·재현 명령을 함께 적는다. 통과한 테스트를 나열하는 문서가 아니다 — 다시 재게 될 사실만 기록한다.
   - 그래디언트·노름 같은 양을 인용할 때는 **어느 지점에서 잰 것인지** 반드시 명시한다. 같은 실험의 두 숫자가 정반대 결론을 지지할 수 있다 (F-005).
6. 제안서 조항을 구현하는 코드에는 `# 제안서 §4.2`처럼 절 번호를 적는다. 나중에 제안서가 개정될 때 영향 범위를 grep으로 찾기 위함이다.

---

## 6. Legacy_LSRR 참조 정책

**`Legacy_LSRR/`은 읽기 전용이다. 어떤 파일도 수정·이동·삭제하지 않는다.**

- 참조는 자유롭게 한다. 재현 검증(수식이 맞는지, 엣지 케이스가 무엇이었는지)에 특히 유용하다.
- 코드를 가져올 때는 **이식(그대로) / 개작(수정) / 폐기(버림)** 중 하나로 판정하고 `documents/LEGACY_MAP.md`에 기록한다.
- 이식한 파일 상단에 출처를 남긴다: `# 이식: Legacy_LSRR/lsrr/engines/hydra_qs.py (개작: ...)`.
- 레거시의 `runs/`·`caches/` 산출물은 신규 코드베이스의 결과와 **섞어 보고하지 않는다**. 디코딩 경로가 다르므로(ADR-001) 비교 불가다.
- **폐기 시점은 정해져 있다.** 원본이 별도 경로에서 관리되므로 이 사본은 편하게 지운다 — M4 완료 후 `runs/`·`caches/`, M7 완료 후 폴더 전체. 다시 필요해지면 사용자에게 요청한다 (`ROADMAP.md` «Legacy_LSRR 폐기 계획»).

---

## 7. 명명 규약

- 패키지·모듈: `snake_case`, 복수형은 컬렉션일 때만(`objectives/`는 손실 여럿, `memory/`는 하나의 파이프라인).
- 클래스: `PascalCase`. 인터페이스는 `Base` 접두어.
- 레지스트리 키: `snake_case`, 논문 표기와 일치(`hydra_qs`, `delta_state`, `mlp_onepass`).
- 런 ID: `<exp>_<engine>_<backbone>_<YYYYMMDD_HHMMSS>_s<seed>` (레거시 규약 계승).
- 축 이름을 헷갈리게 쓰지 않는다: 레이어 축은 항상 `layer`/`l`, 사이클 축은 항상 `cycle`/`m`. `step`은 옵티마이저 스텝에만 쓴다.
