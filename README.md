# LSRR — Layer-State Recurrent Reasoner

*Decoupling Transformer Context Encoding and SSM Reasoning with Convergence-based Termination*

동결된 트랜스포머를 **문맥 인코더로 단 1회** 실행해 레이어별 은닉 상태 전개 `H = [h⁽¹⁾…h⁽ᴸ⁾]`를 얻고, 분리된 **경량 양방향 SSM 사고 엔진**이 격자 밖의 정제 사이클 축에서 이를 반복 정제하며, 사고 메모리의 **수렴(Δ<ε)을 기준으로 자율 종료**한다.

> **현재 상태: M0 — 설계 확정, 구현 미착수.** 이 저장소에는 아직 실행 가능한 코드가 없다. 문서와 폴더 골격만 있다.

---

## 먼저 읽을 것

| 문서 | 내용 |
|---|---|
| [`documents/Research_Proposal_v2.1.md`](documents/Research_Proposal_v2.1.md) | **연구계획서 (정본)** — 무엇을 주장하고 무엇을 증명해야 하는가. 구판은 `documents/archive/` |
| [`documents/ARCHITECTURE.md`](documents/ARCHITECTURE.md) | **시스템 아키텍처 — 여기서 시작** |
| [`documents/MODULE_INDEX.md`](documents/MODULE_INDEX.md) | 모듈 인덱스 (지속 갱신 상태판) |
| [`documents/DESIGN_DECISIONS.md`](documents/DESIGN_DECISIONS.md) | 설계 결정 기록 (ADR) |
| [`documents/FINDINGS.md`](documents/FINDINGS.md) | **실측 기록** — 구현 중 측정된 사실과 미해소 위험 |
| [`documents/CONVENTIONS.md`](documents/CONVENTIONS.md) | 코드베이스 규약 |
| [`documents/LEGACY_MAP.md`](documents/LEGACY_MAP.md) | v1.0 참조 지도 — 무엇을 이식/개작/폐기했는가 |
| [`documents/ROADMAP.md`](documents/ROADMAP.md) | 구현 로드맵과 게이트 |

각 폴더의 `README.md`가 그 폴더의 역할·경계·모듈·계약을 설명한다.

---

## 파이프라인 한눈에

```
질문 → [동결 백본 1회] → H_last, H_pool, h_ctx, KV
     → [memory] R⁰
     → [recurrence × engine] R^(m+1) = (1-α)R^(m) + α·S_φ(R^(m), R⁰, m)
        ↳ [termination] Δ<ε 또는 M_max
     → [readout] h_fusion = h_ctx + W_r·Σ α_l r_l*
     → [백본 연속 디코딩] 답 토큰 자기회귀 생성
```

---

## 폴더

| 경로 | 역할 |
|---|---|
| `documents/` | 연구 계획·아키텍처·결정 기록 |
| `lsrr/` | 구현 패키지 ([폴더 역할표](documents/ARCHITECTURE.md#6-폴더별-역할-최상위)) |
| `configs/` | 설정 트리 — 실험은 설정 조합이다 |
| `scripts/` | CLI 진입점 (얇은 래퍼) |
| `tests/` | 계약·단위·통합·회귀 테스트 |
| `runs/`, `caches/` | 산출물 (버전 관리 제외) |

---

## 지켜야 할 것

1. **작업은 `v1.1` 브랜치에 한다.** `main` 은 공개 기준점이며, 큰 진전마다 fast-forward 로 병합한다. 레거시(v1.0) 참조는 `git show v1.0:<경로>` 로 하고, 가져온 것은 `LEGACY_MAP.md`에 이식/개작/폐기 판정과 함께 기록한다 (`CONVENTIONS.md` §6·§8).
2. **폴더를 만들면 같은 커밋에 그 폴더의 `README.md`를 만든다.** 모듈을 바꾸면 폴더 README와 `MODULE_INDEX.md`를 함께 갱신한다.
3. **설계 변경은 코드보다 ADR이 먼저다.**
4. **불변식 I1–I8**([ARCHITECTURE §4](documents/ARCHITECTURE.md#4-불변식-invariants))은 계약 테스트로 강제한다. 테스트 없는 불변식은 존재하지 않는 것으로 간주한다.
5. **Phase 0 게이트 ③은 킬 스위치다.** 실패 시 다음 마일스톤으로 진행하지 않는다.
