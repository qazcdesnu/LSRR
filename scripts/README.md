# `scripts/` — CLI 진입점

## 역할
명령줄에서 무언가를 실행하는 얇은 래퍼.

## 경계
- **한다:** 인자 파싱 → 설정 로드 → 패키지 함수 호출 → 종료 코드 반환.
- **하지 않는다:** **로직을 담지 않는다.** 스크립트에 알고리즘이 들어가면 테스트할 수 없고 재사용할 수 없다. 스크립트가 길어지면 그 내용은 어느 패키지의 역할인지 다시 묻는다.

## 스크립트

| 스크립트 | 역할 | 호출하는 패키지 | 상태 |
|---|---|---|---|
| `download_data.py` | 데이터 획득·SHA-256 검증 (`--dataset`, `--check`, `--force`) | `data.download` | 계획 |
| `extract_h.py` | H 캐시 사전 추출 (보조 경로, ADR-002) | `backbone`, `data.cache` | 계획 |
| `train.py` | 학습 | `runtime.trainer` | 계획 |
| `eval.py` | 평가·종료 규칙 스윕 | `runtime.evaluator`, `metrics` | 계획 |
| `sweep.py` | 그리드 스윕 | `config.sweep` | 계획 |
| `eval_backbone_direct.py` | No-CoT 백본 직접 평가 (하한 베이스라인) | `backbone`, `metrics` | 계획 |
| `profile_cost.py` | FLOPs·지연 프로파일 | `metrics.cost` | 계획 |
| `analyze.py` | 분석 플러그인 실행 | `analysis` | 계획 |
| `check_gates.py` | 단계 게이트 판정 (**킬 스위치 FAIL 시 비영 종료 코드**) | `gates` | 계획 |
| `make_report.py` | 표·그림 생성 | `reporting` | 계획 |

## 공통 규약
- 설정 지정: `exp=<name>` 또는 `config=<path>`, 이후 dotlist 오버라이드.
- 모든 실행은 런 디렉터리를 만들고 설정 스냅샷을 남긴다.
- 실패는 조용히 넘기지 않는다 — 비영 종료 코드와 명확한 예외 메시지.

## 레거시 참조
`Legacy_LSRR/scripts/` — **개작**. 스크립트 이름과 CLI 규약은 계승하되, 로직을 패키지로 옮기고 래퍼만 남긴다. `eval_backbone_direct.py`는 거의 그대로 이식.

## 상태
계획 — M2부터 점진 추가
