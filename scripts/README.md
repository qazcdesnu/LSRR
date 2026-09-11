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
| `train.py` | 학습 | `runtime.trainer` | 검증 |
| `eval.py` | 평가·종료 규칙 스윕 | `runtime.evaluator`, `metrics` | 검증 |
| `sweep.py` | **YAML 주도** 스윕 (`--list`/`--count`/`--index`) | `config.sweep` | 검증 |
| `slurm/submit.sh`·`slurm/sweep_array.sh` | 배열 작업 제출 — 인덱스 1개 ↔ 자식 1개 | `sweep.py` | 검증 |
| `diagnose_injection.py` | 주입 통로 진단 (F-010) | `model`, `metrics` | 검증 |
| `diagnose_trajectory.py` | 궤적 분화 진단 (F-026) | `model`, `readout` | 검증 |
| `diagnose_state_scale.py` | 상태 스케일·β 포화 진단 (F-028) | `model`, `engine` | 검증 |
| `eval_backbone_direct.py` | No-CoT 백본 직접 평가 (하한 베이스라인) | `backbone`, `metrics` | 계획 |
| `profile_cost.py` | FLOPs·지연 프로파일 | `metrics.cost` | 계획 |
| `analyze.py` | 분석 플러그인 실행 | `analysis` | 계획 |
| `check_gates.py` | 단계 게이트 판정 (**킬 스위치 FAIL 시 비영 종료 코드**) | `gates` | 검증 |
| `make_report.py` | 표·그림 생성 | `reporting` | 계획 |

## 공통 규약
- 설정 지정: `exp=<name>` 또는 `config=<path>`, 이후 dotlist 오버라이드.
- **실험 조합은 CLI 가 아니라 설정의 `sweep:` 절이 정한다.** 조건을 플래그로 받으면 실제로 돌린 조합이 설정 파일에 남지 않아 재현이 셸 히스토리에 의존한다. 시드도 예외가 아니다.
- 런 디렉터리·배열 인덱스처럼 **저장 위치와 실행 방식**은 설정 키가 아니라 플래그다 (`--runs-dir`, `--index`). 설정 키로 두면 스냅샷에 들어가 같은 실험이 저장 위치로 신원이 갈린다.
- 모든 실행은 런 디렉터리를 만들고 설정 스냅샷을 남긴다.
- 실패는 조용히 넘기지 않는다 — 비영 종료 코드와 명확한 예외 메시지.

## 레거시 참조
`v1.0:scripts/` — **개작**. 스크립트 이름과 CLI 규약은 계승하되, 로직을 패키지로 옮기고 래퍼만 남긴다. `eval_backbone_direct.py`는 거의 그대로 이식.

## 상태
검증 — 진입점·진단·스윕·slurm 배선 완료. 나머지(`download_data`·`profile_cost`·`analyze`·`make_report`)는 계획
