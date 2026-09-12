#!/usr/bin/env python
"""단계 게이트 판정 — M5의 완료 조건 (ADR-008, 제안서 §6.3).

    "게이트 ③이 실패하면 반복의 기여가 없다는 뜻이므로 아키텍처를 재검토한다."

**이 스크립트는 실험을 돌리지 않는다.** 런 디렉터리에 이미 기록된 산출물만 읽어
판정한다. 판정과 실행을 분리해야, 실패한 실험 앞에서 기준을 조금 낮추는 일이
구조적으로 불가능해진다.

## 입력

게이트마다 필요한 파일이 다르다. 없으면 그 게이트는 **FAIL**이며, 이유를
"판정 불가"로 명시한다 — 없는 것을 통과로 두면 게이트가 무의미해진다.

| 게이트 | 필요한 파일 | 내용 |
|---|---|---|
| ① | `<run>/gate_inputs.json` → `collapse` | `effective_rank`, `mean_similarity`, `variance_ratio` |
| ② | `<run>/gate_inputs.json` → `anytime` | `{사이클: 정확도}` |
| ③ | 각 조건의 여러 런 → `metrics.json` → `accuracy` | 시드별 최종 정확도 |

게이트 ③의 판정 대상은 v2에서 **방출 구조**다 (ADR-016): 처치군은 궤적 방출
(`dynamic_m`), 대조군은 단일 벡터(v1, `single_v1`). 엔진 구조 비교는 Ablation C
로 격하됐으므로 인자 이름도 방출 구조 중립적이다.
| ④ | `<run>/gate_inputs.json` → `delta_trajectories` | 샘플별 Δ⁽ᵐ⁾ 궤적 |

## 사용

    # 1단 (Phase A 기준선): ③을 측정하되 킬 스위치로 취급하지 않는다
    python scripts/check_gates.py --stage baseline --runs-dir runs \
        --treatment-glob '*condition_dynamic_m*' \
        --control-glob '*condition_single_v1*'

    # 2단 (Phase B): ③이 킬 스위치다 (기본)
    python scripts/check_gates.py --stage kill --runs-dir runs ...

    python scripts/check_gates.py --run runs/<run_id>          # ①②④만
    python scripts/check_gates.py ... --json out.json          # 기계 판독용
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lsrr.analysis.collapse import CollapseReport
from lsrr.gates import (
    PHASE_0,
    GateResult,
    Phase0Thresholds,
    build_report,
    gate_anytime_increasing,
    gate_beats_baseline,
    gate_delta_decreasing,
    gate_no_collapse,
)
from lsrr.metrics.anytime import build_curve
from lsrr.termination.behavior import classify_batch


def _load_json(path: Path) -> Optional[dict[str, Any]]:
    if not path.exists():
        return None
    try:
        with path.open(encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        print(f"[경고] {path} 를 읽지 못했다: {e}", file=sys.stderr)
        return None


def _unavailable(gate_id: str, name: str, why: str, kill: bool = False) -> GateResult:
    """판정 데이터가 없을 때. 통과가 아니라 FAIL이다."""
    return GateResult(
        gate_id=gate_id, name=name, passed=False,
        detail=f"판정 불가: {why}", is_kill_switch=kill, evaluated=False,
    )


def evaluate_collapse(inputs: Optional[dict], th: Phase0Thresholds) -> GateResult:
    node = (inputs or {}).get("collapse")
    if not node:
        return _unavailable("①", "자명해 붕괴 없음", "gate_inputs.json 에 collapse 가 없다")
    report = CollapseReport(
        effective_rank=float(node["effective_rank"]),
        layer_variance=float(node.get("layer_variance", 0.0)),
        mean_similarity=float(node["mean_similarity"]),
        variance_ratio=(
            float(node["variance_ratio"]) if node.get("variance_ratio") is not None else None
        ),
    )
    return gate_no_collapse(
        report, th.min_effective_rank, th.max_mean_similarity, th.min_variance_ratio
    )


def evaluate_anytime(inputs: Optional[dict], th: Phase0Thresholds) -> GateResult:
    node = (inputs or {}).get("anytime")
    if not node:
        return _unavailable("②", "M 증가 → 정확도 증가", "gate_inputs.json 에 anytime 이 없다")
    curve = build_curve({int(k): float(v) for k, v in node.items()})
    return gate_anytime_increasing(curve, th.min_improvement, th.min_trend)


def evaluate_delta(inputs: Optional[dict], eps: float, th: Phase0Thresholds) -> GateResult:
    node = (inputs or {}).get("delta_trajectories")
    if not node:
        return _unavailable("④", "Δ 궤적의 거시적 감소", "gate_inputs.json 에 delta_trajectories 가 없다")
    return gate_delta_decreasing(classify_batch(node, eps), th.min_converged_ratio)


def collect_seed_accuracy(runs_dir: Path, pattern: str) -> list[float]:
    """`<runs>/<pattern>/metrics.json` 의 `accuracy` 를 모은다."""
    values = []
    for run in sorted(runs_dir.glob(pattern)):
        metrics = _load_json(run / "metrics.json")
        if metrics is None:
            continue
        acc = metrics.get("accuracy", metrics.get("val_accuracy"))
        if acc is not None:
            values.append(float(acc))
    return values


#: 게이트 ③의 이름. v1 은 "H+MLP 1회 통과 대비", v2 는 방출 구조다 (ADR-016).
_KILL_NAME = "단일 벡터(v1) 대비 궤적 방출의 유의한 우위"


def evaluate_kill_switch(
    runs_dir: Optional[Path],
    treatment_glob: Optional[str],
    control_glob: Optional[str],
    th: Phase0Thresholds,
    stage: str = "kill",
) -> GateResult:
    """게이트 ③. `stage` 가 판정의 **역할**을 정한다 (ADR-016 재개정).

    - `kill`     2단(Phase B). 측정된 FAIL 이 킬 스위치를 발동한다.
    - `baseline` 1단(Phase A). 같은 술어·같은 임계값으로 **측정하되 판정하지
                 않는다.** 학습되지 않은 수신기 앞에서는 궤적 방출이 이길 이유가
                 없으므로, 여기서의 FAIL 은 수신 병목의 비용이지 판정이 아니다.
    """
    kill = stage == "kill"
    name = _KILL_NAME if kill else _KILL_NAME + " (기준선 — 판정 아님)"
    if runs_dir is None or not treatment_glob or not control_glob:
        return _unavailable(
            "③", name,
            "--runs-dir 와 --treatment-glob/--control-glob 이 필요하다", kill=kill,
        )
    treatment = collect_seed_accuracy(runs_dir, treatment_glob)
    control = collect_seed_accuracy(runs_dir, control_glob)
    if len(treatment) < th.min_seeds or len(control) < th.min_seeds:
        return _unavailable(
            "③", name,
            f"시드가 부족하다 (처치 {len(treatment)}개, 대조 {len(control)}개, "
            f"조건당 {th.min_seeds}개 필요)",
            kill=kill,
        )
    result = gate_beats_baseline(
        treatment, control, th.alpha, th.min_effect_size, th.min_difference
    )
    if kill:
        return result
    # 기준선: 수치는 그대로, 킬 스위치 표시만 뗀다. 술어를 바꾸지 않으므로
    # 1단 수치를 2단과 같은 형식으로 나란히 놓을 수 있다.
    return GateResult(
        gate_id=result.gate_id, name=name, passed=result.passed,
        detail=result.detail, is_kill_switch=False,
        evaluated=result.evaluated, evidence=result.evidence,
    )


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, help="게이트 ①②④ 판정에 쓸 런 디렉터리")
    ap.add_argument("--runs-dir", type=Path, default=Path("runs"), help="게이트 ③의 런 모음")
    ap.add_argument("--treatment-glob", type=str, help="처치군 런 glob — 궤적 방출 (게이트 ③)")
    ap.add_argument("--control-glob", type=str, help="대조군 런 glob — 단일 벡터 v1 (게이트 ③)")
    # ④의 ε 은 설정의 termination.eps 와 같은 값이어야 한다. Δ 정의가 v2.1 에서
    # 바뀌었고(F-024) ADR-017 이 상태를 RMS 1 에 묶었으므로 v1 의 1e-3 은 무의미하다.
    ap.add_argument("--eps", type=float, default=0.1, help="게이트 ④의 수렴 임계값")
    ap.add_argument(
        "--stage", choices=("baseline", "kill"), default="kill",
        help="③의 역할. baseline=1단(Phase A, 측정만) / kill=2단(Phase B, 킬 스위치). ADR-016",
    )
    ap.add_argument("--json", type=Path, help="판정 결과를 기계 판독용 JSON 으로 저장")
    args = ap.parse_args(argv)

    th = Phase0Thresholds()
    inputs = _load_json(args.run / "gate_inputs.json") if args.run else None
    if args.run and inputs is None:
        print(f"[경고] {args.run}/gate_inputs.json 이 없다 — ①②④ 는 판정 불가다.", file=sys.stderr)

    results = [
        evaluate_collapse(inputs, th),
        evaluate_anytime(inputs, th),
        evaluate_kill_switch(
            args.runs_dir, args.treatment_glob, args.control_glob, th, stage=args.stage
        ),
        evaluate_delta(inputs, args.eps, th),
    ]
    report = build_report(PHASE_0, results)
    print(report.render())

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        with args.json.open("w", encoding="utf-8") as f:
            json.dump(report.as_dict(), f, indent=2, ensure_ascii=False)
        print(f"\n[판정] JSON 저장: {args.json}")

    # 종료 코드로도 신호한다 — CI 와 스크립트가 쓸 수 있게.
    #   0 PASS / 1 FAIL / 2 킬 스위치 발동 / 3 판정 데이터 부족
    if report.kill_switch_triggered:
        return 2
    if report.passed:
        return 0
    return 1 if report.failed else 3


if __name__ == "__main__":
    sys.exit(main())
