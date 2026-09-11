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
| ④ | `<run>/gate_inputs.json` → `delta_trajectories` | 샘플별 Δ⁽ᵐ⁾ 궤적 |

## 사용

    python scripts/check_gates.py --runs-dir runs \
        --hydra-glob 'phase0_*hydra_qs*' --mlp-glob 'phase0_*mlp_onepass*'

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
    gate_beats_onepass,
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


def evaluate_kill_switch(
    runs_dir: Optional[Path],
    hydra_glob: Optional[str],
    mlp_glob: Optional[str],
    th: Phase0Thresholds,
) -> GateResult:
    if runs_dir is None or not hydra_glob or not mlp_glob:
        return _unavailable(
            "③", "H+MLP 1회 통과 대비 유의한 우위",
            "--runs-dir 와 --hydra-glob/--mlp-glob 이 필요하다", kill=True,
        )
    hydra = collect_seed_accuracy(runs_dir, hydra_glob)
    mlp = collect_seed_accuracy(runs_dir, mlp_glob)
    if len(hydra) < th.min_seeds or len(mlp) < th.min_seeds:
        return _unavailable(
            "③", "H+MLP 1회 통과 대비 유의한 우위",
            f"시드가 부족하다 (hydra {len(hydra)}개, mlp {len(mlp)}개, "
            f"조건당 {th.min_seeds}개 필요)",
            kill=True,
        )
    return gate_beats_onepass(
        hydra, mlp, th.alpha, th.min_effect_size, th.min_difference
    )


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, help="게이트 ①②④ 판정에 쓸 런 디렉터리")
    ap.add_argument("--runs-dir", type=Path, default=Path("runs"), help="게이트 ③의 런 모음")
    ap.add_argument("--hydra-glob", type=str, help="hydra_qs 런 glob (게이트 ③)")
    ap.add_argument("--mlp-glob", type=str, help="mlp_onepass 런 glob (게이트 ③)")
    ap.add_argument("--eps", type=float, default=1e-3, help="게이트 ④의 수렴 임계값")
    ap.add_argument("--json", type=Path, help="판정 결과를 기계 판독용 JSON 으로 저장")
    args = ap.parse_args(argv)

    th = Phase0Thresholds()
    inputs = _load_json(args.run / "gate_inputs.json") if args.run else None
    if args.run and inputs is None:
        print(f"[경고] {args.run}/gate_inputs.json 이 없다 — ①②④ 는 판정 불가다.", file=sys.stderr)

    results = [
        evaluate_collapse(inputs, th),
        evaluate_anytime(inputs, th),
        evaluate_kill_switch(args.runs_dir, args.hydra_glob, args.mlp_glob, th),
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
