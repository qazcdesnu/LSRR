"""Phase 0 게이트 판정 (ADR-008, M5 완료 조건).

게이트는 **실험 전에 코드로 고정된 기준**이다. 이 파일은 그 기준이 의도대로
동작하는지, 특히 다음 세 가지를 고정한다.

- 판정 데이터가 없을 때 통과하지 않는다 (없는 것 = 통과 는 게이트를 무의미하게 만든다)
- 킬 스위치는 **측정된** 실패에서만 발동한다 (데이터 부족은 킬 스위치가 아니다)
- 게이트 하나라도 빠뜨린 채 PASS 가 나오지 않는다
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from lsrr.analysis.collapse import CollapseReport, diagnose
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
from lsrr.termination.behavior import classify_batch, classify_trajectory

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


# ---------------------------------------------------------------- ① 붕괴

def test_healthy_memory_passes_collapse_gate():
    torch.manual_seed(0)
    R = torch.randn(4, 12, 32)
    assert gate_no_collapse(diagnose(R, R0=R)).passed


@pytest.mark.parametrize("mode", ["identical", "scaled"])
def test_collapsed_memory_fails_collapse_gate(mode):
    """서로 다른 붕괴 양식을 각각 잡는다."""
    torch.manual_seed(0)
    base = torch.randn(4, 1, 32)
    if mode == "identical":
        R = base.expand(4, 12, 32).contiguous() + torch.randn(4, 12, 32) * 1e-4
    else:  # 방향은 같고 크기만 다름
        R = (base * torch.linspace(0.2, 2.0, 12).view(1, 12, 1)).expand(4, 12, 32).contiguous()
    result = gate_no_collapse(diagnose(R, R0=torch.randn(4, 12, 32)))
    assert not result.passed
    assert result.detail.startswith("붕괴 징후")


def test_effective_rank_is_not_centred():
    """평균을 빼면 '모든 레이어가 같은 벡터' 붕괴에서 잔여 잡음의 랭크를 잰다."""
    torch.manual_seed(0)
    collapsed = torch.randn(2, 1, 16).expand(2, 12, 16).contiguous() + torch.randn(2, 12, 16) * 1e-5
    assert diagnose(collapsed).effective_rank < 2.0


# ---------------------------------------------------------------- ② anytime

def test_rising_curve_passes():
    curve = build_curve({0: 0.10, 1: 0.18, 2: 0.24, 3: 0.29})
    assert gate_anytime_increasing(curve).passed


def test_flat_curve_fails():
    curve = build_curve({0: 0.22, 1: 0.221, 2: 0.219, 3: 0.222})
    assert not gate_anytime_increasing(curve).passed


def test_noisy_but_rising_curve_still_passes():
    """사이클 하나가 잡음으로 내려앉는 것과 곡선이 평평한 것은 다른 문제다."""
    curve = build_curve({0: 0.10, 1: 0.17, 2: 0.15, 3: 0.24, 4: 0.28})
    assert gate_anytime_increasing(curve).passed


def test_single_point_curve_cannot_pass():
    assert not gate_anytime_increasing(build_curve({0: 0.9})).passed


# ---------------------------------------------------------------- ③ 킬 스위치

def test_significant_advantage_passes():
    r = gate_beats_onepass([0.315, 0.298, 0.324], [0.221, 0.213, 0.235])
    assert r.passed and r.is_kill_switch


def test_no_advantage_triggers_kill_switch():
    r = gate_beats_onepass([0.224, 0.219, 0.231], [0.221, 0.226, 0.218])
    assert not r.passed and r.is_kill_switch and r.evaluated


def test_worse_than_baseline_never_passes():
    """대조가 더 좋으면 p 값이 어떻든 통과하지 않는다 (단측 판정)."""
    r = gate_beats_onepass([0.10, 0.11, 0.09], [0.30, 0.31, 0.29])
    assert not r.passed


def test_single_seed_is_not_a_verdict():
    """단일 시드 판정을 허용하면 잡음으로 킬 스위치가 통과할 수 있다."""
    r = gate_beats_onepass([0.9], [0.1])
    assert not r.passed
    assert not r.evaluated, "데이터 부족은 '측정된 실패'가 아니다"


# ---------------------------------------------------------------- ④ 거동

def test_decaying_trajectory_counts_even_if_not_settled():
    """M_max 가 짧아 ε 위에서 끝나도 궤적의 모양은 수렴형이다."""
    traj = [2.85 / (2**m) for m in range(10)]
    report = classify_trajectory(traj, eps=1e-3)
    assert report.label == "converged"
    assert not report.settled
    assert report.macroscopically_decreasing


def test_oscillating_trajectory_is_labelled():
    import math

    traj = [1.0 + 0.4 * math.sin(m * 2.1) for m in range(12)]
    assert classify_trajectory(traj, eps=1e-3).label == "oscillating"


def test_flat_trajectory_drifts():
    assert classify_trajectory([1.0] * 10, eps=1e-3).label == "drifting"


def test_eps_does_not_change_the_shape_label():
    """ε 스윕(M6)이 성립하려면 라벨이 ε 에 흔들리면 안 된다."""
    traj = [2.85 / (2**m) for m in range(10)]
    labels = {classify_trajectory(traj, eps=e).label for e in (1e-1, 1e-3, 1e-6)}
    assert labels == {"converged"}


def test_empty_trajectory_is_fatal():
    with pytest.raises(ValueError):
        classify_trajectory([], eps=1e-3)


def test_delta_gate_needs_trajectories():
    assert not gate_delta_decreasing([]).passed
    assert not gate_delta_decreasing([]).evaluated


def test_delta_gate_on_mixed_population():
    import math

    good = [[2.0 / (2**m) for m in range(8)] for _ in range(7)]
    bad = [[1.0 + 0.4 * math.sin(m * 2.1) for m in range(8)] for _ in range(3)]
    r = gate_delta_decreasing(classify_batch(good + bad, eps=1e-3))
    assert r.passed  # 0.7 >= 0.6
    assert r.evidence["counts"]["oscillating"] == 3


# ---------------------------------------------------------------- 보고서

def _four(passed: bool) -> list[GateResult]:
    return [
        GateResult(g, f"게이트{g}", passed, "", is_kill_switch=(g == "③"))
        for g in ("①", "②", "③", "④")
    ]


def test_report_requires_every_gate():
    """일부만 평가하고 PASS 를 내면 게이트가 무의미해진다."""
    with pytest.raises(ValueError, match="판정 결과가 없다"):
        build_report(PHASE_0, _four(True)[:3])


def test_all_pass_is_pass():
    assert build_report(PHASE_0, _four(True)).passed


def test_unevaluated_gate_is_not_a_pass():
    results = _four(True)
    results[0] = GateResult("①", "게이트①", False, "판정 불가", evaluated=False)
    report = build_report(PHASE_0, results)
    assert not report.passed
    assert report.incomplete[0].gate_id == "①"
    assert not report.kill_switch_triggered


def test_kill_switch_needs_a_measured_failure():
    results = _four(True)
    results[2] = GateResult(
        "③", "게이트③", False, "판정 불가", is_kill_switch=True, evaluated=False
    )
    assert not build_report(PHASE_0, results).kill_switch_triggered

    results[2] = GateResult("③", "게이트③", False, "측정됨", is_kill_switch=True)
    assert build_report(PHASE_0, results).kill_switch_triggered


def test_render_names_the_review_order_on_kill():
    results = _four(True)
    results[2] = GateResult("③", "게이트③", False, "측정됨", is_kill_switch=True)
    text = build_report(PHASE_0, results).render()
    assert "킬 스위치 발동" in text
    assert "깊은 감독 과잉" in text


# ---------------------------------------------------------------- CLI

def _write_run(tmp_path: Path, *, collapse, anytime, traj, hydra, mlp) -> Path:
    runs = tmp_path / "runs"
    (runs / "r1").mkdir(parents=True)
    (runs / "r1" / "gate_inputs.json").write_text(
        json.dumps({"collapse": collapse, "anytime": anytime, "delta_trajectories": traj}),
        encoding="utf-8",
    )
    for i, a in enumerate(hydra):
        d = runs / f"phase0_hydra_qs_s{i}"
        d.mkdir()
        (d / "metrics.json").write_text(json.dumps({"accuracy": a}))
    for i, a in enumerate(mlp):
        d = runs / f"phase0_mlp_onepass_s{i}"
        d.mkdir()
        (d / "metrics.json").write_text(json.dumps({"accuracy": a}))
    return runs


def _run_cli(runs: Path, out: Path) -> tuple[int, str]:
    proc = subprocess.run(
        [
            sys.executable, str(REPO_ROOT / "scripts" / "check_gates.py"),
            "--run", str(runs / "r1"), "--runs-dir", str(runs),
            "--hydra-glob", "phase0_hydra_qs_*", "--mlp-glob", "phase0_mlp_onepass_*",
            "--json", str(out),
        ],
        capture_output=True, text=True, cwd=str(REPO_ROOT),
    )
    return proc.returncode, proc.stdout


def test_cli_reports_pass(tmp_path):
    runs = _write_run(
        tmp_path,
        collapse={"effective_rank": 8.4, "mean_similarity": 0.21, "variance_ratio": 0.83},
        anytime={"0": 0.11, "1": 0.19, "2": 0.24, "3": 0.31},
        traj=[[2.85 / (2**m) for m in range(10)] for _ in range(20)],
        hydra=[0.315, 0.298, 0.324],
        mlp=[0.221, 0.213, 0.235],
    )
    code, out = _run_cli(runs, tmp_path / "v.json")
    assert code == 0, out
    assert "판정: PASS" in out
    verdict = json.loads((tmp_path / "v.json").read_text(encoding="utf-8"))
    assert verdict["passed"] and not verdict["kill_switch_triggered"]


def test_cli_signals_kill_switch_with_exit_code_2(tmp_path):
    import math

    runs = _write_run(
        tmp_path,
        collapse={"effective_rank": 1.4, "mean_similarity": 0.99, "variance_ratio": 0.02},
        anytime={"0": 0.22, "1": 0.221, "2": 0.219, "3": 0.222},
        traj=[[1.0 + 0.4 * math.sin(m * 2.1) for m in range(12)] for _ in range(10)],
        hydra=[0.224, 0.219, 0.231],
        mlp=[0.221, 0.226, 0.218],
    )
    code, out = _run_cli(runs, tmp_path / "v.json")
    assert code == 2, out
    assert "킬 스위치 발동" in out
    assert json.loads((tmp_path / "v.json").read_text(encoding="utf-8"))["kill_switch_triggered"]


def test_cli_without_data_is_incomplete_not_pass(tmp_path):
    (tmp_path / "runs").mkdir()
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "check_gates.py"),
         "--runs-dir", str(tmp_path / "runs")],
        capture_output=True, text=True, cwd=str(REPO_ROOT),
    )
    assert proc.returncode == 3, proc.stdout
    assert "데이터 없음은 통과가 아니다" in proc.stdout


def test_thresholds_are_overridable_but_default_is_canonical():
    assert Phase0Thresholds().min_seeds == 3
    cfg = {"gates": {"phase0": {"alpha": 0.01, "min_seeds": 5}}}
    th = Phase0Thresholds.from_config(cfg)
    assert th.alpha == 0.01 and th.min_seeds == 5
    assert th.min_converged_ratio == Phase0Thresholds().min_converged_ratio


# ---------------------------------------------------------------- ③ 절대 차이 하한 (F-016)

def test_tiny_difference_cannot_pass_the_kill_switch():
    """효과크기는 비율이라, 시드가 일관되면 0.3%p 차이도 d가 커진다.

    킬 스위치가 그런 차이로 통과하면 "반복이 기여한다"는 주장이 사실상 검증 없이
    통과한다. 절대 차이 하한이 그것을 막는다.
    """
    import math

    hydra = [0.2247 + 0.006 * math.sin(i) for i in range(3)]
    mlp = [0.2217 + 0.004 * math.cos(i) for i in range(3)]
    r = gate_beats_onepass(hydra, mlp)
    assert r.evidence["p_value"] < 0.05, "p값 자체는 유의하다"
    assert r.evidence["effect_size"] > 0.5, "효과크기 자체도 기준을 넘는다"
    assert not r.passed, "그럼에도 절대 차이가 작아 통과하면 안 된다"
    assert "기준 ≥0.01" in r.detail


def test_meaningful_difference_still_passes():
    assert gate_beats_onepass([0.315, 0.298, 0.324], [0.221, 0.213, 0.235]).passed


def test_min_difference_is_configurable():
    hydra, mlp = [0.23, 0.232, 0.229], [0.221, 0.223, 0.220]
    assert not gate_beats_onepass(hydra, mlp, min_difference=0.05).passed
    assert gate_beats_onepass(hydra, mlp, min_difference=0.005).passed
