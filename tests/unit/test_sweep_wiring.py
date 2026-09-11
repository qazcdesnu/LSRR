"""스윕 배선 — 조건 × 시드를 빠짐없이 돌리고, 판정은 하지 않는다 (ADR-008).

손으로 6런을 돌리면 조건을 빠뜨리거나 시드를 섞기 쉽다. 이 파일이 고정하는 것:

- 조합이 엔진 × 시드의 전수여야 한다
- 런 디렉터리를 **기계 판독용 한 줄**로 주고받는다 (사람이 읽는 문구 파싱 금지)
- 스윕은 게이트를 판정하지 않는다
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.sweep import _already_done, _parse_list, _run_dir_from  # noqa: E402


def test_run_dir_is_parsed_from_a_machine_readable_line():
    out = "학습 파라미터 5,387,777 / 백본 …\n완료: 16 스텝 → runs/x\nRUN_DIR=runs/abc_s0\n"
    assert _run_dir_from(out) == Path("runs/abc_s0")


def test_run_dir_takes_the_last_marker():
    """자식이 여러 줄을 낼 수 있으므로 마지막 것이 유효하다."""
    out = "RUN_DIR=runs/a\nRUN_DIR=runs/b\n"
    assert _run_dir_from(out) == Path("runs/b")


def test_missing_marker_is_not_guessed():
    """문구를 추측해 파싱하면 출력을 고칠 때 스윕이 조용히 깨진다."""
    assert _run_dir_from("완료: 16 스텝 → runs/abc_s0\n") is None


def test_parse_list_handles_spacing():
    assert _parse_list("a, b ,c") == ["a", "b", "c"]
    assert _parse_list("") == []
    assert _parse_list(None) == []


def test_already_done_requires_both_artifacts(tmp_path):
    """둘 중 하나만 있으면 완료가 아니다 — 게이트가 반쪽 데이터로 판정한다."""
    runs = tmp_path / "runs"
    run = runs / "phase0_hydra_qs_gpt2_20260911_000000_s0"
    run.mkdir(parents=True)
    assert _already_done(runs, "phase0", "hydra_qs", 0) is None

    (run / "metrics.json").write_text("{}")
    assert _already_done(runs, "phase0", "hydra_qs", 0) is None

    (run / "gate_inputs.json").write_text("{}")
    assert _already_done(runs, "phase0", "hydra_qs", 0) == run


def test_already_done_matches_engine_and_seed(tmp_path):
    runs = tmp_path / "runs"
    for name in ("phase0_hydra_qs_gpt2_t_s0", "phase0_mlp_onepass_gpt2_t_s1"):
        d = runs / name
        d.mkdir(parents=True)
        (d / "metrics.json").write_text("{}")
        (d / "gate_inputs.json").write_text("{}")
    assert _already_done(runs, "phase0", "hydra_qs", 0) is not None
    assert _already_done(runs, "phase0", "hydra_qs", 1) is None
    assert _already_done(runs, "phase0", "mlp_onepass", 1) is not None


def _dry_run(args: list[str]) -> str:
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "sweep.py"), *args, "--dry-run"],
        capture_output=True, text=True, cwd=str(REPO_ROOT),
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return proc.stdout


def test_dry_run_enumerates_every_combination():
    out = _dry_run(["exp=phase0_mult", "--engines", "hydra_qs,mlp_onepass",
                    "--seeds", "0,1,2"])
    assert "= 6런" in out
    for engine in ("hydra_qs", "mlp_onepass"):
        for seed in (0, 1, 2):
            assert f"engine.type={engine} seed={seed}" in out


def test_dry_run_does_not_judge_gates():
    """판정은 check_gates.py 의 일이다 — 돌리면서 기준을 조정할 수 없게 한다."""
    out = _dry_run(["exp=phase0_mult", "--engines", "hydra_qs,mlp_onepass",
                    "--seeds", "0,1,2"])
    assert "PASS" not in out and "킬 스위치" not in out


def test_sweep_defaults_to_three_seeds():
    """게이트 ③ 이 조건당 시드 3개를 요구한다 (F-016)."""
    out = _dry_run(["exp=phase0_mult", "--engines", "hydra_qs"])
    assert "시드 [0, 1, 2]" in out


def test_runs_dir_is_a_flag_not_a_config_key():
    """설정 키로 두면 스냅샷에 들어가 같은 실험이 저장 위치로 신원이 갈린다."""
    from lsrr.config.schema import TOP_LEVEL_KEYS

    assert "runs_dir" not in TOP_LEVEL_KEYS

    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "train.py"), "--help"],
        capture_output=True, text=True, cwd=str(REPO_ROOT),
    )
    assert "--runs-dir" in proc.stdout + proc.stderr


def test_trainer_writes_a_stable_checkpoint_name():
    """평가·스윕이 에폭 번호를 몰라도 되게 경로가 예측 가능해야 한다."""
    src = (REPO_ROOT / "lsrr" / "runtime" / "trainer.py").read_text(encoding="utf-8")
    assert '"last.pt"' in src
