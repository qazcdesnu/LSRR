"""스윕 배선 — 조합은 YAML 이 정하고, 판정은 하지 않는다 (ADR-008).

이 파일이 고정하는 것:

- **조합은 설정의 `sweep:` 절에서만 온다.** CLI 플래그로 조건을 받으면 실제로
  돌린 조합이 설정 파일에 남지 않아 재현이 셸 히스토리에 의존한다
- 전개는 **인덱스로 주소지정 가능**해야 한다 (slurm 배열 ↔ 자식 1:1)
- 런 디렉터리를 **기계 판독용 한 줄**로 주고받는다 (사람이 읽는 문구 파싱 금지)
- 스윕은 게이트를 판정하지 않는다
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from lsrr.config.sweep import sweep_plan, sweep_size  # noqa: E402
from lsrr.core.errors import ConfigError  # noqa: E402
from scripts.sweep import _already_done, _run_dir_from  # noqa: E402


# ------------------------------------------------------------------ 전개


def _cfg(sweep: dict) -> OmegaConf:
    return OmegaConf.create({"seed": 42, "engine": {"type": "hydra_qs"}, "sweep": sweep})


def test_plan_is_index_addressable():
    """slurm 배열 인덱스가 자식을 가리키려면 순서가 결정적이어야 한다."""
    cfg = _cfg({"engine.type": ["a", "b"], "seed": [0, 1, 2]})
    first = sweep_plan(cfg, "x")
    second = sweep_plan(cfg, "x")
    assert [c.index for c in first] == list(range(6))
    assert [c.name for c in first] == [c.name for c in second]
    assert [c.overrides for c in first] == [c.overrides for c in second]


def test_plan_covers_the_full_cartesian_product():
    cfg = _cfg({"engine.type": ["a", "b"], "seed": [0, 1, 2]})
    plan = sweep_plan(cfg, "x")
    assert len(plan) == sweep_size(cfg) == 6
    assert {(c.cfg.engine.type, c.cfg.seed) for c in plan} == {
        (e, s) for e in ("a", "b") for s in (0, 1, 2)
    }


def test_grouped_axis_does_not_explode_into_impossible_conditions():
    """조건이 여러 키의 특정 조합으로만 성립할 때 데카르트 곱은 틀린 답이다.

    Ablation A 의 세 조건은 emission × termination × anchor 8가지 중 3가지다.
    축을 따로 두면 존재하지 않는 조건 5개가 생긴다.
    """
    cfg = OmegaConf.create({
        "sweep": {
            "condition": [
                {"name": "single_v1", "readout.path.emission": "single",
                 "readout.fusion.anchor": "ctx"},
                {"name": "dynamic_m", "readout.path.emission": "trajectory",
                 "readout.fusion.anchor": "first"},
            ],
            "seed": [0, 1],
        },
        "readout": {"path": {"emission": "trajectory"}, "fusion": {"anchor": "first"}},
        "seed": 42,
    })
    plan = sweep_plan(cfg, "A")
    assert len(plan) == 4
    pairs = {(c.cfg.readout.path.emission, c.cfg.readout.fusion.anchor) for c in plan}
    assert pairs == {("single", "ctx"), ("trajectory", "first")}


def test_grouped_axis_name_reaches_the_run_name():
    """런 디렉터리 이름만 보고 어느 조건인지 알 수 있어야 한다."""
    cfg = OmegaConf.create({
        "sweep": {"condition": [{"name": "single_v1", "a.b": 1}], "seed": [3]},
        "a": {"b": 0}, "seed": 42,
    })
    (child,) = sweep_plan(cfg, "A")
    assert child.name == "A_condition_single_v1_seed_3"


def test_grouped_axis_without_overrides_is_rejected():
    cfg = OmegaConf.create({"sweep": {"condition": [{"name": "empty"}]}})
    with pytest.raises(ConfigError, match="덮어쓸 키가 없다"):
        sweep_plan(cfg, "A")


def test_missing_sweep_yields_one_child():
    """호출부가 스윕 여부로 분기하지 않아도 되게 한다."""
    cfg = OmegaConf.create({"seed": 42})
    (child,) = sweep_plan(cfg, "solo")
    assert child.index == 0 and child.name == "solo" and child.overrides == {}


def test_dotlist_round_trips_to_child_process_args():
    cfg = _cfg({"engine.type": ["a"], "seed": [7]})
    (child,) = sweep_plan(cfg, "x")
    assert child.dotlist == ["engine.type=a", "seed=7"]


# ------------------------------------------------------------ 런 디렉터리


def test_run_dir_is_parsed_from_a_machine_readable_line():
    out = "학습 파라미터 5,387,777 / 백본 …\n완료: 16 스텝 → runs/x\nRUN_DIR=runs/abc_s0\n"
    assert _run_dir_from(out) == Path("runs/abc_s0")


def test_run_dir_takes_the_last_marker():
    out = "RUN_DIR=runs/a\nRUN_DIR=runs/b\n"
    assert _run_dir_from(out) == Path("runs/b")


def test_missing_marker_is_not_guessed():
    """문구를 추측해 파싱하면 출력을 고칠 때 스윕이 조용히 깨진다."""
    assert _run_dir_from("완료: 16 스텝 → runs/abc_s0\n") is None


def _child(name: str):
    from lsrr.config.sweep import SweepChild

    return SweepChild(index=0, name=name, overrides={}, cfg=OmegaConf.create({}))


def test_already_done_requires_both_artifacts(tmp_path):
    """둘 중 하나만 있으면 완료가 아니다 — 게이트가 반쪽 데이터로 판정한다."""
    runs = tmp_path / "runs"
    child = _child("A_condition_dynamic_m_seed_0")
    run = runs / f"{child.name}_hydra_qs_gpt2_20260911_000000_s0"
    run.mkdir(parents=True)
    assert _already_done(runs, child) is None

    (run / "metrics.json").write_text("{}")
    assert _already_done(runs, child) is None

    (run / "gate_inputs.json").write_text("{}")
    assert _already_done(runs, child) == run


def test_already_done_does_not_confuse_sibling_conditions(tmp_path):
    runs = tmp_path / "runs"
    for name in ("A_condition_dynamic_m_seed_0", "A_condition_single_v1_seed_0"):
        d = runs / f"{name}_hydra_qs_gpt2_t_s0"
        d.mkdir(parents=True)
        (d / "metrics.json").write_text("{}")
        (d / "gate_inputs.json").write_text("{}")
    assert _already_done(runs, _child("A_condition_dynamic_m_seed_0")) is not None
    assert _already_done(runs, _child("A_condition_fixed_k_seed_0")) is None


# ---------------------------------------------------------------- 명령줄


def _sweep(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "sweep.py"), *args],
        capture_output=True, text=True, cwd=str(REPO_ROOT),
    )


def test_count_matches_the_config():
    proc = _sweep(["exp=ablation/A_emission_prosqa", "--count"])
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert int(proc.stdout.strip()) == 15  # 3조건 × 5시드


def test_index_selects_exactly_one_child():
    proc = _sweep(["exp=ablation/A_emission_prosqa", "--index", "7", "--dry-run"])
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "1/15런" in proc.stdout
    assert "condition_fixed_k_seed_2" in proc.stdout


def test_out_of_range_index_fails_loudly():
    """배열 크기를 손으로 적어 어긋났을 때 조용히 아무것도 안 돌면 안 된다."""
    proc = _sweep(["exp=ablation/A_emission_prosqa", "--index", "99", "--dry-run"])
    assert proc.returncode == 2
    assert "범위를 벗어났다" in proc.stdout


def test_retired_cli_flags_are_rejected_with_a_pointer_to_yaml():
    """조용히 무시하면 '돌렸다고 생각한 조합'과 실제가 갈린다."""
    proc = _sweep(["exp=ablation/C_engine", "--engines", "hydra_qs", "--dry-run"])
    assert proc.returncode != 0
    assert "sweep:" in proc.stdout + proc.stderr


def test_dry_run_does_not_judge_gates():
    """판정은 check_gates.py 의 일이다 — 돌리면서 기준을 조정할 수 없게 한다."""
    proc = _sweep(["exp=ablation/A_emission_prosqa", "--dry-run"])
    assert "PASS" not in proc.stdout and "킬 스위치" not in proc.stdout


def test_runs_dir_is_a_flag_not_a_config_key():
    """설정 키로 두면 스냅샷에 들어가 같은 실험이 저장 위치로 신원이 갈린다."""
    from lsrr.config.schema import TOP_LEVEL_KEYS

    assert "runs_dir" not in TOP_LEVEL_KEYS

    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "train.py"), "--help"],
        capture_output=True, text=True, cwd=str(REPO_ROOT),
    )
    assert "--runs-dir" in proc.stdout + proc.stderr


def test_slurm_array_size_comes_from_the_config():
    """배열 크기를 손으로 적으면 조합을 늘렸을 때 뒤쪽 자식이 조용히 안 돈다."""
    submit = (REPO_ROOT / "scripts" / "slurm" / "submit.sh").read_text(encoding="utf-8")
    assert "--count" in submit
    array = (REPO_ROOT / "scripts" / "slurm" / "sweep_array.sh").read_text(encoding="utf-8")
    assert "SLURM_ARRAY_TASK_ID" in array and "--index" in array


def test_trainer_writes_a_stable_checkpoint_name():
    """평가·스윕이 에폭 번호를 몰라도 되게 경로가 예측 가능해야 한다."""
    src = (REPO_ROOT / "lsrr" / "runtime" / "trainer.py").read_text(encoding="utf-8")
    assert '"last.pt"' in src
