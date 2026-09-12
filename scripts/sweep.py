#!/usr/bin/env python
"""스윕 — 설정의 `sweep:` 절을 전개해 조건 × 시드를 돌린다.

**조합은 YAML이 정의한다.** 조건을 CLI 플래그로 받으면 실제로 돌린 조합이
설정 파일에 남지 않아, 나중에 같은 실험을 재현하려면 셸 히스토리를 뒤져야 한다.
그래서 이 스크립트에는 `--engines`/`--seeds` 가 없다. 시드조차 `sweep:` 축이다.

    sweep:
      readout.path.emission: [trajectory, single]
      seed: [0, 1, 2, 3, 4]

전개는 **인덱스로 주소지정 가능**하다. `--index K` 는 K번째 자식 하나만 돌리므로
slurm 배열 작업이 그대로 붙는다.

    sbatch --array=0-$(( $(python scripts/sweep.py exp=... --count) - 1 )) job.sh
    # job.sh 안: python scripts/sweep.py exp=... --index $SLURM_ARRAY_TASK_ID

각 자식 런은 `train → eval` 을 이어서 돌고 다음을 남긴다.

    runs/<run_id>/checkpoints/last.pt
    runs/<run_id>/metrics.json       accuracy 등          ← 게이트 ③
    runs/<run_id>/gate_inputs.json   collapse/anytime/Δ   ← 게이트 ①②④

**스윕은 판정하지 않는다.** 판정은 `scripts/check_gates.py`의 일이며, 이 경계를
지켜야 "돌리면서 기준을 조정하는" 일이 구조적으로 불가능해진다 (ADR-008).

사용:
    python scripts/sweep.py exp=ablation/A_emission --list    # 조합 확인
    python scripts/sweep.py exp=ablation/A_emission --count   # 조합 수만
    python scripts/sweep.py exp=ablation/A_emission           # 전부 순차 실행
    python scripts/sweep.py exp=ablation/A_emission --index 3 # 3번 자식만
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lsrr.config import load_config
from lsrr.config.sweep import SweepChild, sweep_plan

REPO_ROOT = Path(__file__).resolve().parent.parent

#: YAML로 옮겨 간 옛 플래그. 조용히 무시하면 "돌렸다고 생각한 조합"과 실제가
#: 갈리므로, 만나면 멈추고 옮길 자리를 알려 준다.
_RETIRED = {
    "--engines": "engine.type",
    "--seeds": "seed",
    "--engine": "engine.type",
    "--seed": "seed",
}


def _reject_retired(argv: list[str]) -> None:
    for a in argv:
        key = a.split("=", 1)[0]
        if key in _RETIRED:
            raise SystemExit(
                f"{key} 는 없앴다. 조합은 설정의 sweep: 절에 적는다 —\n"
                f"    sweep:\n      {_RETIRED[key]}: [...]\n"
                f"실제로 돌린 조합이 설정 파일에 남아야 재현할 수 있다."
            )


def _run(cmd: list[str], label: str) -> tuple[int, str]:
    """자식 프로세스를 돌리고 stdout 을 그대로 흘린다."""
    print(f"    $ {' '.join(cmd[1:])}", flush=True)
    proc = subprocess.Popen(
        cmd, cwd=str(REPO_ROOT), stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, bufsize=1,
    )
    lines: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        lines.append(line)
        print(f"      {line.rstrip()}", flush=True)
    proc.wait()
    if proc.returncode != 0:
        print(f"    [{label}] 실패 (코드 {proc.returncode})", flush=True)
    return proc.returncode, "".join(lines)


def _run_dir_from(output: str) -> Optional[Path]:
    """`RUN_DIR=` 한 줄에서 런 디렉터리를 집는다."""
    for line in reversed(output.splitlines()):
        if line.startswith("RUN_DIR="):
            return Path(line.split("=", 1)[1].strip())
    return None


def _already_done(runs_dir: Path, child: SweepChild) -> Optional[Path]:
    """같은 자식 이름으로 끝난 런이 있으면 그 디렉터리.

    런 이름이 `<자식 이름>_<엔진>_<백본>_<시각>_s<시드>` 이므로 자식 이름을
    접두사로 찾으면 된다. 두 산출물이 모두 있어야 "끝났다"로 본다 — 학습만
    되고 평가가 죽은 런을 완료로 세면 게이트가 빈 표로 판정한다.
    """
    for run in sorted(runs_dir.glob(f"{child.name}_*"), reverse=True):
        if _eval_targets(run) and all(
            (run / f"metrics{t}.json").exists() and (run / f"gate_inputs{t}.json").exists()
            for _, t in _eval_targets(run)
        ):
            return run
    return None


def _eval_targets(run_dir: Path) -> list[tuple[Path, str]]:
    """평가할 (체크포인트, 태그) 목록.

    2원화 학습은 한 런이 페이즈마다 체크포인트를 남긴다 (§5.0). **둘 다 평가해야
    한다** — Phase A 체크포인트가 완전 동결 조건이고 Phase B 증분이 순수 정렬
    이득이므로, 하나만 재면 Ablation B 의 절반이 사라진다.

    페이즈 체크포인트가 없으면 단일 페이즈 런이므로 `last.pt` 하나다.
    """
    ckpt_dir = run_dir / "checkpoints"
    phases = sorted(ckpt_dir.glob("phase_*.pt"))
    if phases:
        return [(p, f"_{p.stem}") for p in phases]
    last = ckpt_dir / "last.pt"
    return [(last, "")] if last.exists() else []


def _init_checkpoint(runs_dir: Path, source_exp: str, child: SweepChild) -> Path:
    """`source_exp` 스윕에서 같은 조건·시드의 런을 찾아 시작점 체크포인트를 돌려준다.

    자식 이름은 `<base>_condition_<c>_seed_<s>` 꼴이므로 `_condition_` 이후를
    맞춘다. 후보가 없거나 둘 이상이면 멈춘다 — 조용히 하나를 고르면 어느
    Phase A 위에 얹었는지 설정만 보고는 알 수 없게 된다.
    """
    suffix = child.name[child.name.index("_condition_"):]
    stem = source_exp.replace("/", "_")
    candidates = sorted(runs_dir.glob(f"{stem}{suffix}_*"))
    candidates = [c for c in candidates if (c / "checkpoints").is_dir()]
    if len(candidates) != 1:
        raise SystemExit(
            f"{child.name} 의 시작점을 정할 수 없다: {stem}{suffix}_* 에 해당하는 "
            f"런이 {len(candidates)}개다 (정확히 1개여야 한다). "
            + (f"{[c.name for c in candidates]}" if candidates else "")
        )
    ckpt_dir = candidates[0] / "checkpoints"
    # 1단은 단일 페이즈라 last.pt 가 Phase A 의 끝이다. 2페이즈 런이면 phase_A.pt.
    for name in ("phase_A.pt", "last.pt"):
        if (ckpt_dir / name).exists():
            return ckpt_dir / name
    raise SystemExit(f"{ckpt_dir} 에 체크포인트가 없다.")


def _child_overrides(child: SweepChild, rest: list[str]) -> list[str]:
    """설정 인자 + 이 자식의 sweep 덮어쓰기. 덮어쓰기가 뒤에 와야 이긴다."""
    return [*rest, *child.dotlist]


def _execute(
    child: SweepChild, args: argparse.Namespace, rest: list[str], tag: str
) -> Optional[dict[str, Any]]:
    """한 자식의 train → eval. 성공하면 대장에 남길 기록을 돌려준다."""
    print(f"\n[{tag}] {child.name}", flush=True)
    print(f"    {child.overrides}", flush=True)

    if args.skip_existing:
        existing = _already_done(args.runs_dir, child)
        if existing is not None:
            print(f"    건너뜀 — 이미 완료: {existing}")
            return {"index": child.index, "name": child.name,
                    "overrides": child.overrides, "run_dir": str(existing),
                    "skipped": True}

    overrides = _child_overrides(child, rest)
    train_cmd = [sys.executable, str(REPO_ROOT / "scripts" / "train.py"),
                 "--runs-dir", str(args.runs_dir), "--run-name", child.name]
    if args.init_from_sweep:
        init = _init_checkpoint(args.runs_dir, args.init_from_sweep, child)
        print(f"    시작점: {init}", flush=True)
        train_cmd += ["--init-from", str(init)]
    code, out = _run([*train_cmd, *overrides], f"{child.name} train")
    run_dir = _run_dir_from(out)
    if code != 0 or run_dir is None:
        return None

    targets = _eval_targets(run_dir)
    if not targets:
        print(f"    평가할 체크포인트가 없다: {run_dir}/checkpoints")
        return None

    for ckpt, tag in targets:
        eval_cmd = [
            sys.executable, str(REPO_ROOT / "scripts" / "eval.py"),
            "--run", str(run_dir),
            "--checkpoint", str(ckpt),
            "--split", args.split,
            "--anytime-batches", str(args.anytime_batches),
        ]
        if tag:
            eval_cmd += ["--tag", tag.lstrip("_")]
        if args.limit:
            eval_cmd += ["--limit", str(args.limit)]
        eval_cmd += overrides

        code, _ = _run(eval_cmd, f"{child.name} eval{tag}")
        if code != 0:
            return None

    return {"index": child.index, "name": child.name,
            "overrides": child.overrides, "run_dir": str(run_dir),
            "evaluated": [t for _, t in targets], "skipped": False}


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    _reject_retired(argv)

    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--runs-dir", type=Path, default=Path("runs"))
    ap.add_argument("--split", default="val")
    ap.add_argument("--limit", type=int, default=None, help="평가 샘플 수 상한")
    ap.add_argument("--anytime-batches", type=int, default=1)
    ap.add_argument("--index", type=int, default=None,
                    help="이 인덱스의 자식 하나만 돌린다 (slurm 배열용)")
    ap.add_argument("--init-from-sweep", type=str, default=None,
                    help="각 자식을 이 스윕(exp 이름)의 같은 조건·시드 런 체크포인트에서 "
                         "시작한다. 2단이 1단의 Phase A 를 이어 받을 때 (ADR-016)")
    ap.add_argument("--list", action="store_true", help="조합을 나열만 한다")
    ap.add_argument("--count", action="store_true", help="조합 수만 출력한다")
    ap.add_argument("--json", action="store_true", help="--list 를 JSON 으로")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--continue-on-error", action="store_true")
    ap.add_argument("-h", "--help", action="store_true")
    args, rest = ap.parse_known_args(argv)
    if args.help:
        print(__doc__)
        return 0

    exp = next(
        (a.split("=", 1)[1] for a in rest if a.startswith(("exp=", "config="))), "exp"
    )
    cfg = load_config(rest)
    # 런 이름에 경로 구분자가 들어가면 런 디렉터리가 중첩된다.
    children = sweep_plan(cfg, base_name=exp.replace("/", "_"))

    if args.count:
        print(len(children))
        return 0

    if args.list:
        if args.json:
            print(json.dumps(
                [{"index": c.index, "name": c.name, "overrides": c.overrides}
                 for c in children], indent=2, ensure_ascii=False))
        else:
            print(f"═══ {exp}: {len(children)}조합 ═══")
            for c in children:
                print(f"  [{c.index:3d}] {c.name}")
                print(f"        {' '.join(c.dotlist) or '(sweep 절 없음)'}")
        return 0

    if args.index is not None:
        if not 0 <= args.index < len(children):
            print(f"인덱스 {args.index}가 범위를 벗어났다 (0..{len(children) - 1}).")
            return 2
        selected = [children[args.index]]
    else:
        selected = children

    print(f"═══ 스윕: {exp} ═══")
    print(f"  {len(selected)}/{len(children)}런"
          + (f"  (인덱스 {args.index})" if args.index is not None else ""))
    print(f"  각 런: train → eval (split={args.split}"
          + (f", limit={args.limit}" if args.limit else "") + ")")
    if args.dry_run:
        for c in selected:
            print(f"    [{c.index:3d}] {c.name}: {' '.join(c.dotlist)}")
        return 0

    started = time.time()
    done: list[dict[str, Any]] = []
    failures: list[str] = []

    for i, child in enumerate(selected, 1):
        record = _execute(child, args, rest, tag=f"{i}/{len(selected)}")
        if record is None:
            failures.append(child.name)
            if not args.continue_on_error:
                print("\n중단한다. 계속하려면 --continue-on-error.")
                return 1
            continue
        done.append(record)

    elapsed = time.time() - started
    print(f"\n═══ 스윕 완료: {len(done)}/{len(selected)}런, {elapsed / 60:.1f}분 ═══")
    for d in done:
        print(f"  [{d['index']:3d}] {d['name']} → {d['run_dir']}")
    if failures:
        print(f"  실패 {len(failures)}: {', '.join(failures)}")

    _write_index(args, exp, children, done, failures)

    # 판정은 스윕의 일이 아니다 (ADR-008). 명령만 알려 준다.
    print("\n판정하려면:")
    print(f"  uv run python scripts/check_gates.py --runs-dir {args.runs_dir} ...")
    return 1 if failures else 0


def _write_index(
    args: argparse.Namespace,
    exp: str,
    children: list[SweepChild],
    done: list[dict[str, Any]],
    failures: list[str],
) -> None:
    """대장을 쓴다. 배열 작업에서는 자식마다 조각으로 남기고 합친다.

    배열 작업은 자식들이 동시에 끝나므로 한 파일에 같이 쓰면 서로 덮어쓴다.
    `--index` 로 돈 런은 자기 조각만 쓰고, 합치는 일은 읽는 쪽이 한다.
    """
    stem = exp.replace("/", "_")
    if args.index is not None:
        path = args.runs_dir / "sweep_parts" / f"{stem}_{args.index:04d}.json"
    else:
        path = args.runs_dir / f"sweep_{stem}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "exp": exp,
                "total": len(children),
                "index": args.index,
                "axes": list(dict.fromkeys(k for c in children for k in c.overrides)),
                "slurm_job": os.environ.get("SLURM_ARRAY_JOB_ID"),
                "runs": done,
                "failures": failures,
            },
            indent=2, ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"  대장: {path}")


if __name__ == "__main__":
    sys.exit(main())
