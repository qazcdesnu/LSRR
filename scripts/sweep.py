#!/usr/bin/env python
"""스윕 — 조건 × 시드를 돌려 Phase 0 게이트 판정 데이터를 만든다.

게이트 ③은 킬 스위치이고 **조건당 시드 3개 이상**을 요구한다 (F-016). 손으로
6런을 돌리면 조건 하나를 빠뜨리거나 시드를 섞기 쉬우므로, 조합을 코드가 만든다.

각 자식 런은 `train → eval` 을 이어서 돌고, 다음을 남긴다.

    runs/<run_id>/checkpoints/last.pt
    runs/<run_id>/metrics.json       accuracy 등          ← 게이트 ③
    runs/<run_id>/gate_inputs.json   collapse/anytime/Δ   ← 게이트 ①②④

**스윕은 판정하지 않는다.** 판정은 `scripts/check_gates.py`의 일이며, 이 경계를
지켜야 "돌리면서 기준을 조정하는" 일이 구조적으로 불가능해진다 (ADR-008).
스윕이 끝나면 판정 명령을 출력한다.

사용:
    # 설정의 sweep: 절을 전개 (예: configs/ablation/C_engine.yaml)
    python scripts/sweep.py exp=ablation/C_engine --seeds 0,1,2

    # Phase 0 킬 스위치용 최소 조합 — 명시 지정
    python scripts/sweep.py exp=phase0_mult \\
        --engines hydra_qs,mlp_onepass --seeds 0,1,2

    python scripts/sweep.py ... --dry-run       # 조합만 출력
    python scripts/sweep.py ... --skip-existing # 이미 끝난 조합 건너뛰기
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lsrr.config import load_config
from lsrr.config.schema import get_path
from lsrr.config.sweep import expand_sweep

REPO_ROOT = Path(__file__).resolve().parent.parent


def _parse_list(text: Optional[str]) -> list[str]:
    return [p.strip() for p in text.split(",") if p.strip()] if text else []


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


def _already_done(runs_dir: Path, exp: str, engine: str, seed: int) -> Optional[Path]:
    """같은 조합의 완료된 런이 있으면 그 디렉터리."""
    for run in sorted(runs_dir.glob(f"*_{engine}_*_s{seed}")):
        if (run / "metrics.json").exists() and (run / "gate_inputs.json").exists():
            return run
    return None


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--engines", type=str, help="쉼표 구분. 없으면 설정의 sweep: 절을 쓴다")
    ap.add_argument("--seeds", type=str, default="0,1,2")
    ap.add_argument("--runs-dir", type=Path, default=Path("runs"))
    ap.add_argument("--split", default="val")
    ap.add_argument("--limit", type=int, default=None, help="평가 샘플 수 상한")
    ap.add_argument("--anytime-batches", type=int, default=1)
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

    engines = _parse_list(args.engines)
    if not engines:
        children = expand_sweep(cfg, base_name=exp)
        engines = sorted({str(get_path(c, "engine.type")) for _, c in children})
        if not engines:
            print("스윕할 엔진이 없다. --engines 로 지정하거나 설정에 sweep: 절을 두라.")
            return 2
    seeds = [int(s) for s in _parse_list(args.seeds)]

    combos = [(e, s) for e in engines for s in seeds]
    print(f"═══ 스윕: {exp} ═══")
    print(f"  엔진 {engines}  × 시드 {seeds}  = {len(combos)}런")
    print(f"  각 런: train → eval (split={args.split}"
          + (f", limit={args.limit}" if args.limit else "") + ")")
    if args.dry_run:
        for e, s in combos:
            print(f"    - engine.type={e} seed={s}")
        return 0

    started = time.time()
    done: list[dict[str, object]] = []
    failures: list[str] = []

    for i, (engine, seed) in enumerate(combos, 1):
        label = f"{engine}/s{seed}"
        print(f"\n[{i}/{len(combos)}] {label}", flush=True)

        if args.skip_existing:
            existing = _already_done(args.runs_dir, exp, engine, seed)
            if existing is not None:
                print(f"    건너뜀 — 이미 완료: {existing}")
                done.append({"engine": engine, "seed": seed, "run_dir": str(existing)})
                continue

        overrides = [a for a in rest] + [
            f"engine.type={engine}",
            f"seed={seed}",
        ]
        code, out = _run(
            [
                sys.executable, str(REPO_ROOT / "scripts" / "train.py"),
                "--runs-dir", str(args.runs_dir), *overrides,
            ],
            f"{label} train",
        )
        run_dir = _run_dir_from(out)
        if code != 0 or run_dir is None:
            failures.append(f"{label} (train)")
            if not args.continue_on_error:
                print("\n중단한다. 계속하려면 --continue-on-error.")
                return 1
            continue

        eval_cmd = [
            sys.executable, str(REPO_ROOT / "scripts" / "eval.py"),
            "--run", str(run_dir),
            "--checkpoint", str(run_dir / "checkpoints" / "last.pt"),
            "--split", args.split,
            "--anytime-batches", str(args.anytime_batches),
        ]
        if args.limit:
            eval_cmd += ["--limit", str(args.limit)]
        eval_cmd += overrides

        code, _ = _run(eval_cmd, f"{label} eval")
        if code != 0:
            failures.append(f"{label} (eval)")
            if not args.continue_on_error:
                return 1
            continue

        done.append({"engine": engine, "seed": seed, "run_dir": str(run_dir)})

    elapsed = time.time() - started
    print(f"\n═══ 스윕 완료: {len(done)}/{len(combos)}런, {elapsed / 60:.1f}분 ═══")
    for d in done:
        print(f"  {d['engine']}/s{d['seed']} → {d['run_dir']}")
    if failures:
        print(f"  실패 {len(failures)}: {', '.join(failures)}")

    index = args.runs_dir / f"sweep_{exp.replace('/', '_')}.json"
    index.parent.mkdir(parents=True, exist_ok=True)
    index.write_text(
        json.dumps(
            {"exp": exp, "engines": engines, "seeds": seeds,
             "runs": done, "failures": failures},
            indent=2, ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"  대장: {index}")

    # 판정은 스윕의 일이 아니다 (ADR-008). 명령만 알려 준다.
    if len(engines) >= 2 and done:
        ref = next((d["run_dir"] for d in done if d["engine"] == engines[0]), None)
        print("\n판정하려면:")
        print(
            f"  uv run python scripts/check_gates.py --run {ref} "
            f"--runs-dir {args.runs_dir} "
            f"--hydra-glob '*_{engines[0]}_*' --mlp-glob '*_{engines[1]}_*'"
        )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
