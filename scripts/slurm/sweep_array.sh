#!/bin/bash
# slurm 배열 작업 — 배열 인덱스 하나 ↔ 스윕 자식 하나.
#
# 스윕 조합은 설정의 `sweep:` 절이 정한다. 이 스크립트는 조합을 만들지 않고,
# `--index $SLURM_ARRAY_TASK_ID` 로 몇 번째를 돌릴지만 고른다. 배열 크기도
# 손으로 적지 않는다 — submit.sh 가 `--count` 로 물어본다.
#
#   sbatch --array=0-14 scripts/slurm/sweep_array.sh exp=ablation/A_emission_prosqa
#   scripts/slurm/submit.sh exp=ablation/A_emission_prosqa   # 배열 크기 자동
#
#SBATCH --job-name=lsrr-sweep
#SBATCH --output=logs/slurm/%x_%A_%a.out
#SBATCH --error=logs/slurm/%x_%A_%a.out
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$REPO_ROOT"
mkdir -p logs/slurm

INDEX="${SLURM_ARRAY_TASK_ID:-0}"
RUNS_DIR="${LSRR_RUNS_DIR:-runs}"

echo "═══ lsrr sweep: index=$INDEX host=$(hostname) ═══"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# uv 가 있으면 uv 로, 없으면 저장소 가상환경으로. 어느 쪽이든 같은 잠금 파일이다.
if command -v uv >/dev/null 2>&1; then
    RUNNER=(uv run python)
else
    RUNNER=("$REPO_ROOT/.venv/bin/python")
fi

exec "${RUNNER[@]}" scripts/sweep.py \
    --index "$INDEX" \
    --runs-dir "$RUNS_DIR" \
    --skip-existing \
    "$@"
