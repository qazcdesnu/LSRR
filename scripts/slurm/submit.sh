#!/bin/bash
# 스윕 전체를 배열 작업으로 제출한다. 배열 크기는 설정에서 읽는다.
#
#   scripts/slurm/submit.sh exp=ablation/A_emission_prosqa
#   scripts/slurm/submit.sh exp=ablation/A_emission_prosqa --dry   # 명령만 출력
#
# 배열 크기를 손으로 적으면 조합을 늘렸을 때 뒤쪽 자식이 조용히 안 돈다.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

DRY=0
ARGS=()
for a in "$@"; do
    if [[ "$a" == "--dry" ]]; then DRY=1; else ARGS+=("$a"); fi
done

if command -v uv >/dev/null 2>&1; then
    RUNNER=(uv run python)
else
    RUNNER=("$REPO_ROOT/.venv/bin/python")
fi

N=$("${RUNNER[@]}" scripts/sweep.py --count "${ARGS[@]}")
if ! [[ "$N" =~ ^[0-9]+$ ]] || [[ "$N" -lt 1 ]]; then
    echo "조합 수를 읽지 못했다: $N" >&2
    exit 1
fi
LAST=$((N - 1))

# 동시에 돌릴 작업 수 상한. GPU 를 나눠 쓰는 큐에서 배열 전체가 한꺼번에
# 뜨는 것을 막는다. LSRR_MAX_CONCURRENT 로 조절한다.
LIMIT="${LSRR_MAX_CONCURRENT:-4}"

CMD=(sbatch "--array=0-${LAST}%${LIMIT}" scripts/slurm/sweep_array.sh "${ARGS[@]}")
echo "조합 ${N}개 → ${CMD[*]}"
if [[ "$DRY" -eq 1 ]]; then exit 0; fi
"${CMD[@]}"
