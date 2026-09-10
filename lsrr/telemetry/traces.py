"""사이클 진단 트레이스 — 분석의 원재료 (telemetry/README).

무엇을 기록할지는 **분석 계획에서 역산**해 정한다:

| 분석 | 필요한 필드 |
|---|---|
| Δ 궤적·거동 분류 | `delta_state` 전 사이클 |
| 홉별 수렴 사이클 | `stopping_cycle` + `meta.hop_count` |
| α 분포 | `alpha` |
| anytime 곡선 | 사이클별 판독 정오 |
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import torch

from lsrr.core.types import ReasoningTrace


def summarize_trace(
    trace: ReasoningTrace,
    include_alpha: bool = True,
    include_state_stats: bool = True,
) -> dict[str, Any]:
    """`ReasoningTrace` → JSON 직렬화 가능한 요약.

    원본 `R⁽ᵐ⁾` 전체는 담지 않는다 — 용량이 크고, 필요한 분석(logit lens)은
    별도 플래그로 부분집합만 저장한다.
    """
    record: dict[str, Any] = {
        "num_cycles": trace.num_cycles,
        "deltas": [d.delta_state for d in trace.per_cycle],
        "tbptt_window": list(trace.tbptt_window),
        "supervised_cycles": list(trace.supervised_cycles),
    }

    kls = [d.kl_div for d in trace.per_cycle if d.kl_div is not None]
    if kls:
        record["kl"] = kls
    ents = [d.entropy for d in trace.per_cycle if d.entropy is not None]
    if ents:
        record["entropy"] = ents

    if trace.stopping_cycles is not None:
        record["stopping_cycles"] = trace.stopping_cycles.tolist()
        record["avg_cycles"] = float(trace.stopping_cycles.float().mean())

    if include_alpha and trace.alpha is not None:
        record["alpha_mean"] = trace.alpha.detach().float().mean(dim=0).tolist()

    if include_state_stats and trace.R_star is not None:
        record.update(state_statistics(trace.R_star))

    record.update({k: v for k, v in trace.meta.items() if _jsonable(v)})
    return record


def state_statistics(R: torch.Tensor) -> dict[str, float]:
    """자명해 붕괴 진단용 요약 (게이트 ①의 예비 지표).

    본격적인 붕괴 진단은 `analysis/collapse.py`(M5)가 하고, 여기서는 매 스텝
    싸게 잴 수 있는 것만 남긴다.
    """
    with torch.no_grad():
        x = R.detach().float()
        return {
            "state_norm": float(x.norm(dim=-1).mean()),
            "state_layer_std": float(x.std(dim=1).mean()),
            "state_feat_std": float(x.std(dim=-1).mean()),
        }


def _jsonable(value: Any) -> bool:
    return isinstance(value, (int, float, str, bool, list, type(None)))


__all__ = ("summarize_trace", "state_statistics")
