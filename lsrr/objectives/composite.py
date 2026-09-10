"""손실 합성 (제안서 §5).

    L = L_NLL + λ_ds·L_DeepSup + λ_reg·L_VarReg (+ λ_KD·L_KD : ablation 전용)

항별 지표를 개별로도 보고한다 — 합계만 보면 어느 항이 움직였는지 알 수 없고,
Ablation D(감독 장치 기여 분해)가 성립하지 않는다.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional, Sequence

import torch
import torch.nn as nn

from lsrr.core.errors import ConfigError
from lsrr.core.registry import OBJECTIVE_REGISTRY
from lsrr.core.types import ReasoningTrace


class CompositeObjective(nn.Module):
    """가중 합성.

    Args:
        terms: `(이름, 모듈, 가중치)` 목록.
    """

    def __init__(self, terms: Sequence[tuple[str, nn.Module, float]]) -> None:
        super().__init__()
        if not terms:
            raise ConfigError("활성화된 손실 항이 하나도 없다.")
        self.names = [n for n, _, _ in terms]
        self.terms = nn.ModuleList([m for _, m, _ in terms])
        self.weights = [float(w) for _, _, w in terms]

    @classmethod
    def from_config(cls, cfg: Any, **kwargs: Any) -> "CompositeObjective":
        """설정의 `objective:` 절에서 활성 항만 골라 만든다.

        각 항은 `{enabled: bool, w: float, ...}` 형태이며, `enabled=false`인
        항은 아예 만들어지지 않는다 — 가중치 0으로 두면 계산만 낭비된다.
        """
        terms: list[tuple[str, nn.Module, float]] = []
        for name in ("answer_nll", "deep_supervision", "variance_reg", "distillation"):
            node = cfg.get(name) if hasattr(cfg, "get") else None
            if node is None or not node.get("enabled", False):
                continue
            params = {
                k: v for k, v in dict(node).items() if k not in ("enabled", "w")
            }
            module = OBJECTIVE_REGISTRY.build({"type": name, **params}, **kwargs)
            terms.append((name, module, float(node.get("w", 1.0))))
        return cls(terms)

    def forward(
        self, trace: ReasoningTrace, batch: dict[str, Any]
    ) -> dict[str, torch.Tensor]:
        total: Optional[torch.Tensor] = None
        metrics: dict[str, torch.Tensor] = {}

        for name, term, w in zip(self.names, self.terms, self.weights):
            result = term(trace, batch)
            sub = result["loss"]
            total = sub * w if total is None else total + sub * w
            for k, v in result.items():
                metrics[f"loss/{k}" if k != "loss" else f"loss/{name}"] = (
                    v.detach() if torch.is_tensor(v) else v
                )

        metrics["loss/total"] = total.detach()
        return {"loss": total, **metrics}

    def __repr__(self) -> str:
        parts = ", ".join(f"{n}×{w}" for n, w in zip(self.names, self.weights))
        return f"CompositeObjective({parts})"


__all__ = ("CompositeObjective",)
