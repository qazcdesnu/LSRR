"""체크포인트 — 학습 파라미터만 저장한다 (I1).

백본 가중치를 저장하지 않는다: 용량 낭비이자 "무엇이 학습되었는가"의 혼동
원인이다. 대신 백본 **식별자와 가중치 해시**를 저장해 로드 시 대조한다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn

from lsrr.core.errors import FrozenBackboneViolation


def save_checkpoint(
    model: nn.Module,
    path: str | Path,
    optimizer: Optional[torch.optim.Optimizer] = None,
    step: int = 0,
    meta: Optional[dict[str, Any]] = None,
) -> Path:
    """학습 파라미터 + 옵티마이저 상태 + 백본 지문을 저장한다."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    encoder = getattr(model, "encoder", None)
    payload: dict[str, Any] = {
        "state_dict": model.state_dict(),  # 백본은 트리 밖이라 포함되지 않는다
        "step": step,
        "meta": meta or {},
        "backbone": {
            "id": getattr(encoder, "model_name", None),
            "weight_hash": encoder.weight_hash() if encoder is not None else None,
            "num_layers": getattr(encoder, "num_layers", None),
            "hidden_dim": getattr(encoder, "hidden_dim", None),
        },
        "widths": getattr(model, "widths", {}),
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()

    torch.save(payload, path)
    return path


def load_checkpoint(
    model: nn.Module,
    path: str | Path,
    optimizer: Optional[torch.optim.Optimizer] = None,
    strict: bool = True,
    verify_backbone: bool = True,
) -> dict[str, Any]:
    """체크포인트를 적재하고 백본 지문을 대조한다.

    다른 백본에서 학습된 헤드를 조용히 얹으면 결과가 무의미해진다.
    """
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)

    if verify_backbone:
        encoder = getattr(model, "encoder", None)
        saved = payload.get("backbone", {})
        if encoder is not None and saved.get("weight_hash") is not None:
            if encoder.weight_hash() != saved["weight_hash"]:
                raise FrozenBackboneViolation(
                    f"체크포인트의 백본 지문({saved['weight_hash']})과 현재 "
                    f"백본({encoder.weight_hash()})이 다르다. 저장 시 백본은 "
                    f"'{saved.get('id')}'였다."
                )

    model.load_state_dict(payload["state_dict"], strict=strict)
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    return payload


def parameter_summary(model: nn.Module) -> dict[str, Any]:
    """모듈별 학습 파라미터 수. §5의 예산 주장을 매 런에서 확인한다."""
    by_module: dict[str, int] = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        top = name.split(".")[0]
        by_module[top] = by_module.get(top, 0) + param.numel()
    return {"total": sum(by_module.values()), "by_module": by_module}


__all__ = ("save_checkpoint", "load_checkpoint", "parameter_summary")
