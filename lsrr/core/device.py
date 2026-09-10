"""디바이스·dtype 해석 (L0 공용 원시).

디바이스는 조립 루트(builder)와 구동(runtime) 양쪽이 알아야 하는데, 레이어
규칙상 builder(L4)는 runtime(L5)을 import할 수 없다. 그래서 해석 자체는
`core`에 둔다 — 정책이 아니라 원시 연산이기 때문이다.
"""

from __future__ import annotations

from typing import Any

import torch

_DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def resolve_device(spec: str = "auto") -> torch.device:
    if spec in (None, "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


def resolve_dtype(spec: str = "float32") -> torch.dtype:
    if spec not in _DTYPES:
        raise ValueError(f"dtype '{spec}'를 모른다. 가능: {sorted(_DTYPES)}")
    return _DTYPES[spec]


def device_report(device: torch.device) -> dict[str, Any]:
    report: dict[str, Any] = {"device": str(device)}
    if device.type == "cuda":
        report["gpu_name"] = torch.cuda.get_device_name(device)
        report["gpu_memory_gb"] = round(
            torch.cuda.get_device_properties(device).total_memory / 1024**3, 1
        )
    return report


__all__ = ("resolve_device", "resolve_dtype", "device_report")
