"""FLOPs·지연 회계 — 백본 1회를 반드시 포함한다 (I7, 제안서 §6).

제안서의 효율 주장은 "Coconut이 트랜스포머를 k회 호출할 때 우리는 1회"다. 그
1회를 비용에서 빼면 주장 자체가 검증 불가능해진다. 따라서 백본 항이 측정되지
않았으면 **0으로 두지 않고 `None`으로 두고 플래그를 내린다** (규약 §3).

개작: Legacy_LSRR/lsrr/utils/flops.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from lsrr.core.errors import CostAccountingError


@dataclass(frozen=True)
class CostReport:
    """샘플 1개당 비용 분해.

    합계만 보고하면 어느 항이 지배적인지 알 수 없고, 백본을 빼먹은 보고와
    구분되지 않는다.
    """

    backbone_flops: Optional[float]
    memory_flops: float
    engine_flops_per_cycle: float
    cycles: float
    readout_flops: float
    backbone_latency_ms: Optional[float] = None
    refinement_latency_ms: Optional[float] = None

    @property
    def engine_flops(self) -> float:
        return self.engine_flops_per_cycle * self.cycles

    @property
    def includes_backbone(self) -> bool:
        return self.backbone_flops is not None

    @property
    def total_flops(self) -> float:
        """백본이 없으면 던진다 — 부분 합계를 총계로 보고하면 I7 위반이다."""
        if self.backbone_flops is None:
            raise CostAccountingError(
                "백본 순전파 비용이 측정되지 않아 총 FLOPs를 낼 수 없다. "
                "제안서 §6의 효율 주장은 이 항에 걸려 있으므로 0으로 두지 않는다."
            )
        return (
            self.backbone_flops
            + self.memory_flops
            + self.engine_flops
            + self.readout_flops
        )

    @property
    def total_latency_ms(self) -> Optional[float]:
        if self.backbone_latency_ms is None or self.refinement_latency_ms is None:
            return None
        return self.backbone_latency_ms + self.refinement_latency_ms

    def as_dict(self) -> dict[str, object]:
        """보고용. 백본 미측정을 **드러내는** 형태로 직렬화한다."""
        out: dict[str, object] = {
            "backbone_flops": self.backbone_flops,
            "memory_flops": self.memory_flops,
            "engine_flops": self.engine_flops,
            "engine_flops_per_cycle": self.engine_flops_per_cycle,
            "cycles": self.cycles,
            "readout_flops": self.readout_flops,
            "includes_backbone": self.includes_backbone,
            "backbone_latency_ms": self.backbone_latency_ms,
            "refinement_latency_ms": self.refinement_latency_ms,
            "total_latency_ms": self.total_latency_ms,
        }
        out["total_flops"] = self.total_flops if self.includes_backbone else None
        return out


def transformer_forward_flops(num_layers: int, d_model: int, seq_len: int) -> float:
    """트랜스포머 순전파 1회. 레이어·토큰당 `24d² + 4·T·d`.

    QKVO 사영 8d², 어텐션 점수+가중합 4Td, FFN(4d 확장) 16d². LM 헤드는 뺀다 —
    문맥 인코딩 패스는 은닉 상태만 읽는다.
    """
    return float(num_layers) * seq_len * (24.0 * d_model**2 + 4.0 * seq_len * d_model)


def ssm_engine_flops(d_model: int, num_layers_axis: int, n_blocks: int, expand: int = 2) -> float:
    """SSM 코어 1사이클. 레이어 축 길이에 **선형**이다."""
    d_inner = expand * d_model
    per_block = (
        4.0 * num_layers_axis * d_model * d_inner  # in/out 사영
        + 12.0 * num_layers_axis * d_inner  # 스캔
    )
    return float(n_blocks) * per_block


def attention_engine_flops(d_model: int, num_layers_axis: int, n_blocks: int, d_ffn: int) -> float:
    """어텐션 코어 1사이클. 레이어 축 길이에 **이차**다 — 이 차이가 §4.2의 논거다."""
    L = num_layers_axis
    per_block = (
        8.0 * L * d_model**2  # QKVO
        + 4.0 * (L**2) * d_model  # 점수 + 가중합
        + 2.0 * L * d_model * d_ffn * 2  # FFN
    )
    return float(n_blocks) * per_block


def adapter_flops(d_in: int, d_model: int, num_layers_axis: int) -> float:
    """레이어별 어파인 사영."""
    return 2.0 * num_layers_axis * d_in * d_model


__all__ = (
    "CostReport",
    "adapter_flops",
    "attention_engine_flops",
    "ssm_engine_flops",
    "transformer_forward_flops",
)
