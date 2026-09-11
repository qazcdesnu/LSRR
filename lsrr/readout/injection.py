"""주입 공간 정합 (I8, ADR-013).

`h_fusion`은 백본이 **입력 임베딩 자리에서** 받는 벡터다. 차원이 맞는 것만으로는
부족하다 — 스케일이 맞아야 한다.

측정된 사실 (GPT-2, M2):
    입력 임베딩 노름 ≈ 3.0   vs   h⁽ᴸ⁾(ln_f 후) 노름 ≈ 236.6   → 79배 불일치

제안서 §4.4는 Coconut의 연속 사고 주입과 같은 통로를 쓴다고 명시하지만, Coconut은
백본을 미세조정하므로 모델이 불일치에 적응한다. **Phase A 에는 적응할 주체가
없으므로** 보정은 주입 경로의 책임이다.

Phase B 의 LoRA 가 수신 측을 정렬하더라도(ADR-014) 이 보정은 남는다. 79배
불일치를 0.65% 파라미터의 델타로 흡수시키는 것은 정렬이 아니라 스케일 교정을
LoRA 에 떠넘기는 것이고, 그러면 Phase A 체크포인트가 성립하지 않아 §5.0 의
2원화 분해(= Ablation B)가 무너진다. **보정이 먼저, 정렬이 그 위다.**
"""

from __future__ import annotations

from typing import Any, Literal, Optional

import torch
import torch.nn as nn

from lsrr.core.errors import AssemblyError
from lsrr.core.invariants import assert_injection_space

CalibrationMode = Literal["none", "rms", "learned_rms", "fixed_scale"]


def embedding_rms(model: nn.Module) -> float:
    """백본 입력 임베딩 행렬의 성분 RMS.

    주입 벡터가 맞춰야 할 목표 스케일이다. 노름이 아니라 성분 RMS를 쓰는 이유는
    폭(d_in)이 다른 백본 사이에서 그대로 옮겨지기 때문이다.
    """
    weight = model.get_input_embeddings().weight
    with torch.no_grad():
        return float(weight.float().pow(2).mean(dim=-1).sqrt().mean())


class InjectionCalibrator(nn.Module):
    """주입 벡터를 백본 입력 임베딩 분포로 되돌린다.

    `h_fusion = h_ctx + W_r·h_SSM`의 **합 전체**를 정규화한다. 방향은 보존되므로
    잔차 구조(작은 W_r 출력이 h_ctx를 조금 움직인다)는 그대로 유지되고, 학습
    도중 W_r 출력이 커져도 주입은 항상 분포 안에 머문다.

    Args:
        mode: 보정 방식.
            - `none`: 보정 없음. 진단 전용.
            - `rms`: 샘플별 RMS 정규화 후 목표 RMS로 스케일. 고정.
            - `learned_rms`: 위와 같되 게인이 학습 스칼라 (기본).
            - `fixed_scale`: 상수배만 한다. 야코비안이 c·I이므로 방향을 왜곡하지
              않고 상류 그래디언트를 균일하게만 스케일한다. 대신 샘플별 노름
              편차가 남아 W_r이 커지면 분포를 벗어날 수 있다.
        target_rms: 목표 성분 RMS. `from_backbone`으로 백본에서 재는 것이 정석이다.
    """

    def __init__(
        self,
        d_in: int,
        mode: CalibrationMode = "learned_rms",
        target_rms: Optional[float] = None,
        eps: float = 1e-6,
        **_: Any,
    ) -> None:
        super().__init__()
        if mode not in ("none", "rms", "learned_rms", "fixed_scale"):
            raise AssemblyError(f"주입 보정 모드 '{mode}'를 모른다.")
        if mode != "none" and target_rms is None:
            raise AssemblyError(
                f"mode='{mode}'는 target_rms를 요구한다. "
                f"InjectionCalibrator.from_backbone(model)을 쓰라."
            )
        self.d_in = d_in
        self.mode = mode
        self.eps = eps
        self.base_rms = float(target_rms or 0.0)
        if mode == "learned_rms":
            self.gain = nn.Parameter(torch.tensor(self.base_rms))
        else:
            self.register_buffer("gain", torch.tensor(self.base_rms), persistent=False)
        self.register_buffer(
            "source_rms", torch.tensor(float("nan")), persistent=False
        )  # fixed_scale 모드에서 관측된 입력 RMS

    @classmethod
    def from_backbone(
        cls, model: nn.Module, mode: CalibrationMode = "learned_rms", **kwargs: Any
    ) -> "InjectionCalibrator":
        return cls(
            d_in=int(model.config.hidden_size),
            mode=mode,
            target_rms=embedding_rms(model),
            **kwargs,
        )

    def forward(self, h_fusion: torch.Tensor) -> torch.Tensor:
        """[B, d_in] → [B, d_in]. 차원은 항상 검사한다 (I8)."""
        assert_injection_space(h_fusion, self.d_in)
        if self.mode == "none":
            return h_fusion
        if self.mode == "fixed_scale":
            if torch.isnan(self.source_rms):
                raise AssemblyError(
                    "fixed_scale 모드는 calibrate_from_batch()로 입력 RMS를 먼저 "
                    "관측해야 한다."
                )
            return h_fusion * (self.gain / self.source_rms.clamp(min=self.eps))
        rms = h_fusion.float().pow(2).mean(dim=-1, keepdim=True).sqrt().clamp(min=self.eps)
        return (h_fusion.float() / rms * self.gain).to(h_fusion.dtype)

    @torch.no_grad()
    def calibrate_from_batch(self, h_fusion: torch.Tensor) -> float:
        """fixed_scale 모드의 상수를 정하기 위해 입력 RMS를 관측한다."""
        observed = float(h_fusion.float().pow(2).mean(dim=-1).sqrt().mean())
        self.source_rms.fill_(observed)
        return observed

    def scale_report(self, h_fusion: torch.Tensor) -> dict[str, float]:
        """보정 전후 노름. 런 메타·진단에 기록한다."""
        with torch.no_grad():
            before = float(h_fusion.norm(dim=-1).mean())
            after = float(self(h_fusion).norm(dim=-1).mean())
        return {
            "norm_before": before,
            "norm_after": after,
            "ratio": before / max(after, 1e-9),
            "mode": self.mode,
        }

    def extra_repr(self) -> str:
        return f"d_in={self.d_in}, mode={self.mode}, target_rms={self.base_rms:.4f}"


__all__ = ("InjectionCalibrator", "embedding_rms", "CalibrationMode")
