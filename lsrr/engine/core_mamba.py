"""단방향·휴리스틱 양방향 SSM — Ablation C의 semiseparable 조건 (제안서 §7).

믹서 행렬 클래스로 보면 이들은 **semiseparable**(삼각)이다. `hydra_qs`의
quasiseparable과 대비되는 지점이 세 가지다.

- `mamba_up`   하위층 → 상위층 단방향. 어휘 → 의미 방향의 인과만 모델링한다.
- `mamba_down` 상위층 → 하위층 단방향. 역방향만.
- `bidir_add`  두 스캔을 **그냥 더한다**. shift가 없으므로 대각이 세 번 계산되며
  quasiseparable이 아니다. `hydra_qs`와 **shift 하나로만** 갈리므로, 둘의 차이가
  곧 "행렬 클래스를 제대로 맞추는 것의 값어치"다.

`bidir_add`가 존재하는 이유는 휴리스틱 양방향의 열세를 실측으로 보이기 위해서다.
"양방향이면 다 같다"는 반론을 막는 조건이므로 Ablation C에서 빠지면 안 된다.

이식: Legacy_LSRR/lsrr/engines/mamba_up_down.py
"""

from __future__ import annotations

from typing import Any, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from lsrr.core.errors import ConfigError
from lsrr.core.registry import ENGINE_REGISTRY
from lsrr.engine.scan import selective_scan_sequential
from lsrr.engine.stack import PreNormResidualStack
from lsrr.engine.wrapper import EngineWrapper

Direction = Literal["up", "down", "bidir_add"]


class DirectionalSSMCore(nn.Module):
    """방향이 지정된 SSM 코어. `[B, L, d_model] → [B, L, d_model]`."""

    def __init__(
        self,
        d_model: int = 768,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        direction: Direction = "up",
        **_: Any,
    ) -> None:
        super().__init__()
        if direction not in ("up", "down", "bidir_add"):
            raise ConfigError(f"direction '{direction}'를 모른다.")

        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        self.d_inner = int(expand * d_model)
        self.direction = direction

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            bias=True,
            padding=d_conv - 1,
            groups=self.d_inner,
        )

        a_init = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)

        # 순방향 파라미터는 항상 둔다 — 'down'에서도 이름을 유지해야 `hydra_qs`와
        # 파라미터를 맞대어 복사하는 동치 테스트가 성립한다.
        self.x_proj_fwd = nn.Linear(self.d_inner, self.d_inner + 2 * d_state, bias=False)
        self.dt_proj_fwd = nn.Linear(self.d_inner, self.d_inner, bias=True)
        self.A_log_fwd = nn.Parameter(torch.log(a_init))

        if direction in ("down", "bidir_add"):
            self.x_proj_bwd = nn.Linear(
                self.d_inner, self.d_inner + 2 * d_state, bias=False
            )
            self.dt_proj_bwd = nn.Linear(self.d_inner, self.d_inner, bias=True)
            self.A_log_bwd = nn.Parameter(torch.log(a_init.clone()))

        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def _run_branch(
        self,
        u: torch.Tensor,
        x_proj: nn.Linear,
        dt_proj: nn.Linear,
        a_log: nn.Parameter,
    ) -> torch.Tensor:
        dt_raw, B, C = torch.split(
            x_proj(u), [self.d_inner, self.d_state, self.d_state], dim=-1
        )
        delta = F.softplus(dt_proj(dt_raw))
        A = -torch.exp(a_log.float())
        return selective_scan_sequential(u, delta, A, B, C)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """`[B, L, d_model] → [B, L, d_model]`."""
        length = x.shape[1]
        u, z = torch.split(self.in_proj(x), [self.d_inner, self.d_inner], dim=-1)
        u_act = F.silu(self.conv1d(u.transpose(1, 2))[:, :, :length].transpose(1, 2))

        if self.direction == "up":
            y = self._run_branch(
                u_act, self.x_proj_fwd, self.dt_proj_fwd, self.A_log_fwd
            )
        elif self.direction == "down":
            y = torch.flip(
                self._run_branch(
                    torch.flip(u_act, dims=[1]),
                    self.x_proj_bwd,
                    self.dt_proj_bwd,
                    self.A_log_bwd,
                ),
                dims=[1],
            )
        else:  # bidir_add — shift 없음. 이것이 hydra_qs 와의 유일한 차이다.
            y = self._run_branch(
                u_act, self.x_proj_fwd, self.dt_proj_fwd, self.A_log_fwd
            ) + torch.flip(
                self._run_branch(
                    torch.flip(u_act, dims=[1]),
                    self.x_proj_bwd,
                    self.dt_proj_bwd,
                    self.A_log_bwd,
                ),
                dims=[1],
            )

        return self.out_proj((y + self.D * u_act) * F.silu(z))


def _build_directional(direction: Direction, **kwargs: Any) -> EngineWrapper:
    d_model = int(kwargs.get("d_model", 768))
    n_blocks = int(kwargs.get("n_blocks", 2))
    cores = [
        DirectionalSSMCore(
            d_model=d_model,
            d_state=int(kwargs.get("d_state", 16)),
            d_conv=int(kwargs.get("d_conv", 4)),
            expand=int(kwargs.get("expand", 2)),
            direction=direction,
        )
        for _ in range(n_blocks)
    ]
    # 잔차·정규화 없이 이어 붙이면 블록의 증폭이 곱해진다 (F-029).
    core: nn.Module = PreNormResidualStack(cores)
    return EngineWrapper(
        core=core,
        d_model=d_model,
        damping_alpha=float(kwargs.get("damping_alpha", 0.8)),
        max_cycles=int(kwargs.get("max_cycles", 32)),
        cycle_embedding=bool(kwargs.get("cycle_embedding", True)),
        reinject_r0=kwargs.get("reinject_r0", "gate"),
        norm_type=kwargs.get("norm_type", "rmsnorm"),
    )


@ENGINE_REGISTRY.register("mamba_up")
def build_mamba_up(**kwargs: Any) -> EngineWrapper:
    """하위층 → 상위층 단방향 (semiseparable)."""
    return _build_directional("up", **kwargs)


@ENGINE_REGISTRY.register("mamba_down")
def build_mamba_down(**kwargs: Any) -> EngineWrapper:
    """상위층 → 하위층 단방향 (semiseparable)."""
    return _build_directional("down", **kwargs)


@ENGINE_REGISTRY.register("bidir_add")
def build_bidir_add(**kwargs: Any) -> EngineWrapper:
    """휴리스틱 양방향 — shift 없이 두 스캔을 더한다. quasiseparable이 아니다."""
    return _build_directional("bidir_add", **kwargs)


__all__ = (
    "DirectionalSSMCore",
    "build_mamba_up",
    "build_mamba_down",
    "build_bidir_add",
)
