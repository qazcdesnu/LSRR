"""quasiseparable 양방향 스캔 (Hydra) — 기본 엔진 (제안서 §4.2).

    Y = shift(SS_fwd(X)) + flip(shift(SS_bwd(flip(X)))) + D·X

Hydra (Hwang, Lahoti, Dao, Gu; arXiv:2407.09941)를 따른다. 참조 구현은 shift를
`roll(y, 1, dim=1)` + 0번 위치 영치로 적용하고, 스캔에는 `D=None`을 넘겨 D를
바깥에서 정확히 한 번만 더한다.

**shift가 quasiseparable을 만든다.** SSM 스캔 출력은 위치 `l`에서 이미 `l` 자신의
기여를 담고 있다. 순·역 스캔을 그냥 더하면 대각이 두 번, 여기에 D 스킵까지 세 번
계산된다. shift는 두 스캔에서 대각을 제거해 D를 그 유일한 출처로 남긴다.

레이어 축은 **완성된 궤적**이므로 인과 마스킹이 불필요하다 — 정제는 필터링이 아니라
평활화(smoothing)이며 역방향 패스를 원리적으로 요구한다. 그래서 휴리스틱 결합인
`bidir_add`가 아니라 행렬 클래스로서의 `hydra_qs`가 기본이다.

`disable_backward=True`면 스캔이 하나뿐이라 중복 계산이 없고, 따라서 shift를
적용하지 않는다 — 표준 semiseparable Mamba로 축퇴하며 `mamba_up`과 수치적으로
같아진다. 이 동치가 구현 정확성의 근거다 (`tests/unit/test_hydra_port_fidelity.py`).

참조 구현과의 차이: 여기서 D는 채널별 정적 파라미터이나 Hydra는 데이터 의존
헤드별 사영을 쓴다. 이렇게 두면 `hydra_qs`와 `bidir_add`가 **shift 하나로만**
갈리므로 Ablation C가 quasiseparable 구조를 분리해 측정한다 — 스킵 파라미터화와
교락되지 않는다.

이식: Legacy_LSRR/lsrr/engines/hydra_qs.py
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from lsrr.core.registry import ENGINE_REGISTRY
from lsrr.engine.scan import selective_scan_sequential, shift_layers
from lsrr.engine.stack import PreNormResidualStack
from lsrr.engine.wrapper import EngineWrapper


class HydraQSCore(nn.Module):
    """quasiseparable 양방향 코어. `[B, L, d_model] → [B, L, d_model]`."""

    def __init__(
        self,
        d_model: int = 768,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        disable_backward: bool = False,
        **_: Any,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        self.d_inner = int(expand * d_model)
        self.disable_backward = disable_backward

        # 입력 사영 → (u, z). z는 게이팅 분기.
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # 레이어 축 depthwise conv (국소 문맥).
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            bias=True,
            padding=d_conv - 1,
            groups=self.d_inner,
        )

        a_init = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)

        # 순방향 스캔 파라미터 (하위층 → 상위층).
        self.x_proj_fwd = nn.Linear(self.d_inner, self.d_inner + 2 * d_state, bias=False)
        self.dt_proj_fwd = nn.Linear(self.d_inner, self.d_inner, bias=True)
        self.A_log_fwd = nn.Parameter(torch.log(a_init))

        # 역방향 스캔 파라미터 (상위층 → 하위층). 방향마다 독립이어야
        # 추상화 수준이 다른 두 방향의 인과를 각각 모델링한다.
        self.x_proj_bwd = nn.Linear(self.d_inner, self.d_inner + 2 * d_state, bias=False)
        self.dt_proj_bwd = nn.Linear(self.d_inner, self.d_inner, bias=True)
        self.A_log_bwd = nn.Parameter(torch.log(a_init.clone()))

        # 대각을 담당하는 유일한 항.
        self.D = nn.Parameter(torch.ones(self.d_inner))

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def _run_branch(
        self,
        u: torch.Tensor,
        x_proj: nn.Linear,
        dt_proj: nn.Linear,
        a_log: nn.Parameter,
    ) -> torch.Tensor:
        """스캔 1방향. `[B, L, d_inner] → [B, L, d_inner]`."""
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

        y_fwd = self._run_branch(u_act, self.x_proj_fwd, self.dt_proj_fwd, self.A_log_fwd)

        if self.disable_backward:
            # 단방향: 대각이 스캔 안에 있다 (Mamba 의미론). shift 없음.
            y_total = y_fwd + self.D * u_act
        else:
            y_bwd_raw = self._run_branch(
                torch.flip(u_act, dims=[1]),
                self.x_proj_bwd,
                self.dt_proj_bwd,
                self.A_log_bwd,
            )
            # 각 스캔의 자기 방향에서 shift한 뒤 원 순서로 되돌린다.
            y_total = (
                shift_layers(y_fwd)
                + torch.flip(shift_layers(y_bwd_raw), dims=[1])
                + self.D * u_act
            )

        return self.out_proj(y_total * F.silu(z))


@ENGINE_REGISTRY.register("hydra_qs")
def build_hydra_qs(
    d_model: int = 768,
    d_state: int = 16,
    d_conv: int = 4,
    expand: int = 2,
    n_blocks: int = 2,
    disable_backward: bool = False,
    damping_alpha: float = 0.8,
    max_cycles: int = 32,
    cycle_embedding: bool = True,
    reinject_r0: str = "gate",
    norm_type: str = "rmsnorm",
    **_: Any,
) -> EngineWrapper:
    """레지스트리 진입점. 코어를 래퍼로 감싸 §4.2 갱신식을 적용한다."""
    cores = [
        HydraQSCore(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            disable_backward=disable_backward,
        )
        for _ in range(n_blocks)
    ]
    # 잔차·정규화 없이 이어 붙이면 블록의 증폭이 곱해진다 (F-029).
    core: nn.Module = PreNormResidualStack(cores)
    return EngineWrapper(
        core=core,
        d_model=d_model,
        damping_alpha=damping_alpha,
        max_cycles=max_cycles,
        cycle_embedding=cycle_embedding,
        reinject_r0=reinject_r0,
        norm_type=norm_type,
    )


__all__ = ("HydraQSCore", "build_hydra_qs")
