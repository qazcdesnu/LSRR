"""레이어 축 정제 연산자 (제안서 §4.2).

엔진은 루프를 소유하지 않는다 — `m`은 조건 입력일 뿐이다.

Ablation C의 비교 대상은 **믹서 행렬 클래스**다:
대각(`mlp_onepass`) / dense(`attn_block`) / semiseparable(`mamba_up`·`mamba_down`) /
quasiseparable(`hydra_qs`). `bidir_add`는 shift 없는 휴리스틱 양방향으로,
`hydra_qs`와 shift 하나로만 갈린다.
"""

from lsrr.engine.core_attention import AttentionCore, build_attn_block
from lsrr.engine.core_hydra import HydraQSCore, build_hydra_qs
from lsrr.engine.core_mamba import (
    DirectionalSSMCore,
    build_bidir_add,
    build_mamba_down,
    build_mamba_up,
)
from lsrr.engine.core_mlp import MLPCore, build_mlp_onepass
from lsrr.engine.scan import selective_scan_sequential, shift_layers
from lsrr.engine.wrapper import EngineWrapper

__all__ = (
    "EngineWrapper",
    "MLPCore",
    "build_mlp_onepass",
    "HydraQSCore",
    "build_hydra_qs",
    "DirectionalSSMCore",
    "build_mamba_up",
    "build_mamba_down",
    "build_bidir_add",
    "AttentionCore",
    "build_attn_block",
    "selective_scan_sequential",
    "shift_layers",
)
