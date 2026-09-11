"""테스트 공용 픽스처와 **더미 슬롯 구현**.

M1 시점에는 실제 슬롯 구현이 없다. 계약 층이 끝까지 조립되는지 확인하려면
인터페이스만 만족하는 최소 구현이 필요하고, 그것을 테스트 안에 둔다 — 이렇게
하면 `core`/`config`의 검증이 하위 패키지 구현 일정에 묶이지 않는다.

더미들은 import 시점에 레지스트리에 등록된다.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import pytest
import torch
import torch.nn as nn

from lsrr.core.interfaces import (
    BaseContextEncoder,
    BaseDataModule,
    BaseCycleRunner,
    BaseCycleSchedule,
    BaseFusionHead,
    BaseLayerAdapter,
    BaseMemoryComposer,
    BaseMemoryScope,
    BaseReadoutPath,
    BaseRefinementEngine,
    BaseTerminationRule,
    CycleHook,
)
from lsrr.core.registry import (
    ADAPTER_REGISTRY,
    BACKBONE_REGISTRY,
    COMPOSER_REGISTRY,
    DATA_REGISTRY,
    ENGINE_REGISTRY,
    FUSION_REGISTRY,
    READOUT_REGISTRY,
    SCHEDULE_REGISTRY,
    SCOPE_REGISTRY,
    STABILITY_REGISTRY,
    TERMINATION_REGISTRY,
)
from lsrr.core.types import (
    ContextBundle,
    CycleDiagnostics,
    ReadoutResult,
    ReasoningTrace,
    TerminationSignals,
)

D_IN = 32
NUM_LAYERS = 6
VOCAB = 50


# ------------------------------------------------------------ 더미 슬롯


@BACKBONE_REGISTRY.register("_dummy_backbone")
class DummyEncoder(BaseContextEncoder):
    """토큰 임베딩 하나로 레이어별 상태를 흉내 내는 동결 인코더."""

    def __init__(self, d_in: int = D_IN, num_layers: int = NUM_LAYERS, **_: Any) -> None:
        self._d_in = d_in
        self._num_layers = num_layers
        self.embed = nn.Embedding(VOCAB, d_in)
        for p in self.embed.parameters():
            p.requires_grad_(False)
        self.calls = 0

    def encode(
        self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> ContextBundle:
        self.calls += 1
        B, T = input_ids.shape
        with torch.no_grad():
            tok = self.embed(input_ids)  # [B, T, d]
            last = tok[:, -1, :]
            H_last = last.unsqueeze(1).repeat(1, self._num_layers, 1)
            H_pool = tok.mean(dim=1).unsqueeze(1).repeat(1, self._num_layers, 1)
        return ContextBundle(
            H_last=H_last,
            h_ctx=last,
            H_pool=H_pool,
            kv_cache={"dummy": True},
            attention_mask=attention_mask,
            meta={"backbone_id": "dummy"},
        )

    @property
    def num_layers(self) -> int:
        return self._num_layers

    @property
    def hidden_dim(self) -> int:
        return self._d_in

    def weight_hash(self) -> str:
        return "dummyhash"

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.embed.parameters())


@COMPOSER_REGISTRY.register("_dummy_composer")
class DummyComposer(BaseMemoryComposer):
    def __init__(self, d_in: int = D_IN, **_: Any) -> None:
        super().__init__()
        self.gate = nn.Linear(2 * d_in, d_in)

    def forward(
        self, H_last: torch.Tensor, H_pool: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if H_pool is None:
            return H_last
        g = torch.sigmoid(self.gate(torch.cat([H_last, H_pool], dim=-1)))
        return (1.0 - g) * H_last + g * H_pool


@SCOPE_REGISTRY.register("_dummy_scope_all")
class DummyScopeAll(BaseMemoryScope):
    def __init__(self, num_layers: int = NUM_LAYERS, **_: Any) -> None:
        super().__init__()
        self.num_layers = num_layers

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        return H

    def layer_indices(self, num_layers: int) -> list[int]:
        return list(range(num_layers))


@SCOPE_REGISTRY.register("_dummy_scope_final")
class DummyScopeFinal(BaseMemoryScope):
    def __init__(self, **_: Any) -> None:
        super().__init__()

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        return H[:, -1:, :]

    def layer_indices(self, num_layers: int) -> list[int]:
        return [num_layers - 1]


@ADAPTER_REGISTRY.register("_dummy_adapter")
class DummyAdapter(BaseLayerAdapter):
    def __init__(
        self, d_in: int = D_IN, d_model: int = D_IN, num_layers: int = NUM_LAYERS, **_: Any
    ) -> None:
        super().__init__()
        self._d_model = d_model
        self.weight = nn.Parameter(torch.randn(num_layers, d_in, d_model) * 0.02)
        self.bias = nn.Parameter(torch.zeros(num_layers, d_model))

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        L = H.shape[1]
        return torch.einsum("bli,lio->blo", H, self.weight[:L]) + self.bias[:L]

    @property
    def d_model(self) -> int:
        return self._d_model


@ENGINE_REGISTRY.register("_dummy_engine")
class DummyEngine(BaseRefinementEngine):
    """감쇠 갱신만 흉내 내는 최소 엔진 (수축하도록 만들어 종료를 테스트한다)."""

    def __init__(self, d_model: int = D_IN, damping_alpha: float = 0.5, **_: Any) -> None:
        super().__init__()
        self.alpha = damping_alpha
        self.proj = nn.Linear(d_model, d_model)

    def forward_step(self, R_m: torch.Tensor, R0: torch.Tensor, m: int) -> torch.Tensor:
        f = torch.tanh(self.proj(R_m)) * 0.1 + R0 * 0.1
        return (1.0 - self.alpha) * R_m + self.alpha * f


@SCHEDULE_REGISTRY.register("_dummy_schedule")
class DummySchedule(BaseCycleSchedule):
    def __init__(self, mean: float = 6, max: int = 16, gamma: float = 0.85, **_: Any) -> None:
        self.M = int(mean)
        self.max = int(max)
        self.gamma = gamma

    def sample_M(self) -> int:
        return min(self.M, self.max)

    def weights(self, M: int) -> list[float]:
        return [self.gamma ** (M - m) for m in range(M)]


@TERMINATION_REGISTRY.register("_dummy_termination")
class DummyTermination(BaseTerminationRule):
    def __init__(self, eps: float = 1e-3, m_max: int = 32, **_: Any) -> None:
        self.eps = eps
        self.m_max = m_max
        self.stopped: Optional[torch.Tensor] = None

    def reset(self, batch_size: int, device: torch.device) -> None:
        self.stopped = torch.zeros(batch_size, dtype=torch.bool, device=device)

    def should_stop(
        self, signals: TerminationSignals, m: int
    ) -> tuple[torch.Tensor, CycleDiagnostics]:
        if self.stopped is None:
            self.reset(signals.delta_state.shape[0], signals.delta_state.device)
        meets = signals.delta_state < self.eps
        fallback = (m + 1) >= self.m_max  # I5: 폴백은 우회 불가
        self.stopped = self.stopped | meets | fallback
        return self.stopped, CycleDiagnostics(
            m=m, delta_state=float(signals.delta_state.mean())
        )

    def needs_logits(self) -> bool:
        return False


@FUSION_REGISTRY.register("_dummy_fusion")
class DummyFusion(BaseFusionHead):
    def __init__(
        self, d_model: int = D_IN, d_out: int = D_IN, w_r_init_scale: float = 0.01, **_: Any
    ) -> None:
        super().__init__()
        self.score = nn.Linear(d_model, 1, bias=False)
        self.w_r = nn.Linear(d_model, d_out)
        with torch.no_grad():
            self.w_r.weight.mul_(w_r_init_scale)
            self.w_r.bias.zero_()

    def forward(
        self, R_star: torch.Tensor, h_ctx: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        alpha = torch.softmax(self.score(R_star).squeeze(-1), dim=-1)
        h_ssm = torch.einsum("bl,bld->bd", alpha, R_star)
        return h_ctx + self.w_r(h_ssm), alpha


@READOUT_REGISTRY.register("_dummy_readout")
class DummyReadout(BaseReadoutPath):
    """전 사이클 공유 판독 경로 (I3)."""

    def __init__(self, fusion: Any = None, d_in: int = D_IN, **_: Any) -> None:
        super().__init__()
        self.fusion = fusion
        self.head = nn.Linear(d_in, VOCAB, bias=False)
        for p in self.head.parameters():
            p.requires_grad_(False)  # 백본 소유 판독을 흉내 낸다
        self.call_count = 0

    def readout(
        self,
        R: torch.Tensor,
        h_ctx: torch.Tensor,
        context: ContextBundle,
        answer_ids: Optional[torch.Tensor] = None,
        m: Optional[int] = None,
        prefix: Optional[Sequence[torch.Tensor]] = None,
    ) -> ReadoutResult:
        self.call_count += 1
        states = [*prefix, R] if prefix else [R]
        fused = [self.fusion(st, h_ctx) for st in states]
        trajectory = torch.stack([h for h, _ in fused], dim=1)
        h_fusion, alpha = fused[-1]
        T = answer_ids.shape[1] if answer_ids is not None else 1
        logits = self.head(h_fusion).unsqueeze(1).expand(-1, T, -1)
        return ReadoutResult(
            logits=logits, h_fusion=h_fusion, alpha=alpha, m=m,
            h_thought=trajectory,
        )


@DATA_REGISTRY.register("_dummy_data")
class DummyData(BaseDataModule):
    def __init__(self, digits: int = 4, **_: Any) -> None:
        self.digits = digits

    def get_split(self, split: str) -> list:
        return []

    def score(self, prediction: str, target: str, meta: dict) -> bool:
        return prediction.strip() == target.strip()


@STABILITY_REGISTRY.register("_dummy_stability")
class DummyStabilityNone:
    def __init__(self, **_: Any) -> None:
        pass

    def apply(self, engine: nn.Module) -> None:
        return None

    def penalty(self, R_m: torch.Tensor, R_next: torch.Tensor) -> torch.Tensor:
        return torch.zeros((), device=R_m.device)


class DummyRunner(BaseCycleRunner):
    """M4의 CycleRunner 자리를 메우는 최소 구현."""

    def __init__(self, engine: Any, termination: Any, schedule: Any) -> None:
        self.engine = engine
        self.termination = termination
        self.schedule = schedule

    def _loop(
        self, R0: torch.Tensor, M: int, hooks: Sequence[CycleHook], eval_mode: bool
    ) -> ReasoningTrace:
        trace = ReasoningTrace(R0=R0)
        R = R0
        if eval_mode:
            self.termination.reset(R0.shape[0], R0.device)
        for m in range(M):
            R_next = self.engine.forward_step(R, R0, m)
            delta = (R_next - R).detach().norm(dim=-1).mean(dim=-1)
            diag = CycleDiagnostics(m=m, delta_state=float(delta.mean()))
            if eval_mode:
                stopped, diag = self.termination.should_stop(
                    TerminationSignals(delta_state=delta), m
                )
                diag.stopped = stopped
            trace.per_cycle.append(diag)
            for h in hooks:
                h.on_cycle(m, R, R_next, diag)
            R = R_next
            if eval_mode and bool(diag.stopped.all()):
                break
        trace.R_star = R
        trace.stopping_cycles = torch.full(
            (R0.shape[0],), len(trace.per_cycle), dtype=torch.long
        )
        return trace

    def run_train(
        self, R0: torch.Tensor, hooks: Sequence[CycleHook] = (), M: Optional[int] = None
    ) -> ReasoningTrace:
        return self._loop(R0, M or self.schedule.sample_M(), hooks, eval_mode=False)

    def run_eval(
        self, R0: torch.Tensor, hooks: Sequence[CycleHook] = ()
    ) -> ReasoningTrace:
        return self._loop(R0, self.termination.m_max, hooks, eval_mode=True)


# ------------------------------------------------------------ 픽스처


@pytest.fixture
def dummy_cfg():
    """더미 슬롯으로 채운 최소 설정."""
    from omegaconf import OmegaConf

    return OmegaConf.create(
        {
            "seed": 0,
            "backbone": {"type": "_dummy_backbone", "d_in": D_IN, "num_layers": NUM_LAYERS},
            "memory": {
                "composer": {"type": "_dummy_composer"},
                "scope": {"type": "_dummy_scope_all"},
                "adapter": {"type": "_dummy_adapter"},
            },
            "engine": {"type": "_dummy_engine", "damping_alpha": 0.5},
            "recurrence": {
                "schedule": {"type": "_dummy_schedule", "mean": 4, "max": 8},
                "tbptt_k": 2,
                "hooks": {"readout_per_cycle": False},
            },
            "termination": {"type": "_dummy_termination", "eps": 1e-3, "m_max": 8},
            "stability": {"type": "_dummy_stability"},
            "readout": {
                "fusion": {"type": "_dummy_fusion", "fusion_type": "residual"},
                "path": {"type": "_dummy_readout"},
            },
            "data": {"type": "_dummy_data"},
        }
    )


@pytest.fixture
def dummy_batch():
    return {
        "input_ids": torch.randint(0, VOCAB, (3, 7)),
        "attention_mask": torch.ones(3, 7, dtype=torch.long),
        "target_ids": torch.randint(0, VOCAB, (3, 4)),
        "labels": torch.randint(0, VOCAB, (3, 4)),
    }


@pytest.fixture
def runner_factory():
    """DummyRunner 팩토리.

    `from tests.conftest import DummyRunner`로 가져가면 conftest가 별도 모듈로
    다시 실행되어 레지스트리 등록이 중복된다 — 픽스처로 넘긴다.
    """

    def _make(bundle):
        return DummyRunner(bundle.engine, bundle.termination, bundle.schedule)

    return _make


@pytest.fixture
def dummy_model(dummy_cfg):
    from lsrr.builder import build_slots
    from lsrr.model import LSRRModel

    bundle = build_slots(dummy_cfg)
    runner = DummyRunner(bundle.engine, bundle.termination, bundle.schedule)
    return LSRRModel(bundle=bundle, cfg=dummy_cfg, runner=runner)


# ------------------------------------------------- 실제 백본 (M2 이후)

GPT2_ID = "gpt2"


@pytest.fixture(scope="session")
def gpt2_backbone():
    """실제 동결 GPT-2 세션. 세션 스코프로 한 번만 로드한다."""
    from lsrr.backbone import HFFrozenCausalBackbone

    return HFFrozenCausalBackbone(GPT2_ID, device="cpu")


@pytest.fixture(scope="session")
def prompt_encoder(gpt2_backbone):
    from lsrr.data import PromptEncoder, PromptSpec

    return PromptEncoder(
        gpt2_backbone.tokenizer, PromptSpec(max_question_tokens=32, max_answer_tokens=8)
    )


@pytest.fixture(scope="session")
def mult_samples():
    from lsrr.core.registry import DATA_REGISTRY

    return DATA_REGISTRY.build({"type": "multiplication", "digits": 2}).get_split("test")[:4]


@pytest.fixture
def gpt2_batch(prompt_encoder, mult_samples):
    return prompt_encoder.encode_batch(mult_samples)


@pytest.fixture
def calibrator(gpt2_backbone):
    from lsrr.readout import InjectionCalibrator

    return InjectionCalibrator.from_backbone(gpt2_backbone.model)
