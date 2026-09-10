"""계약 층 — 인터페이스·데이터 타입·레지스트리·불변식 단언.

이 패키지는 lsrr의 다른 어떤 것도 import하지 않는다 (레이어 L0).
"""

from lsrr.core import errors, interfaces, invariants, registry, types
from lsrr.core.registry import ALL_REGISTRIES, Registry, registry_snapshot
from lsrr.core.types import (
    ContextBundle,
    CostReport,
    CycleDiagnostics,
    DataSample,
    GateVerdict,
    ReadoutResult,
    ReasoningTrace,
    TerminationSignals,
)

__all__ = (
    "errors",
    "interfaces",
    "invariants",
    "registry",
    "types",
    "Registry",
    "ALL_REGISTRIES",
    "registry_snapshot",
    "ContextBundle",
    "CostReport",
    "CycleDiagnostics",
    "DataSample",
    "GateVerdict",
    "ReadoutResult",
    "ReasoningTrace",
    "TerminationSignals",
)
