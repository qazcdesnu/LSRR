"""레이어 축 정제 연산자 (제안서 §4.2).

엔진은 루프를 소유하지 않는다 — `m`은 조건 입력일 뿐이다.
"""

from lsrr.engine.core_mlp import MLPCore, build_mlp_onepass
from lsrr.engine.wrapper import EngineWrapper

__all__ = ("EngineWrapper", "MLPCore", "build_mlp_onepass")
