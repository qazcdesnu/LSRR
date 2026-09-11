"""메커니즘 분석 (제안서 §7).

학습 경로에 개입하지 않는다 — 저장된 트레이스와 체크포인트만 읽는다.
"""

from lsrr.analysis.collapse import CollapseReport, diagnose

__all__ = ("CollapseReport", "diagnose")
