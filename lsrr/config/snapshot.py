"""설정 스냅샷과 런 신원.

모든 런은 해석 완료된 설정 전체를 `runs/<run_id>/config.yaml`에 스냅샷하며,
그 해시가 런 신원이다 (CONVENTIONS.md §2). 캐시 키와 런 인덱스 조회에 쓴다.
"""

from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from omegaconf import DictConfig, OmegaConf

from lsrr.config.schema import get_path


def config_hash(cfg: DictConfig, length: int = 12) -> str:
    """해석 완료 설정의 결정적 해시.

    키 정렬 후 해싱하므로 병합 순서가 달라도 같은 내용이면 같은 해시가 나온다.
    """
    container = OmegaConf.to_container(cfg, resolve=True)
    canonical = _canonicalize(container)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:length]


def _canonicalize(node: Any, indent: int = 0) -> str:
    pad = "  " * indent
    if isinstance(node, dict):
        return "".join(
            f"{pad}{k}:\n{_canonicalize(node[k], indent + 1)}" for k in sorted(node)
        )
    if isinstance(node, list):
        return "".join(f"{pad}-\n{_canonicalize(v, indent + 1)}" for v in node)
    return f"{pad}{node!r}\n"


def make_run_id(cfg: DictConfig, exp_name: str, seed: int = 0) -> str:
    """`<exp>_<engine>_<backbone>_<YYYYMMDD_HHMMSS>_s<seed>` (레거시 규약 계승)."""
    engine = get_path(cfg, "engine.type", "engine")
    backbone = get_path(cfg, "backbone.type", "backbone")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{exp_name}_{engine}_{backbone}_{stamp}_s{seed}"


def save_snapshot(cfg: DictConfig, path: Path) -> Path:
    """해석 완료 설정을 YAML로 덤프한다."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8")
    return path


def load_snapshot(path: Path) -> DictConfig:
    """스냅샷을 다시 읽는다 (재개·평가·보고에서 쓴다)."""
    return OmegaConf.load(Path(path))  # type: ignore[return-value]


def run_metadata(
    cfg: DictConfig,
    exp_name: str,
    seed: int = 0,
    extra: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """런 메타의 설정 유래 부분.

    백본 해시·학습 파라미터 수 등 실행 시점에만 알 수 있는 값은 `extra`로 받는다.
    안정화 단·I2 예외 플래그처럼 **표에 병기해야 하는 값**을 여기서 고정한다.
    """
    meta: dict[str, Any] = {
        "run_id": make_run_id(cfg, exp_name, seed),
        "exp_name": exp_name,
        "seed": seed,
        "config_hash": config_hash(cfg),
        "engine": get_path(cfg, "engine.type"),
        "backbone": get_path(cfg, "backbone.type"),
        "memory_scope": get_path(cfg, "memory.scope.type"),
        "termination": get_path(cfg, "termination.type"),
        "stability_rung": get_path(cfg, "stability.type", "none"),
        "deep_supervision": bool(
            get_path(cfg, "objective.deep_supervision.enabled", False)
        ),
        "reencoding_loop": bool(get_path(cfg, "experimental.reencoding_loop", False)),
    }
    if extra:
        meta.update(extra)
    return meta


__all__ = (
    "config_hash",
    "make_run_id",
    "save_snapshot",
    "load_snapshot",
    "run_metadata",
)
