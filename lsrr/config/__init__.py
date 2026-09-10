"""설정 처리 — 로드·검증·스윕·스냅샷.

`lsrr/config/`는 설정을 다루는 코드, `configs/`는 설정 파일(YAML)이다.
"""

from lsrr.config.loader import (
    DEFAULT_CONFIG_DIR,
    PROJECT_ROOT,
    load_config,
    parse_cli,
    resolve_hierarchical,
)
from lsrr.config.schema import SLOTS, SlotSpec, get_path
from lsrr.config.snapshot import (
    config_hash,
    load_snapshot,
    make_run_id,
    run_metadata,
    save_snapshot,
)
from lsrr.config.sweep import expand_sweep, sweep_size
from lsrr.config.validate import validate_config

__all__ = (
    "PROJECT_ROOT",
    "DEFAULT_CONFIG_DIR",
    "load_config",
    "parse_cli",
    "resolve_hierarchical",
    "validate_config",
    "expand_sweep",
    "sweep_size",
    "config_hash",
    "make_run_id",
    "save_snapshot",
    "load_snapshot",
    "run_metadata",
    "SLOTS",
    "SlotSpec",
    "get_path",
)
