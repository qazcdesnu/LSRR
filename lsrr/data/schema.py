"""데이터 스키마와 split 규약."""

from __future__ import annotations

from typing import Final

from lsrr.core.types import DataSample

TRAIN: Final = "train"
VAL: Final = "val"
TEST: Final = "test"
SPLITS: Final = (TRAIN, VAL, TEST)

SPLIT_ALIASES: Final = {"valid": VAL, "validation": VAL, "dev": VAL, "eval": TEST}


def normalize_split(split: str) -> str:
    """`valid`/`validation`/`dev`를 `val`로 통일한다.

    데이터셋마다 split 이름이 달라 조용히 빈 split을 읽는 사고가 잦다.
    """
    s = SPLIT_ALIASES.get(split, split)
    if s not in SPLITS:
        raise ValueError(f"split '{split}'을 모른다. 가능: {SPLITS} (별칭: {sorted(SPLIT_ALIASES)})")
    return s


__all__ = ("DataSample", "TRAIN", "VAL", "TEST", "SPLITS", "normalize_split")
