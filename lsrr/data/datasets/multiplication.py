"""다자리 곱셈 합성 — 용량 확장 검증 (제안서 §6.1).

시드에서 생성하므로 다운로드가 필요 없다. Phase 0의 두 무대 중 하나다.

이식: v1.0:lsrr/data/multiplication.py
"""

from __future__ import annotations

import random
from typing import Any

from lsrr.core.errors import ConfigError
from lsrr.core.interfaces import BaseDataModule
from lsrr.core.registry import DATA_REGISTRY
from lsrr.core.types import DataSample
from lsrr.data.schema import normalize_split

_SPLIT_SEED_OFFSET = {"train": 0, "val": 1_000_000, "test": 2_000_000}
_DEFAULT_SIZES = {"train": 20000, "val": 500, "test": 1000}


@DATA_REGISTRY.register("multiplication")
class MultiplicationDataset(BaseDataModule):
    """n자리 × n자리 곱셈.

    split마다 시드 오프셋을 달리해 학습/평가 표본이 겹치지 않게 한다 —
    합성 데이터에서 가장 흔한 사고다.
    """

    def __init__(
        self,
        digits: int = 4,
        seed: int = 42,
        sizes: dict[str, int] | None = None,
        min_operand: int | None = None,
        **_: Any,
    ) -> None:
        """
        Args:
            min_operand: 피연산자 하한. 1자리에서 `1 * n = n`처럼 **정답이 질문에
                그대로 보이는** 조합을 배제할 때 쓴다. 1자리 81조합 중 17개(21%)가
                여기 해당하며, 그대로 두면 복사만으로 21%를 맞혀 정확도가
                무의미해진다 (I6가 로드 시점에 거부한다). `min_operand=2`로 두면
                64조합이 남고 누출은 0이다.
                None이면 자릿수에서 정해지는 기본 하한을 쓴다.
        """
        self.digits = digits
        self.seed = seed
        self.sizes = {**_DEFAULT_SIZES, **(sizes or {})}
        self.min_operand = min_operand

    def get_split(self, split: str) -> list[DataSample]:
        split = normalize_split(split)
        rng = random.Random(self.seed + _SPLIT_SEED_OFFSET[split])
        lo, hi = 10 ** (self.digits - 1), 10**self.digits - 1
        if self.min_operand is not None:
            lo = max(lo, int(self.min_operand))
            if lo > hi:
                raise ConfigError(
                    f"min_operand={self.min_operand}가 {self.digits}자리 상한 {hi}를 "
                    f"넘어 표본을 만들 수 없다."
                )
        samples = []
        for _ in range(self.sizes[split]):
            a, b = rng.randint(lo, hi), rng.randint(lo, hi)
            samples.append(
                DataSample(
                    question=f"{a} * {b} =",
                    answer=str(a * b),
                    meta={"digits": self.digits, "a": a, "b": b},
                )
            )
        return samples

    def score(self, prediction: str, target: str, meta: dict[str, Any]) -> bool:
        """숫자만 남겨 비교한다."""
        return _digits_only(prediction) == _digits_only(target)


def _digits_only(text: str) -> str:
    return "".join(c for c in text if c.isdigit())


__all__ = ("MultiplicationDataset",)
