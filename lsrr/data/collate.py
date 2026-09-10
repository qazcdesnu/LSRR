"""배치 구성 (I6).

`PromptEncoder`가 실제 토크나이즈를 하고, 여기서는 DataLoader 계약만 맞춘다.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Sequence

import torch

from lsrr.core.types import DataSample
from lsrr.data.prompting import PromptEncoder


class SampleCollator:
    """`list[DataSample]` → 배치 dict.

    `torch.utils.data.DataLoader(collate_fn=...)`에 그대로 넘긴다.
    """

    def __init__(
        self,
        encoder: PromptEncoder,
        device: Optional[torch.device] = None,
        check_leakage: bool = True,
    ) -> None:
        self.encoder = encoder
        self.device = device
        self.check_leakage = check_leakage

    def __call__(self, samples: Sequence[DataSample]) -> dict[str, Any]:
        return self.encoder.encode_batch(
            samples, device=self.device, check_leakage=self.check_leakage
        )


class SampleDataset(torch.utils.data.Dataset):
    """`list[DataSample]`을 감싸는 최소 Dataset."""

    def __init__(self, samples: Sequence[DataSample]) -> None:
        self.samples = list(samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> DataSample:
        return self.samples[idx]


def make_loader(
    samples: Sequence[DataSample],
    encoder: PromptEncoder,
    batch_size: int = 16,
    shuffle: bool = False,
    device: Optional[torch.device] = None,
    generator: Optional[torch.Generator] = None,
    **kwargs: Any,
) -> torch.utils.data.DataLoader:
    return torch.utils.data.DataLoader(
        SampleDataset(samples),
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=SampleCollator(encoder, device=device),
        generator=generator,
        **kwargs,
    )


__all__ = ("SampleCollator", "SampleDataset", "make_loader")
