"""시드와 결정성."""

from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> torch.Generator:
    """전역 시드를 고정하고 DataLoader용 제너레이터를 돌려준다."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:  # 일부 연산은 결정적 커널이 없다
            pass

    gen = torch.Generator()
    gen.manual_seed(seed)
    return gen


__all__ = ("set_seed",)
