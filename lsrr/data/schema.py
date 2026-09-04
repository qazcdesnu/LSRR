from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from lsrr.interfaces import DataSample, BaseDataModule
from lsrr.registry import DATA_REGISTRY

@dataclass
class StandardDataBatch:
    questions: List[str]
    answers: List[str]
    cot_steps: List[List[str]]
    meta: List[Dict[str, Any]]
    input_ids: Optional[Any] = None
    attention_mask: Optional[Any] = None
    target_ids: Optional[Any] = None
