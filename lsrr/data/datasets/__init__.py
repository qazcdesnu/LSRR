"""데이터셋 모듈. import만으로 DATA 레지스트리에 등록된다."""

from lsrr.data.datasets.multiplication import MultiplicationDataset
from lsrr.data.datasets.prosqa import ProsQADataset

__all__ = ("MultiplicationDataset", "ProsQADataset")
