import random
import re
from typing import List, Dict, Any
from lsrr.interfaces import BaseDataModule, DataSample
from lsrr.registry import DATA_REGISTRY
from lsrr.data.answer_scoring import match_final_number

@DATA_REGISTRY.register("multiplication")
class MultiplicationDataset(BaseDataModule):
    """Synthetic multi-digit multiplication dataset generator (4x4 to 9x9)."""
    def __init__(
        self,
        digits: int = 4,
        num_train: int = 1000,
        num_val: int = 200,
        num_test: int = 200,
        seed: int = 42,
        **kwargs
    ):
        self.digits = digits
        self.num_train = num_train
        self.num_val = num_val
        self.num_test = num_test
        self.seed = seed

        self.min_val = 10 ** (digits - 1)
        self.max_val = (10 ** digits) - 1

        self.splits: Dict[str, List[DataSample]] = {
            "train": self._generate_split(num_train, seed + 101),
            "val": self._generate_split(num_val, seed + 202),
            "test": self._generate_split(num_test, seed + 303),
        }

    def _generate_split(self, count: int, split_seed: int) -> List[DataSample]:
        rng = random.Random(split_seed)
        samples = []
        for i in range(count):
            a = rng.randint(self.min_val, self.max_val)
            b = rng.randint(self.min_val, self.max_val)
            product = a * b
            question = f"Calculate {a} * {b}."
            answer = str(product)
            # Intermediate step simulation for meta
            cot_steps = [f"{a} * {b} = {product}"]
            meta = {
                "a": a,
                "b": b,
                "digits": self.digits,
                "hops": self.digits,  # complexity proportional to digits
                "sample_id": i
            }
            samples.append(DataSample(
                question=question,
                answer=answer,
                cot_steps=cot_steps,
                meta=meta
            ))
        return samples

    def get_split(self, split: str) -> List[DataSample]:
        if split not in self.splits:
            raise KeyError(f"Unknown split '{split}', available: {list(self.splits.keys())}")
        return self.splits[split]

    def evaluate_answer(self, prediction: str, target: str, meta: Dict[str, Any]) -> bool:
        """Final-answer exact match: only the last number the model emits counts."""
        return match_final_number(prediction, target)
