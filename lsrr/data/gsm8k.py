import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from lsrr.interfaces import BaseDataModule, DataSample
from lsrr.registry import DATA_REGISTRY
from lsrr.data.answer_scoring import match_final_number

@DATA_REGISTRY.register("gsm8k")
@DATA_REGISTRY.register("gsm8k_aug")
class GSM8KDataset(BaseDataModule):
    """GSM8K / GSM8K-Aug (Grade School Math 8K with step-by-step reasoning)."""
    def __init__(
        self,
        data_path: Optional[str] = "data/gsm8k-aug",
        max_train_samples: Optional[int] = None,
        num_mock_samples: int = 200,
        seed: int = 42,
        **kwargs
    ):
        p = Path(data_path) if data_path else None
        if p and not p.is_absolute() and not p.exists():
            repo_root = Path(__file__).resolve().parent.parent.parent
            if (repo_root / p).exists():
                p = repo_root / p
        self.data_path = p
        self.max_train_samples = max_train_samples
        self.num_mock_samples = num_mock_samples
        self.seed = seed

        self.splits: Dict[str, List[DataSample]] = {
            "train": [],
            "val": [],
            "test": []
        }

        if self.data_path and self.data_path.exists():
            self._load_from_path(self.data_path)
        else:
            self._generate_synthetic_gsm8k()

        if self.max_train_samples is not None and len(self.splits["train"]) > self.max_train_samples:
            self.splits["train"] = self.splits["train"][:self.max_train_samples]

    def _generate_synthetic_gsm8k(self):
        """Generate mock grade school math problems for offline testing."""
        import random
        rng = random.Random(self.seed)

        for split, count in [("train", self.num_mock_samples), ("val", 50), ("test", 50)]:
            samples = []
            for i in range(count):
                a = rng.randint(5, 50)
                b = rng.randint(2, 10)
                c = rng.randint(1, 20)
                subtotal = a * b
                total = subtotal + c

                question = f"Alice has {a} boxes with {b} apples each. She finds {c} more apples. How many apples does she have in total?"
                answer = str(total)
                cot = [
                    f"<<{a}*{b}={subtotal}>>",
                    f"<<{subtotal}+{c}={total}>>"
                ]
                samples.append(DataSample(
                    question=question,
                    answer=answer,
                    cot_steps=cot,
                    meta={"hops": len(cot), "sample_id": i}
                ))
            self.splits[split] = samples

    def _load_from_path(self, path: Path):
        """Loads splits from directory or single file."""
        if path.is_dir():
            split_candidates = {
                "train": ["train.json", "train.jsonl", "gsm8k_train.json", "gsm8k_train.jsonl"],
                "val": ["valid.json", "valid.jsonl", "val.json", "val.jsonl", "gsm8k_valid.json", "gsm8k_val.json"],
                "test": ["test.json", "test.jsonl", "gsm8k_test.json", "gsm8k_test.jsonl"]
            }
            for split, filenames in split_candidates.items():
                for fname in filenames:
                    fpath = path / fname
                    if fpath.exists():
                        self.splits[split] = self._read_file(fpath)
                        break
        elif path.is_file():
            self.splits["train"] = self._read_file(path)

    def _read_file(self, fpath: Path) -> List[DataSample]:
        samples = []
        with open(fpath, "r", encoding="utf-8") as f:
            if fpath.suffix == ".json":
                data = json.load(f)
                if isinstance(data, list):
                    raw_items = data
                elif isinstance(data, dict):
                    raw_items = data.get("data", data.get("samples", []))
                else:
                    raw_items = []
            else:
                raw_items = [json.loads(line) for line in f if line.strip()]

            for idx, item in enumerate(raw_items):
                steps = item.get("steps", item.get("cot_steps", []))
                meta = item.get("meta", {})
                if not isinstance(meta, dict):
                    meta = {}

                if "hops" not in meta:
                    meta["hops"] = len(steps) if steps else 1
                if "sample_id" not in meta:
                    meta["sample_id"] = idx

                # Parse answer: if contains ####, extract portion after ####
                raw_answer = str(item.get("answer", item.get("label", ""))).strip()
                if "####" in raw_answer:
                    answer = raw_answer.split("####")[-1].strip().replace(",", "")
                else:
                    answer = raw_answer.replace(",", "")

                samples.append(DataSample(
                    question=item.get("question", item.get("text", "")),
                    answer=answer,
                    cot_steps=steps,
                    meta=meta
                ))
        return samples

    def get_split(self, split: str) -> List[DataSample]:
        if split in self.splits:
            return self.splits[split]
        if split == "validation" and "val" in self.splits:
            return self.splits["val"]
        raise KeyError(f"Unknown split '{split}', available: {list(self.splits.keys())}")

    def evaluate_answer(self, prediction: str, target: str, meta: Dict[str, Any]) -> bool:
        """Final-answer exact match, the protocol used by the CoT/latent-reasoning baselines.

        Only the last number the model emits counts (or the one after '####' when the
        model emits that delimiter). A gold answer appearing mid-generation does not.
        """
        return match_final_number(prediction, target)
