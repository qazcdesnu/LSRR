import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from lsrr.interfaces import BaseDataModule, DataSample
from lsrr.registry import DATA_REGISTRY

@DATA_REGISTRY.register("prosqa")
class ProsQADataset(BaseDataModule):
    """ProsQA (Propositional Reasoning QA) with mandatory reasoning hops metadata."""
    def __init__(
        self,
        data_path: Optional[str] = None,
        num_mock_samples: int = 200,
        seed: int = 42,
        **kwargs
    ):
        self.data_path = Path(data_path) if data_path else None
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
            self._generate_synthetic_prosqa()

    def _generate_synthetic_prosqa(self):
        """Generate synthetic propositional reasoning samples with explicit 1 to 5 hops."""
        import random
        rng = random.Random(self.seed)

        entities = ["Alice", "Bob", "Charlie", "David", "Emma", "Frank", "Grace"]
        predicates = ["fawn", "gale", "hedge", "imp", "jolt", "kale", "loom", "mist", "nexus", "prism"]

        for split, count in [("train", self.num_mock_samples), ("val", 50), ("test", 50)]:
            samples = []
            for i in range(count):
                hops = rng.randint(1, 5)
                entity = rng.choice(entities)
                chain = rng.sample(predicates, hops + 1)
                
                # Build context rules: A is B, B is C, C is D...
                rules = [f"{entity} is a {chain[0]}."]
                cot = []
                for h in range(hops):
                    rules.append(f"Each {chain[h]} is a {chain[h+1]}.")
                    cot.append(f"Since {entity} is a {chain[h]}, {entity} is a {chain[h+1]}.")

                rng.shuffle(rules)
                context = " ".join(rules)
                
                # 50% True, 50% False query
                is_true = rng.choice([True, False])
                if is_true:
                    target_pred = chain[-1]
                    answer = "True"
                else:
                    distractors = [p for p in predicates if p not in chain]
                    target_pred = rng.choice(distractors) if distractors else "unknown"
                    answer = "False"

                question = f"{context} Question: Is {entity} a {target_pred}?"
                samples.append(DataSample(
                    question=question,
                    answer=answer,
                    cot_steps=cot,
                    meta={"hops": hops, "entity": entity, "target_predicate": target_pred, "sample_id": i}
                ))
            self.splits[split] = samples

    def _load_from_path(self, path: Path):
        # Support loading json / jsonl files
        if path.is_dir():
            for s in ["train", "val", "test"]:
                fpath = path / f"{s}.jsonl"
                if fpath.exists():
                    self.splits[s] = self._read_file(fpath)
        elif path.is_file():
            self.splits["train"] = self._read_file(path)

    def _read_file(self, fpath: Path) -> List[DataSample]:
        samples = []
        with open(fpath, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                item = json.loads(line)
                samples.append(DataSample(
                    question=item.get("question", item.get("text", "")),
                    answer=str(item.get("answer", item.get("label", ""))),
                    cot_steps=item.get("cot_steps", []),
                    meta=item.get("meta", {"hops": item.get("hops", 1), "sample_id": idx})
                ))
        return samples

    def get_split(self, split: str) -> List[DataSample]:
        if split not in self.splits:
            raise KeyError(f"Unknown split '{split}'")
        return self.splits[split]

    def evaluate_answer(self, prediction: str, target: str, meta: Dict[str, Any]) -> bool:
        pred_norm = prediction.strip().lower()
        target_norm = target.strip().lower()

        # Binary evaluation for True/False
        if target_norm in ["true", "false"]:
            # Check for isolated word
            tokens = re.findall(r"\b(true|false)\b", pred_norm)
            if tokens:
                return tokens[-1] == target_norm
            return target_norm in pred_norm

        return pred_norm == target_norm
