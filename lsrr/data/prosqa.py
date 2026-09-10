import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from lsrr.interfaces import BaseDataModule, DataSample
from lsrr.registry import DATA_REGISTRY
from lsrr.data.answer_scoring import match_boolean, match_free_form, normalize_text

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
        p = Path(data_path) if data_path else None
        if p and not p.is_absolute() and not p.exists():
            repo_root = Path(__file__).resolve().parent.parent.parent
            if (repo_root / p).exists():
                p = repo_root / p
        self.data_path = p
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
        # Support loading directory with json / jsonl files, or a single file
        if path.is_dir():
            split_candidates = {
                "train": ["train.json", "train.jsonl", "prosqa_train.json", "prosqa_train.jsonl"],
                "val": ["val.json", "val.jsonl", "valid.json", "valid.jsonl", "prosqa_val.json", "prosqa_valid.json", "prosqa_val.jsonl", "prosqa_valid.jsonl"],
                "test": ["test.json", "test.jsonl", "prosqa_test.json", "prosqa_test.jsonl"]
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
                cot_steps = item.get("steps", item.get("cot_steps", []))
                meta = item.get("meta", {})
                if not isinstance(meta, dict):
                    meta = {}

                if "hops" not in meta:
                    meta["hops"] = len(cot_steps) if cot_steps else item.get("hops", 1)
                if "sample_id" not in meta:
                    meta["sample_id"] = idx

                for key in ["idx_to_symbol", "edges", "root", "target", "neg_target"]:
                    if key in item and key not in meta:
                        meta[key] = item[key]

                samples.append(DataSample(
                    question=item.get("question", item.get("text", "")),
                    answer=str(item.get("answer", item.get("label", ""))),
                    cot_steps=cot_steps,
                    meta=meta
                ))
        return samples

    def get_split(self, split: str) -> List[DataSample]:
        if split not in self.splits:
            raise KeyError(f"Unknown split '{split}'")
        return self.splits[split]

    def evaluate_answer(self, prediction: str, target: str, meta: Dict[str, Any]) -> bool:
        """Final-answer exact match.

        Synthetic True/False items are decided by the last true/false token. Real ProsQA
        answers are sentences ("Sally is a sterpus."), decided by normalized equality or
        by the final entity. Substring containment is NOT accepted: it scored an empty
        generation, "sally" and "is a" as correct.
        """
        if normalize_text(target) in ("true", "false"):
            return match_boolean(prediction, target)
        return match_free_form(prediction, target)
