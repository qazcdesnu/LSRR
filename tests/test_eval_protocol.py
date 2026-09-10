"""Regression tests for the evaluation protocol.

Guards the four failures found in verification:
  1. the gold answer never reaching the scorer (eval scored against "")
  2. scoring rules looser than the latent-reasoning baselines
  3. evaluating the wrong split
  4. latency/FLOPs omitting the frozen-backbone forward pass
"""
import json
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from lsrr.data.answer_scoring import match_final_number, match_free_form, match_boolean
from lsrr.data.gsm8k import GSM8KDataset
from lsrr.data.prosqa import ProsQADataset
from lsrr.data.multiplication import MultiplicationDataset
from lsrr.data.cache import ShardedHCacheWriter, ShardedHCacheDataset
from lsrr.decoders.light_decoder import TrainedLightDecoder
from lsrr.model import LSRRModel
from lsrr.training.evaluator import Evaluator, MissingTargetError
from lsrr.utils.flops import (
    estimate_backbone_flops_per_forward,
    estimate_engine_flops_per_step,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------- (2) scoring rules

@pytest.mark.parametrize("pred,gold,expected", [
    ("The answer is 99", "99", True),
    ("#### 42", "42", True),
    ("42.0", "42", True),
    ("1,000", "1000", True),
    # A gold answer buried mid-generation must NOT count.
    ("5 12 7 42", "12", False),
    (" 42 999 999", "42", False),
    ("", "42", False),
    ("no digits here", "42", False),
])
def test_numeric_scoring_is_final_answer_only(pred, gold, expected):
    assert match_final_number(pred, gold) is expected


@pytest.mark.parametrize("pred,expected", [
    ("Sally is a sterpus.", True),
    ("sterpus", True),
    ("The answer is Sally is a sterpus", True),
    # Substring containment must not score: these were all "correct" before.
    ("", False),
    ("sally", False),
    ("is a", False),
    ("Sally is a", False),
    ("Sally is a bompus.", False),
])
def test_free_form_scoring_rejects_substrings(pred, expected):
    assert match_free_form(pred, "Sally is a sterpus.") is expected


def test_boolean_scoring_uses_last_token():
    assert match_boolean("maybe false, actually true", "true") is True
    assert match_boolean("true then false", "true") is False
    assert match_boolean("", "true") is False


def test_dataset_modules_use_the_shared_protocol():
    gs, pq, ml = GSM8KDataset(num_mock_samples=2), ProsQADataset(num_mock_samples=2), MultiplicationDataset(num_train=2, num_val=1, num_test=1)
    assert gs.evaluate_answer("5 12 7 42", "12", {}) is False
    assert gs.evaluate_answer("the answer is 12", "12", {}) is True
    assert ml.evaluate_answer("5 12 7 42", "12", {}) is False
    assert pq.evaluate_answer("", "Sally is a sterpus.", {}) is False
    # Empty predictions are never correct on any dataset.
    for ds, gold in ((gs, "12"), (ml, "12"), (pq, "Sally is a sterpus.")):
        assert ds.evaluate_answer("", gold, {}) is False


@pytest.mark.parametrize("real_data", [True])
def test_real_prosqa_answers_are_scored_strictly(real_data):
    data_dir = REPO_ROOT / "data" / "prosqa"
    if not data_dir.exists():
        pytest.skip("real ProsQA data not present")
    pq = ProsQADataset(data_path=str(data_dir))
    gold = pq.get_split("test")[0].answer
    assert pq.evaluate_answer(gold, gold, {}) is True
    assert pq.evaluate_answer("", gold, {}) is False
    assert pq.evaluate_answer(gold.split(" ")[0], gold, {}) is False


# ---------------------------------------------------------------- (2) EOS truncation

def test_generate_truncates_each_sequence_at_its_own_eos():
    """Post-EOS tokens must not survive into the string the scorer sees."""
    EOS, V = 7, 16
    dec = TrainedLightDecoder(d_model=8, vocab_size=V, n_layers=1, n_heads=2, bos_token_id=EOS)

    class Scripted(nn.Module):
        """seq0: token 3, EOS, then junk forever. seq1: never emits EOS."""
        def __init__(self):
            super().__init__()
            self.step = 0

        def forward(self, x):
            logits = torch.zeros(*x.shape[:-1], V)
            if x.dim() == 2:
                plan = [3, EOS, 9, 9, 9]
                logits[0, plan[min(self.step, len(plan) - 1)]] = 100.0
                logits[1, 5] = 100.0
                self.step += 1
            return logits

    dec.lm_head = Scripted()
    with torch.no_grad():
        toks = dec.generate(torch.randn(2, 8), max_new_tokens=5, eos_token_id=EOS)

    seq0 = toks[0].tolist()
    assert seq0[0] == 3
    # Everything from the first EOS onward is EOS, so skip_special_tokens drops the tail.
    assert set(seq0[1:]) == {EOS}, f"post-EOS junk survived: {seq0}"
    # The unfinished sequence keeps generating.
    assert toks[1].tolist() == [5] * 5


def test_generate_ignores_eos_outside_vocabulary():
    dec = TrainedLightDecoder(d_model=8, vocab_size=16, n_layers=1, n_heads=2, bos_token_id=0)
    with torch.no_grad():
        toks = dec.generate(torch.randn(2, 8), max_new_tokens=4, eos_token_id=50256)
    assert toks.shape == (2, 4)


# ---------------------------------------------------------------- (1) target plumbing

class _StubTokenizer:
    eos_token_id = 0

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(str(int(i)) for i in ids if int(i) != 0)

    def batch_decode(self, seqs, skip_special_tokens=True):
        return [self.decode(s, skip_special_tokens) for s in seqs]


class _AlwaysCorrect:
    """Scores by whether a gold answer was supplied at all."""
    def __init__(self):
        self.seen_targets = []

    def evaluate_answer(self, prediction, target, meta):
        self.seen_targets.append(target)
        return bool(target)


def _tiny_model(d_in=16, num_layers=4):
    return LSRRModel(
        adapter_cfg={"type": "per_layer_affine+rmsnorm"},
        engine_cfg={"type": "mamba_up", "n_blocks": 1, "d_state": 4},
        fusion_cfg={"type": "attention_pooling", "fusion_type": "residual"},
        decoder_cfg={"type": "trained_light_decoder", "vocab_size": 32, "n_layers": 1, "n_heads": 2},
        iteration_cfg={"train_m": {"type": "fixed", "k": 2}, "tbptt_k": 1},
        termination_cfg={"type": "delta_state", "eps": 1e-3, "m_max": 3},
        d_in=d_in, num_layers=num_layers,
    )


def _collate_with_targets(batch):
    H = torch.stack([b["H"] for b in batch])
    lens = torch.tensor([b["target_ids"].size(0) for b in batch])
    m = int(lens.max()) if len(lens) else 0
    tids = torch.zeros((len(batch), m), dtype=torch.long)
    for i, b in enumerate(batch):
        tids[i, : b["target_ids"].size(0)] = b["target_ids"]
    return {"H": H, "target_ids": tids, "target_lens": lens, "meta": [b["meta"] for b in batch]}


def _collate_without_targets(batch):
    return {
        "H": torch.stack([b["H"] for b in batch]),
        "meta": [b["meta"] for b in batch],
    }


def _write_cache(tmp_path, with_answer_in_meta: bool):
    writer = ShardedHCacheWriter(tmp_path / "cache", shard_size=8)
    for i in range(4):
        meta = {"hops": (i % 2) + 1, "sample_id": i}
        if with_answer_in_meta:
            meta["answer"] = str(10 + i)
        writer.add_sample(torch.randn(4, 16), target_ids=torch.tensor([3, 4]), meta=meta)
    writer.finalize()
    return ShardedHCacheDataset(tmp_path / "cache")


def _evaluator(model, dataset_mod):
    return Evaluator(
        model=model, dataset_mod=dataset_mod, tokenizer=_StubTokenizer(),
        device=torch.device("cpu"), engine_name="mamba_up", d_model=model.d_model,
        num_engine_layers=1, num_model_layers=4, max_new_tokens=3,
    )


def test_gold_answer_reaches_the_scorer_via_meta(tmp_path):
    ds = _write_cache(tmp_path, with_answer_in_meta=True)
    loader = DataLoader(ds, batch_size=2, collate_fn=_collate_without_targets)
    scorer = _AlwaysCorrect()
    res = _evaluator(_tiny_model(), scorer).evaluate_rule(loader)

    assert res["total_samples"] == 4
    assert scorer.seen_targets == ["10", "11", "12", "13"]
    assert all(t for t in scorer.seen_targets), "scorer received an empty target"


def test_gold_answer_reaches_the_scorer_via_target_ids(tmp_path):
    """Caches written before meta carried the answer still score correctly."""
    ds = _write_cache(tmp_path, with_answer_in_meta=False)
    loader = DataLoader(ds, batch_size=2, collate_fn=_collate_with_targets)
    scorer = _AlwaysCorrect()
    res = _evaluator(_tiny_model(), scorer).evaluate_rule(loader)

    assert res["accuracy"] == 1.0
    assert all(t == "3 4" for t in scorer.seen_targets)


def test_missing_gold_answer_is_fatal_not_silent(tmp_path):
    """The original bug: no answer anywhere, scored against "" without complaint."""
    ds = _write_cache(tmp_path, with_answer_in_meta=False)
    loader = DataLoader(ds, batch_size=2, collate_fn=_collate_without_targets)
    with pytest.raises(MissingTargetError):
        _evaluator(_tiny_model(), _AlwaysCorrect()).evaluate_rule(loader)


def test_extraction_records_answer_in_cache_meta(tmp_path):
    """process_batch_with_oom_recovery must persist the gold answer."""
    from lsrr.utils.oom import process_batch_with_oom_recovery
    from lsrr.interfaces import DataSample

    class _Extractor:
        hidden_dim, num_layers = 16, 4

        class _Tok:
            def __call__(self, texts, padding=True, truncation=True, return_tensors="pt"):
                n = len(texts)
                return {"input_ids": torch.ones(n, 3, dtype=torch.long),
                        "attention_mask": torch.ones(n, 3, dtype=torch.long)}

        tokenizer = _Tok()

        def extract_hidden_states(self, input_ids, attention_mask=None, position_rule="last_token"):
            return torch.randn(input_ids.size(0), 4, 16)

    writer = ShardedHCacheWriter(tmp_path / "c", shard_size=8)
    samples = [DataSample(question="q1", answer="42", cot_steps=[], meta={"hops": 1}),
               DataSample(question="q2", answer="7", cot_steps=[], meta={"hops": 2})]
    process_batch_with_oom_recovery(_Extractor(), writer, samples)
    writer.finalize()

    metas = json.loads((tmp_path / "c" / "samples_meta.json").read_text())
    assert [m["answer"] for m in metas] == ["42", "7"]
    assert [m["hops"] for m in metas] == [1, 2]


def test_per_hop_accuracy_is_reported(tmp_path):
    ds = _write_cache(tmp_path, with_answer_in_meta=True)
    loader = DataLoader(ds, batch_size=2, collate_fn=_collate_without_targets)
    res = _evaluator(_tiny_model(), _AlwaysCorrect()).evaluate_rule(loader)
    assert set(res["accuracy_by_hops"]) == {"1", "2"}
    assert sum(v["n"] for v in res["accuracy_by_hops"].values()) == 4


# ---------------------------------------------------------------- (4) cost accounting

def test_backbone_flops_dominate_a_refinement_cycle():
    """Sanity: omitting the backbone hides the bulk of the cost."""
    backbone = estimate_backbone_flops_per_forward(num_layers=12, d_model=768, seq_len=128)
    cycle = estimate_engine_flops_per_step("hydra_qs", 768, 2, 12)
    assert backbone > 0 and cycle > 0
    assert backbone > 10 * cycle, "backbone pass must not be a rounding error in the total"


def test_cost_report_flags_when_backbone_is_excluded(tmp_path):
    ds = _write_cache(tmp_path, with_answer_in_meta=True)
    loader = DataLoader(ds, batch_size=2, collate_fn=_collate_without_targets)
    res = _evaluator(_tiny_model(), _AlwaysCorrect()).evaluate_rule(loader)

    # No extractor supplied -> the report must say so rather than quietly under-reporting.
    assert res["flops_includes_backbone"] is False
    assert res["latency_includes_backbone"] is False
    assert res["flops_backbone_per_sample"] is None
    # The remaining terms are still broken out and sum to the reported total.
    parts = res["flops_adapter_per_sample"] + res["flops_engine_per_sample"] + res["flops_decoder_per_sample"]
    assert res["flops_total_per_sample"] == pytest.approx(parts)
    assert res["latency_ms_per_sample"] == pytest.approx(res["latency_ms_refinement_only"])


def test_cost_report_includes_backbone_when_measured(tmp_path):
    ds = _write_cache(tmp_path, with_answer_in_meta=True)
    loader = DataLoader(ds, batch_size=2, collate_fn=_collate_without_targets)

    class _Extractor:
        hidden_dim, num_layers = 16, 4

        class _Tok:
            def __call__(self, texts, padding=True, truncation=True, return_tensors="pt"):
                n = len(texts)
                return {"input_ids": torch.ones(n, 5, dtype=torch.long),
                        "attention_mask": torch.ones(n, 5, dtype=torch.long)}

        tokenizer = _Tok()

        def extract_hidden_states(self, input_ids, attention_mask=None, position_rule="last_token"):
            return torch.randn(input_ids.size(0), 4, 16)

    class _Sample:
        question = "how many apples?"

    model = _tiny_model()
    ev = Evaluator(
        model=model, dataset_mod=_AlwaysCorrect(), tokenizer=_StubTokenizer(),
        device=torch.device("cpu"), engine_name="mamba_up", d_model=model.d_model,
        num_engine_layers=1, num_model_layers=4, max_new_tokens=3,
        extractor=_Extractor(), eval_samples=[_Sample()] * 4,
    )
    cost = ev.measure_backbone_cost(batch_size=2)
    assert cost["backbone_flops_per_sample"] > 0
    assert cost["backbone_timed_samples"] == 4

    res = ev.evaluate_rule(loader)
    assert res["flops_includes_backbone"] is True
    assert res["latency_includes_backbone"] is True
    assert res["flops_backbone_per_sample"] > 0
    assert res["latency_ms_per_sample"] >= res["latency_ms_refinement_only"]
    assert res["flops_total_per_sample"] > res["flops_engine_per_sample"]


# ---------------------------------------------------------------- (3) split selection

def test_eval_script_defaults_to_the_test_split():
    src = (REPO_ROOT / "scripts" / "eval.py").read_text()
    assert 'get("split", "test")' in src, "eval.py must default to the held-out test split"
    assert 'split="val"' not in src, "eval.py must not hardcode the val split"
    # And it must keep target_ids so the scorer receives a gold answer.
    assert "target_ids" in src


@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_real_datasets_expose_all_splits(split):
    for name, path in (("prosqa", "data/prosqa"), ("gsm8k", "data/gsm8k-aug")):
        d = REPO_ROOT / path
        if not d.exists():
            pytest.skip(f"{name} data not present")
        ds = ProsQADataset(data_path=str(d)) if name == "prosqa" else GSM8KDataset(data_path=str(d))
        assert len(ds.get_split(split)) > 0, f"{name}/{split} is empty"
