"""Regression tests for answer-target construction.

Guards the failure found before the GSM8K run: cached targets were returned with the
shard's zero padding still attached, GPT-2 token 0 is '!', and the answer loss did not
mask it -- so 84% of the supervised positions trained the model to emit exclamation
marks. Targets also carried no EOS, so the decoder never learned to stop.
"""
import torch
import pytest
from torch.utils.data import DataLoader

from lsrr.data.cache import ShardedHCacheWriter, ShardedHCacheDataset
from lsrr.data.collate import collate_h_cache, IGNORE_INDEX
from lsrr.losses.composite import AnswerNLLLoss

EOS = 50256


def _write_cache(tmp_path, targets, name="cache", with_lens=True):
    """Write a shard; `with_lens=False` reproduces the legacy layout."""
    w = ShardedHCacheWriter(tmp_path / name, shard_size=16)
    for i, t in enumerate(targets):
        w.add_sample(torch.randn(4, 8), target_ids=torch.tensor(t), meta={"sample_id": i})
    w.finalize()

    if not with_lens:
        # Strip target_lens to emulate a cache written before it was recorded.
        from safetensors.torch import load_file, save_file
        p = tmp_path / name / "shard_00000.safetensors"
        d = load_file(p)
        d.pop("target_lens")
        save_file(d, p)
    return ShardedHCacheDataset(tmp_path / name)


def test_dataset_trims_shard_padding(tmp_path):
    ds = _write_cache(tmp_path, [[15277], [15426, 999], [3064]])
    assert ds[0]["target_ids"].tolist() == [15277]
    assert ds[1]["target_ids"].tolist() == [15426, 999]
    assert ds[2]["target_ids"].tolist() == [3064]


def test_legacy_cache_without_target_lens_is_still_trimmed(tmp_path):
    """The 14 GB GSM8K cache predates target_lens; trailing zeros identify the padding."""
    ds = _write_cache(tmp_path, [[15277], [15426, 999], [3064]], with_lens=False)
    from safetensors.torch import safe_open
    with safe_open(str(tmp_path / "cache" / "shard_00000.safetensors"), framework="pt") as h:
        assert "target_lens" not in set(h.keys())
    assert ds[0]["target_ids"].tolist() == [15277]
    assert ds[1]["target_ids"].tolist() == [15426, 999]


def test_writer_records_target_lens(tmp_path):
    _write_cache(tmp_path, [[1], [2, 3, 4], [5, 6]])
    from safetensors.torch import safe_open
    with safe_open(str(tmp_path / "cache" / "shard_00000.safetensors"), framework="pt") as h:
        assert "target_lens" in set(h.keys())
        assert h.get_slice("target_lens")[0:3].reshape(-1).tolist() == [1, 3, 2]


def test_collate_masks_padding_and_appends_eos(tmp_path):
    ds = _write_cache(tmp_path, [[15277], [15426, 999], [3064]])
    b = collate_h_cache([ds[i] for i in range(3)], eos_token_id=EOS)

    assert b["target_lens"].tolist() == [2, 3, 2]
    # Decoder input pads with a real token id; labels mark padding as ignored.
    assert b["target_ids"][0].tolist() == [15277, EOS, EOS]
    assert b["labels"][0].tolist() == [15277, EOS, IGNORE_INDEX]
    assert b["labels"][1].tolist() == [15426, 999, EOS]
    assert IGNORE_INDEX not in b["target_ids"].tolist()[0]

    # Every target terminates, so generation can stop.
    for i, n in enumerate(b["target_lens"].tolist()):
        assert int(b["target_ids"][i][n - 1]) == EOS


def test_collate_does_not_duplicate_an_existing_eos(tmp_path):
    ds = _write_cache(tmp_path, [[15277, EOS]])
    b = collate_h_cache([ds[0]], eos_token_id=EOS)
    assert b["target_ids"][0].tolist() == [15277, EOS]
    assert b["target_lens"].tolist() == [2]


def test_answer_loss_ignores_padded_positions():
    """The padded tail must not move the loss at all."""
    torch.manual_seed(0)
    V, B, T = 32, 2, 5
    logits = torch.randn(B, T, V)
    labels = torch.tensor([[7, EOS % V, IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX],
                           [9, 11, EOS % V, IGNORE_INDEX, IGNORE_INDEX]])
    loss_fn = AnswerNLLLoss()

    base = loss_fn({"logits": logits}, {"labels": labels})["loss"]

    # Replacing the ignored positions entirely leaves the loss unchanged.
    # (A uniform shift across the vocabulary would not test anything -- softmax is
    # invariant to it -- so these rows are resampled outright.)
    perturbed = logits.clone()
    perturbed[0, 2:] = torch.randn(3, V) * 10.0
    perturbed[1, 3:] = torch.randn(2, V) * 10.0
    after = loss_fn({"logits": perturbed}, {"labels": labels})["loss"]
    assert torch.allclose(base, after, atol=1e-6)

    # Changing one supervised position's correct-class logit does move it.
    moved = logits.clone()
    moved[0, 0, 7] += 50.0
    assert not torch.allclose(base, loss_fn({"logits": moved}, {"labels": labels})["loss"], atol=1e-6)


def test_answer_loss_falls_back_to_target_ids_without_labels():
    logits = torch.randn(1, 3, 16)
    tids = torch.tensor([[1, 2, 3]])
    loss_fn = AnswerNLLLoss()
    assert torch.isfinite(loss_fn({"logits": logits}, {"target_ids": tids})["loss"])


def test_padding_supervision_would_have_dominated(tmp_path):
    """Quantifies the original bug: unmasked padding swamped the real answer tokens."""
    ds = _write_cache(tmp_path, [[15277]] * 8)
    b = collate_h_cache([ds[i] for i in range(8)], eos_token_id=EOS)
    supervised = int((b["labels"] != IGNORE_INDEX).sum())
    assert supervised == b["labels"].numel(), "single-token answers leave no padding here"

    # Mixed lengths: only the real tokens are supervised.
    ds2 = _write_cache(tmp_path, [[15277], [1, 2, 3, 4, 5, 6, 7]], name="mixed")
    b2 = collate_h_cache([ds2[0], ds2[1]], eos_token_id=EOS)
    assert int((b2["labels"] != IGNORE_INDEX).sum()) == 2 + 8
    assert int((b2["labels"] == IGNORE_INDEX).sum()) == 6


def test_extraction_appends_eos_to_targets(tmp_path):
    from lsrr.utils.oom import process_batch_with_oom_recovery
    from lsrr.interfaces import DataSample

    class _Extractor:
        hidden_dim, num_layers = 8, 4

        class _Tok:
            eos_token_id = EOS

            def __call__(self, texts, padding=True, truncation=True, return_tensors="pt"):
                n = len(texts)
                return {"input_ids": torch.ones(n, 3, dtype=torch.long),
                        "attention_mask": torch.ones(n, 3, dtype=torch.long)}

        tokenizer = _Tok()

        def extract_hidden_states(self, input_ids, attention_mask=None, position_rule="last_token"):
            return torch.randn(input_ids.size(0), 4, 8)

    w = ShardedHCacheWriter(tmp_path / "e", shard_size=8)
    process_batch_with_oom_recovery(
        _Extractor(), w,
        [DataSample(question="q", answer="42", cot_steps=[], meta={})],
    )
    w.finalize()
    ds = ShardedHCacheDataset(tmp_path / "e")
    assert int(ds[0]["target_ids"][-1]) == EOS


def test_train_and_eval_share_one_collate():
    """Two divergent copies is how the eval path lost target_ids in the first place."""
    import scripts.train as tr
    import scripts.eval as ev
    from lsrr.data.collate import collate_h_cache as canonical
    assert tr.collate_h_cache is canonical
    assert ev.collate_h_cache is canonical
