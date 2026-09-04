import tempfile
from pathlib import Path
import torch
import pytest
from lsrr.data.cache import compute_cache_key, ShardedHCacheWriter, ShardedHCacheDataset

def test_cache_key_collision_prevention():
    """Test 3a: Distinct inputs to compute_cache_key must never collide."""
    key1 = compute_cache_key("gpt2", "gsm8k", "train", "tok1", "last_token", "float32")
    key2 = compute_cache_key("gpt2", "gsm8k", "val", "tok1", "last_token", "float32")
    key3 = compute_cache_key("llama", "gsm8k", "train", "tok1", "last_token", "float32")
    key4 = compute_cache_key("gpt2", "gsm8k", "train", "tok2", "last_token", "float32")
    key5 = compute_cache_key("gpt2", "gsm8k", "train", "tok1", "mean_pooling", "float32")
    key6 = compute_cache_key("gpt2", "gsm8k", "train", "tok1", "last_token", "bfloat16")

    keys = [key1, key2, key3, key4, key5, key6]
    assert len(set(keys)) == len(keys), f"Cache key collision detected: {keys}"

def test_cache_reconstruction_fidelity():
    """Test 3b: Cached H vs original on-the-fly H error < 1e-5."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        writer = ShardedHCacheWriter(tmp_path, shard_size=5, precision="float32")

        num_samples = 12
        L, d = 12, 64
        torch.manual_seed(42)
        original_Hs = [torch.randn(L, d) for _ in range(num_samples)]
        original_targets = [torch.randint(0, 1000, (4,)) for _ in range(num_samples)]

        for i in range(num_samples):
            writer.add_sample(original_Hs[i], target_ids=original_targets[i], meta={"idx": i})
        writer.finalize()

        # Load back via ShardedHCacheDataset
        dataset = ShardedHCacheDataset(tmp_path)
        assert len(dataset) == num_samples

        for i in range(num_samples):
            item = dataset[i]
            h_loaded = item["H"]
            t_loaded = item["target_ids"]
            meta_loaded = item["meta"]

            diff = (h_loaded - original_Hs[i]).abs().max().item()
            assert diff < 1e-5, f"Cache H deviation exceeded tolerance at index {i}: {diff}"
            assert torch.equal(t_loaded, original_targets[i]), f"Target IDs mismatch at index {i}"
            assert meta_loaded["idx"] == i

        print("\n[Test 3 Passed] Cache fidelity verified with max diff < 1e-5.")
