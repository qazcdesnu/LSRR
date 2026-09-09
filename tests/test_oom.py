import torch
from unittest.mock import MagicMock
from lsrr.interfaces import DataSample
from lsrr.utils.oom import process_batch_with_oom_recovery

class DummyOOMExtractor:
    def __init__(self, oom_threshold_batch_size=4):
        self.oom_threshold = oom_threshold_batch_size
        def mock_tokenize(texts, padding=True, truncation=True, return_tensors="pt"):
            B = len(texts)
            return {
                "input_ids": torch.zeros((B, 10), dtype=torch.long),
                "attention_mask": torch.ones((B, 10), dtype=torch.long)
            }
        self.tokenizer = mock_tokenize
        self.device = torch.device("cpu")
        self.model = MagicMock()

    def extract_hidden_states(self, input_ids, attention_mask, position_rule="last_token"):
        batch_size = input_ids.shape[0]
        if batch_size > self.oom_threshold:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory in test")
        return torch.randn(batch_size, 12, 64)

class DummyWriter:
    def __init__(self):
        self.samples = []

    def add_sample(self, h_sample, target_ids, meta):
        self.samples.append((h_sample, target_ids, meta))

def test_oom_recovery_recursive_splitting():
    extractor = DummyOOMExtractor(oom_threshold_batch_size=4)
    writer = DummyWriter()

    samples = [DataSample(question=f"Q{i}", answer=f"A{i}", cot_steps=[], meta={"id": i}) for i in range(10)]

    # Initial batch size 10 will trigger OOM, then split to 5 + 5, then 5 splits to 2 + 3 (all <= 4), successfully writing all 10 samples
    process_batch_with_oom_recovery(extractor, writer, samples)

    assert len(writer.samples) == 10, f"Expected 10 samples to be processed, got {len(writer.samples)}"
