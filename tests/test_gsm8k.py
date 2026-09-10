import pytest
from pathlib import Path
from lsrr.data.gsm8k import GSM8KDataset
from lsrr.registry import DATA_REGISTRY

def test_gsm8k_registry():
    assert "gsm8k" in DATA_REGISTRY
    assert "gsm8k_aug" in DATA_REGISTRY

def test_gsm8k_synthetic_fallback():
    # When data_path points to nonexistent folder
    ds = GSM8KDataset(data_path="nonexistent_folder_xyz", num_mock_samples=10)
    train_samples = ds.get_split("train")
    val_samples = ds.get_split("val")
    test_samples = ds.get_split("test")

    assert len(train_samples) == 10
    assert len(val_samples) == 50
    assert len(test_samples) == 50
    assert train_samples[0].question != ""
    assert train_samples[0].answer != ""
    assert len(train_samples[0].cot_steps) > 0

def test_gsm8k_load_from_disk():
    # Test loading actual exported data/gsm8k-aug files
    data_dir = Path("data/gsm8k-aug")
    if not (data_dir / "valid.json").exists():
        pytest.skip("data/gsm8k-aug not present on disk")

    ds = GSM8KDataset(data_path="data/gsm8k-aug", max_train_samples=50)
    val_samples = ds.get_split("val")
    test_samples = ds.get_split("test")
    train_samples = ds.get_split("train")

    assert len(val_samples) == 500
    assert len(test_samples) == 1319
    assert len(train_samples) == 50

    # Check sample contents
    s0 = val_samples[0]
    assert s0.question != ""
    assert s0.answer != ""
    assert isinstance(s0.cot_steps, list)
    assert s0.meta.get("hops") == len(s0.cot_steps)

def test_gsm8k_evaluate_answer():
    ds = GSM8KDataset(data_path="nonexistent_mock", num_mock_samples=5)

    # Clean integers
    assert ds.evaluate_answer("360", "360", {}) is True
    assert ds.evaluate_answer("361", "360", {}) is False

    # Numbers with commas and currency
    assert ds.evaluate_answer("1,400", "1400", {}) is True
    assert ds.evaluate_answer("$1,400", "1400", {}) is True
    assert ds.evaluate_answer("The answer is $1,400.", "1400", {}) is True

    # Decimals
    assert ds.evaluate_answer("5.5", "5.5", {}) is True
    assert ds.evaluate_answer("5.50", "5.5", {}) is True

    # CoT formatted answer with ####
    assert ds.evaluate_answer("Calculations show... #### 72", "72", {}) is True
    assert ds.evaluate_answer("Calculations show... #### 75", "72", {}) is False
