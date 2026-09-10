"""Offline checks for scripts/download_data.py.

Nothing here touches the network: the conversion is exercised on a fixture, and the
manifest is checked against the paths the data configs actually load from.
"""
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.download_data import MANIFEST, icot_to_json, sha256_of  # noqa: E402


ICOT_LINE = (
    "Janet has 3 apples.||<<3*2=6>> <<6+1=7>>##7\n"
    "It’s Meghan’s turn.||<<2+2=4>>##4\n"
)


def test_icot_conversion_produces_the_coconut_schema(tmp_path):
    src = tmp_path / "in.txt"
    src.write_text(ICOT_LINE, encoding="utf-8")
    dest = tmp_path / "out.json"
    icot_to_json(src, dest)

    records = json.loads(dest.read_text(encoding="utf-8"))
    assert len(records) == 2
    assert records[0] == {
        "question": "Janet has 3 apples.",
        "answer": "7",
        "steps": ["<<3*2=6>>", "<<6+1=7>>"],
    }
    # Key order matters: it is part of reproducing the published files byte-for-byte.
    assert list(records[0].keys()) == ["question", "answer", "steps"]


def test_icot_conversion_keeps_utf8_unescaped(tmp_path):
    """ensure_ascii=False; escaping to \\u2019 changes the bytes and breaks the checksum."""
    src = tmp_path / "in.txt"
    src.write_text(ICOT_LINE, encoding="utf-8")
    dest = tmp_path / "out.json"
    icot_to_json(src, dest)

    raw = dest.read_bytes()
    assert "’".encode("utf-8") in raw
    assert b"\\u2019" not in raw


def test_manifest_covers_the_paths_the_configs_load(tmp_path):
    produced = {p for files in MANIFEST.values() for p in files}
    assert produced == {
        "prosqa/prosqa_train.json",
        "prosqa/prosqa_valid.json",
        "prosqa/prosqa_test.json",
        "gsm8k-aug/train.json",
        "gsm8k-aug/valid.json",
        "gsm8k-aug/test.json",
    }
    # Every entry must carry a checksum; an unverified download is the failure mode
    # this script exists to prevent.
    for files in MANIFEST.values():
        for rel, spec in files.items():
            assert len(spec["sha256"]) == 64, rel
            assert spec["bytes"] > 0, rel
            assert spec["url"].startswith("https://"), rel


def test_manifest_urls_are_pinned_to_commits():
    """Pinning keeps the recorded checksums meaningful over time."""
    for files in MANIFEST.values():
        for rel, spec in files.items():
            assert "/main/" not in spec["url"], f"{rel} tracks a moving branch"
            assert "/master/" not in spec["url"], f"{rel} tracks a moving branch"


def test_gsm8k_train_uses_the_lfs_media_endpoint():
    """train.txt is a Git LFS object; the raw endpoint returns a 133-byte pointer."""
    url = MANIFEST["gsm8k-aug"]["gsm8k-aug/train.json"]["url"]
    assert url.startswith("https://media.githubusercontent.com/media/")


@pytest.mark.parametrize("rel,spec", [
    (rel, spec) for files in MANIFEST.values() for rel, spec in files.items()
])
def test_recorded_checksums_match_local_data_if_present(rel, spec):
    """When the datasets are on disk, the manifest must describe them exactly."""
    path = REPO_ROOT / "data" / rel
    if not path.exists():
        pytest.skip(f"{rel} not downloaded")
    assert path.stat().st_size == spec["bytes"]
    assert sha256_of(path) == spec["sha256"]


def test_data_directory_is_gitignored():
    ignore = (REPO_ROOT / ".gitignore").read_text()
    assert "/data/" in ignore, "datasets must stay out of git"
