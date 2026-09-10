#!/usr/bin/env python3
"""Fetch the ProsQA and GSM8K-Aug datasets into ./data.

The datasets are large (131 MB) and publicly available, so they are not committed to
this repository. This script reproduces them byte-for-byte from their upstream sources,
pinned to specific commits, and verifies every file against a recorded SHA-256.

Sources
-------
ProsQA      facebookresearch/coconut, data/prosqa_*.json (used as published)
GSM8K-Aug   da03/Internalize_CoT_Step_by_Step, data/gsm8k/*.txt, converted to the
            Coconut JSON schema by the same transform as coconut's
            preprocessing/gsm_icot.py. train.txt is stored with Git LFS and is
            therefore fetched from the media endpoint.

Usage
-----
    python scripts/download_data.py                # fetch anything missing
    python scripts/download_data.py --dataset prosqa
    python scripts/download_data.py --force        # re-download and overwrite
    python scripts/download_data.py --check        # verify what is on disk, fetch nothing
"""
import argparse
import hashlib
import json
import shutil
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"

COCONUT_COMMIT = "27273cb8cca4bb763c041a63b036d0c3b7cbbb48"
ICOT_COMMIT = "e06a32ee5e4cd117171daeb4755d2a97ece62761"

COCONUT_RAW = f"https://raw.githubusercontent.com/facebookresearch/coconut/{COCONUT_COMMIT}/data"
ICOT_RAW = f"https://raw.githubusercontent.com/da03/Internalize_CoT_Step_by_Step/{ICOT_COMMIT}/data/gsm8k"
ICOT_MEDIA = f"https://media.githubusercontent.com/media/da03/Internalize_CoT_Step_by_Step/{ICOT_COMMIT}/data/gsm8k"

# dest path (relative to data/) -> {url, sha256, convert}
MANIFEST: Dict[str, Dict[str, Dict]] = {
    "prosqa": {
        "prosqa/prosqa_train.json": {
            "url": f"{COCONUT_RAW}/prosqa_train.json",
            "sha256": "99e40ce7e9107fd02e35bcd78a7a4479bd57f415bab0a6efd37cc11a72e594f6",
            "bytes": 30872695,
        },
        "prosqa/prosqa_valid.json": {
            "url": f"{COCONUT_RAW}/prosqa_valid.json",
            "sha256": "c74e0de24f1e90ec48b1a993e6458a7be4caeef3928f5104e3c2e45d689a5249",
            "bytes": 518208,
        },
        "prosqa/prosqa_test.json": {
            "url": f"{COCONUT_RAW}/prosqa_test.json",
            "sha256": "82ddc0afe6ee30eb31bb16fa208e293e55278a20fa987098d48519cae7e12680",
            "bytes": 856611,
        },
    },
    "gsm8k-aug": {
        "gsm8k-aug/train.json": {
            # Git LFS object: the raw endpoint returns a 133-byte pointer, not the data.
            "url": f"{ICOT_MEDIA}/train.txt",
            "sha256": "132ab7ef22197ab0aded7d5135b8c82c90c663d71082a646f099a6e7f481f2f5",
            "bytes": 104308420,
            "convert": "icot_to_json",
        },
        "gsm8k-aug/valid.json": {
            "url": f"{ICOT_RAW}/valid.txt",
            "sha256": "1cc0d534fc020afc0165faba35f105283e5e7afed51acc0ea041c29ca0c97928",
            "bytes": 187136,
            "convert": "icot_to_json",
        },
        "gsm8k-aug/test.json": {
            "url": f"{ICOT_RAW}/test.txt",
            "sha256": "d942081ab20af482c81c7bfa4f18662ced7c8bd3b1261ae484c9561261419741",
            "bytes": 504563,
            "convert": "icot_to_json",
        },
    },
}


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024.0:
            return f"{n:.1f}{unit}"
        n /= 1024.0
    return f"{n:.1f}TB"


def icot_to_json(src: Path, dest: Path) -> None:
    """Convert Internalize-CoT text to the Coconut JSON schema.

    Each line is `question || step step step ## answer`. The transform matches
    coconut's preprocessing/gsm_icot.py; the serialization (indent=2,
    ensure_ascii=False, key order question/answer/steps) reproduces the published
    files byte-for-byte.
    """
    with src.open(encoding="utf-8") as f:
        lines = f.readlines()

    records = [
        {
            "question": line.split("||")[0],
            "answer": line.split("##")[-1].strip(),
            "steps": line.split("||")[1].split("##")[0].strip().split(" "),
        }
        for line in lines
    ]
    with dest.open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)


CONVERTERS = {"icot_to_json": icot_to_json}


def download(url: str, dest: Path) -> None:
    """Stream `url` to `dest`, reporting progress."""
    req = urllib.request.Request(url, headers={"User-Agent": "lsrr-download-data"})
    with urllib.request.urlopen(req) as resp:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        with dest.open("wb") as out:
            while True:
                block = resp.read(1 << 20)
                if not block:
                    break
                out.write(block)
                done += len(block)
                if total:
                    pct = done / total * 100
                    print(f"\r    {human(done)} / {human(total)} ({pct:5.1f}%)", end="", flush=True)
                else:
                    print(f"\r    {human(done)}", end="", flush=True)
        print()


def fetch_one(rel_path: str, spec: Dict, force: bool) -> bool:
    """Fetch and verify a single file. Returns True on success."""
    dest = DATA_DIR / rel_path
    expected = spec["sha256"]

    if dest.exists() and not force:
        actual = sha256_of(dest)
        if actual == expected:
            print(f"  [ok]   {rel_path}  ({human(dest.stat().st_size)}, checksum verified)")
            return True
        print(f"  [stale] {rel_path} checksum mismatch, re-downloading")

    dest.parent.mkdir(parents=True, exist_ok=True)
    converter = CONVERTERS.get(spec.get("convert"))
    print(f"  [get]  {rel_path}  <- {spec['url']}")

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td) / "payload"
        try:
            download(spec["url"], tmp)
        except (urllib.error.URLError, urllib.error.HTTPError) as e:
            print(f"  [FAIL] {rel_path}: download failed: {e}")
            return False

        if converter is not None:
            converted = Path(td) / "converted"
            converter(tmp, converted)
            produced = converted
        else:
            produced = tmp

        actual = sha256_of(produced)
        if actual != expected:
            print(
                f"  [FAIL] {rel_path}: checksum mismatch\n"
                f"         expected {expected}\n"
                f"         actual   {actual}\n"
                f"         The upstream source may have changed. Do not use this file."
            )
            return False

        shutil.move(str(produced), str(dest))

    print(f"  [done] {rel_path}  ({human(dest.stat().st_size)}, checksum verified)")
    return True


def check_only(datasets: List[str]) -> int:
    missing = 0
    for name in datasets:
        print(f"\n{name}:")
        for rel_path, spec in MANIFEST[name].items():
            dest = DATA_DIR / rel_path
            if not dest.exists():
                print(f"  [missing] {rel_path}")
                missing += 1
            elif sha256_of(dest) != spec["sha256"]:
                print(f"  [BAD]     {rel_path} checksum mismatch")
                missing += 1
            else:
                print(f"  [ok]      {rel_path}  ({human(dest.stat().st_size)})")
    return missing


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=sorted(MANIFEST) + ["all"], default="all",
                        help="which dataset to fetch (default: all)")
    parser.add_argument("--force", action="store_true", help="re-download even if the file is present and valid")
    parser.add_argument("--check", action="store_true", help="verify what is on disk without downloading")
    args = parser.parse_args(argv)

    datasets = sorted(MANIFEST) if args.dataset == "all" else [args.dataset]

    if args.check:
        missing = check_only(datasets)
        print(f"\n{'All files present and verified.' if not missing else f'{missing} file(s) missing or corrupt.'}")
        return 1 if missing else 0

    total_bytes = sum(s["bytes"] for name in datasets for s in MANIFEST[name].values())
    print(f"Downloading {len(datasets)} dataset(s) into {DATA_DIR} (~{human(total_bytes)} on disk)")

    failed = []
    for name in datasets:
        print(f"\n{name}:")
        for rel_path, spec in MANIFEST[name].items():
            if not fetch_one(rel_path, spec, args.force):
                failed.append(rel_path)

    if failed:
        print(f"\n{len(failed)} file(s) failed:")
        for f in failed:
            print(f"  - {f}")
        return 1

    print("\nAll datasets ready.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
