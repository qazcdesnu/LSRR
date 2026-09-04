import os
import json
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import torch
from torch.utils.data import Dataset
from safetensors.torch import save_file, load_file, safe_open

def compute_cache_key(
    backbone_id: str,
    dataset_id: str,
    split: str,
    tokenizer_hash: str,
    position_rule: str = "last_token",
    precision: str = "float32",
    include_embedding: bool = False
) -> str:
    """Deterministic hash key uniquely identifying the extracted hidden states."""
    raw = f"{backbone_id}:{dataset_id}:{split}:{tokenizer_hash}:{position_rule}:{precision}:{include_embedding}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

class ShardedHCacheWriter:
    """Writes extracted H tensors into sharded safetensors files with manifest.json."""
    def __init__(
        self,
        output_dir: Path,
        shard_size: int = 5000,
        precision: str = "float32"
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.shard_size = shard_size
        self.precision = precision

        self.current_shard_idx = 0
        self.current_h_list: List[torch.Tensor] = []
        self.current_target_ids: List[torch.Tensor] = []
        self.manifest_entries: List[Dict[str, Any]] = []
        self.total_samples = 0
        self.meta_list: List[Dict[str, Any]] = []

    def add_sample(
        self,
        h_tensor: torch.Tensor,
        target_ids: Optional[torch.Tensor] = None,
        meta: Optional[Dict[str, Any]] = None
    ):
        """h_tensor: [L, d]"""
        dtype = torch.bfloat16 if self.precision == "bfloat16" else (torch.float16 if self.precision == "float16" else torch.float32)
        self.current_h_list.append(h_tensor.to(dtype).cpu())
        if target_ids is not None:
            self.current_target_ids.append(target_ids.cpu())
        else:
            self.current_target_ids.append(torch.empty((0,), dtype=torch.long))
        
        self.meta_list.append(meta or {})
        self.total_samples += 1

        if len(self.current_h_list) >= self.shard_size:
            self._flush_shard()

    def _flush_shard(self):
        if not self.current_h_list:
            return
        shard_filename = f"shard_{self.current_shard_idx:05d}.safetensors"
        shard_path = self.output_dir / shard_filename

        h_stacked = torch.stack(self.current_h_list, dim=0)  # [N_shard, L, d]
        
        # Target ids padding if variable length
        max_len = max((t.size(0) for t in self.current_target_ids), default=0)
        target_padded = torch.zeros((len(self.current_target_ids), max_len), dtype=torch.long)
        for i, t in enumerate(self.current_target_ids):
            if t.size(0) > 0:
                target_padded[i, :t.size(0)] = t

        tensors = {
            "H": h_stacked,
            "target_ids": target_padded
        }
        save_file(tensors, shard_path)

        self.manifest_entries.append({
            "shard_idx": self.current_shard_idx,
            "filename": shard_filename,
            "num_samples": len(self.current_h_list),
            "h_shape": list(h_stacked.shape),
            "precision": self.precision
        })

        self.current_shard_idx += 1
        self.current_h_list = []
        self.current_target_ids = []

    def finalize(self, extra_meta: Optional[Dict[str, Any]] = None) -> Path:
        self._flush_shard()
        manifest_path = self.output_dir / "manifest.json"
        meta_path = self.output_dir / "samples_meta.json"

        manifest_data = {
            "total_samples": self.total_samples,
            "num_shards": self.current_shard_idx,
            "shards": self.manifest_entries,
            "precision": self.precision,
            "extra_meta": extra_meta or {}
        }
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=2)

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(self.meta_list, f)

        return manifest_path

class ShardedHCacheDataset(Dataset):
    """Memory-mapped dataset reading H tensors directly from sharded safetensors."""
    def __init__(self, cache_dir: Path):
        self.cache_dir = Path(cache_dir)
        manifest_path = self.cache_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found in {cache_dir}")

        with open(manifest_path, "r", encoding="utf-8") as f:
            self.manifest = json.load(f)

        meta_path = self.cache_dir / "samples_meta.json"
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                self.samples_meta = json.load(f)
        else:
            self.samples_meta = []

        self.total_samples = self.manifest["total_samples"]
        self.shards_info = self.manifest["shards"]

        # Build index mapping: global_idx -> (shard_filename, local_idx)
        self.index_map = []
        for s_info in self.shards_info:
            s_fname = s_info["filename"]
            for l_idx in range(s_info["num_samples"]):
                self.index_map.append((s_fname, l_idx))

        # Lazy open handles
        self._open_handles: Dict[str, Any] = {}

    def _get_handle(self, filename: str):
        if filename not in self._open_handles:
            fpath = str(self.cache_dir / filename)
            self._open_handles[filename] = safe_open(fpath, framework="pt", device="cpu")
        return self._open_handles[filename]

    def __len__(self) -> int:
        return self.total_samples

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        shard_filename, local_idx = self.index_map[idx]
        handle = self._get_handle(shard_filename)

        # safe_open slice indexing: handle.get_slice("H")[local_idx]
        h_slice = handle.get_slice("H")
        h_tensor = h_slice[local_idx:local_idx+1].squeeze(0).float()

        target_slice = handle.get_slice("target_ids")
        target_ids = target_slice[local_idx:local_idx+1].squeeze(0)

        meta = self.samples_meta[idx] if idx < len(self.samples_meta) else {}
        return {
            "H": h_tensor,
            "target_ids": target_ids,
            "meta": meta
        }
