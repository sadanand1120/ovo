import json
from pathlib import Path

import numpy as np


CLIP_FEATURE_FILE = "clip_feats.npy"
CLIP_FEATURE_SHARD_DIR = "clip_feat_shards"
CLIP_FEATURE_MANIFEST = "manifest.json"


def write_shard_manifest(shard_dir: Path, shard_sizes: list[int], feature_dim: int) -> None:
    start = 0
    shards = []
    for shard_id, size in enumerate(shard_sizes):
        shards.append(
            {
                "path": f"{shard_id:06d}.npy",
                "start": start,
                "size": int(size),
            }
        )
        start += int(size)
    with open(shard_dir / CLIP_FEATURE_MANIFEST, "w") as f:
        json.dump(
            {
                "dtype": "float16",
                "feature_dim": int(feature_dim),
                "n_features": int(start),
                "shards": shards,
            },
            f,
            indent=2,
        )


class ClipFeatureStore:
    def __init__(self, map_dir: Path) -> None:
        clip_path = map_dir / CLIP_FEATURE_FILE
        if clip_path.exists():
            self._mode = "single"
            self._features = np.load(clip_path, mmap_mode="r")
            self.shape = self._features.shape
            self.dtype = self._features.dtype
            return

        shard_dir = map_dir / CLIP_FEATURE_SHARD_DIR
        manifest_path = shard_dir / CLIP_FEATURE_MANIFEST
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing CLIP features at {clip_path} or {manifest_path}")

        manifest = json.loads(manifest_path.read_text())
        self._mode = "sharded"
        self._starts = np.asarray([shard["start"] for shard in manifest["shards"]], dtype=np.int64)
        self._sizes = np.asarray([shard["size"] for shard in manifest["shards"]], dtype=np.int64)
        self._shards = [np.load(shard_dir / shard["path"], mmap_mode="r") for shard in manifest["shards"]]
        self.shape = (int(manifest["n_features"]), int(manifest["feature_dim"]))
        self.dtype = np.dtype(manifest["dtype"])

    def __getitem__(self, idx):
        if self._mode == "single":
            return self._features[idx]
        if isinstance(idx, slice):
            start, stop, step = idx.indices(self.shape[0])
            if step != 1:
                return self[np.arange(start, stop, step, dtype=np.int64)]
            return self._read_slice(start, stop)
        if isinstance(idx, (int, np.integer)):
            row = self._read_rows(np.asarray([int(idx)], dtype=np.int64))
            return row[0]
        idx_arr = np.asarray(idx)
        flat = idx_arr.astype(np.int64, copy=False).reshape(-1)
        rows = self._read_rows(flat)
        return rows.reshape(idx_arr.shape + (self.shape[1],))

    def _normalize_rows(self, rows: np.ndarray) -> np.ndarray:
        rows = rows.copy()
        rows[rows < 0] += self.shape[0]
        if ((rows < 0) | (rows >= self.shape[0])).any():
            raise IndexError("clip feature index out of bounds")
        return rows

    def _read_slice(self, start: int, stop: int) -> np.ndarray:
        if stop <= start:
            return np.empty((0, self.shape[1]), dtype=self.dtype)
        out = np.empty((stop - start, self.shape[1]), dtype=self.dtype)
        cursor = 0
        shard_idx = int(np.searchsorted(self._starts, start, side="right") - 1)
        pos = start
        while pos < stop:
            shard_start = int(self._starts[shard_idx])
            shard_stop = shard_start + int(self._sizes[shard_idx])
            take_stop = min(stop, shard_stop)
            out[cursor : cursor + (take_stop - pos)] = self._shards[shard_idx][pos - shard_start : take_stop - shard_start]
            cursor += take_stop - pos
            pos = take_stop
            shard_idx += 1
        return out

    def _read_rows(self, rows: np.ndarray) -> np.ndarray:
        rows = self._normalize_rows(rows)
        out = np.empty((rows.shape[0], self.shape[1]), dtype=self.dtype)
        shard_idx = np.searchsorted(self._starts, rows, side="right") - 1
        for sid in np.unique(shard_idx):
            mask = shard_idx == sid
            local_rows = rows[mask] - self._starts[int(sid)]
            out[mask] = self._shards[int(sid)][local_rows]
        return out
