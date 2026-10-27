"""Dataset inspection and lazy HDF5-backed PyTorch dataset."""
from __future__ import annotations

from pathlib import Path
from typing import Callable
import h5py
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from .labels import CLASS_NAMES, NUM_CLASSES


def inspect_hdf5(path: str | Path) -> dict:
    """Inspect keys, shapes, dtypes, labels, and basic class counts without loading all images."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}. Download Galaxy10_DECals.h5 and place it at this path."
        )
    with h5py.File(path, "r") as h5:
        keys = list(h5.keys())
        if "images" not in h5 or "ans" not in h5:
            raise ValueError(f"Expected HDF5 datasets 'images' and 'ans'; found keys: {keys}")
        image_ds, label_ds = h5["images"], h5["ans"]
        labels = np.asarray(label_ds).reshape(-1).astype(int)
        unique, counts = np.unique(labels, return_counts=True)
        return {
            "path": str(path),
            "keys": keys,
            "images_shape": list(image_ds.shape),
            "images_dtype": str(image_ds.dtype),
            "labels_shape": list(label_ds.shape),
            "labels_dtype": str(label_ds.dtype),
            "n_samples": int(image_ds.shape[0]),
            "label_counts": {str(int(k)): int(v) for k, v in zip(unique, counts)},
            "labels_valid_0_to_9": bool(set(unique.tolist()).issubset(set(range(NUM_CLASSES)))),
            "class_names": {str(i): CLASS_NAMES[i] for i in range(NUM_CLASSES)},
        }


class Galaxy10H5Dataset(Dataset):
    """Fast, cache-backed Galaxy10 dataset using memory-mapped NumPy arrays."""

    def __init__(self, hdf5_path: str | Path, indices: list[int] | np.ndarray,
                 transform: Callable | None = None):
        import json
        self.hdf5_path = str(hdf5_path)
        self.indices = np.asarray(indices, dtype=np.int64)
        self.transform = transform
        source_path = Path(self.hdf5_path)
        cache_dir = source_path.parent.parent / "cache"
        images_path = cache_dir / "galaxy10_images.npy"
        labels_path = cache_dir / "galaxy10_labels.npy"
        meta_path = cache_dir / "cache_complete.json"
        if not source_path.exists():
            raise FileNotFoundError(f"Original dataset not found: {source_path}")
        if not all(p.exists() for p in (images_path, labels_path, meta_path)):
            raise FileNotFoundError("Galaxy10 cache is incomplete. Run scripts/build_cache.py first.")
        meta = json.loads(meta_path.read_text())
        if not meta.get("complete"):
            raise RuntimeError("Cache is not marked complete.")
        self._images_path = str(images_path)
        self.images = None
        self.labels_all = np.load(labels_path, mmap_mode="r")
        if len(self.labels_all) != int(meta["num_images"]):
            raise ValueError("Cached labels do not match cache metadata.")
        if self.indices.size and (self.indices.min() < 0 or self.indices.max() >= len(self.labels_all)):
            raise IndexError("A split index is out of range.")
        if not set(np.unique(self.labels_all).tolist()).issubset(set(range(NUM_CLASSES))):
            raise ValueError("Expected original Galaxy10 labels 0-9.")
        self.labels = self.labels_all[self.indices]

    def __len__(self) -> int:
        return len(self.indices)

    def _open_images(self):
        if self.images is None:
            self.images = np.load(self._images_path, mmap_mode="r")
        return self.images

    def __getitem__(self, item: int):
        row_id = int(self.indices[item])
        array = np.asarray(self._open_images()[row_id])
        if array.dtype != np.uint8:
            array = np.clip(array, 0, 255).astype(np.uint8)
        image = Image.fromarray(array).convert("RGB")
        label = int(self.labels[item])
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    def __getstate__(self):
        state = self.__dict__.copy()
        state["images"] = None
        return state

    def close(self):
        self.images = None
