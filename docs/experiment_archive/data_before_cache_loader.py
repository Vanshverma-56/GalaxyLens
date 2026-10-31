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
    """Reads images lazily from the official HDF5 arrays `images` and `ans`.

    `indices` are stable integer row IDs into the original HDF5 file. Each worker
    opens its own HDF5 handle on first access to avoid sharing file handles.
    """
    def __init__(
        self,
        hdf5_path: str | Path,
        indices: list[int] | np.ndarray,
        transform: Callable | None = None,
    ):
        self.hdf5_path = str(hdf5_path)
        self.indices = np.asarray(indices, dtype=np.int64)
        self.transform = transform
        if not Path(self.hdf5_path).exists():
            raise FileNotFoundError(f"Dataset file not found: {self.hdf5_path}")
        with h5py.File(self.hdf5_path, "r") as h5:
            if "images" not in h5 or "ans" not in h5:
                raise ValueError("HDF5 must contain datasets named 'images' and 'ans'.")
            labels = np.asarray(h5["ans"]).reshape(-1).astype(np.int64)
            if len(labels) != h5["images"].shape[0]:
                raise ValueError("Image and label counts differ.")
            if self.indices.size and (self.indices.min() < 0 or self.indices.max() >= len(labels)):
                raise IndexError("A split index is outside the dataset.")
            if not set(np.unique(labels).tolist()).issubset(set(range(NUM_CLASSES))):
                raise ValueError("Found labels outside the expected original class range 0-9.")
            self.labels = labels[self.indices]
        self._h5 = None

    def __len__(self) -> int:
        return len(self.indices)

    def _file(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.hdf5_path, "r")
        return self._h5

    def __getitem__(self, item: int):
        row_id = int(self.indices[item])
        array = np.asarray(self._file()["images"][row_id])
        if array.dtype != np.uint8:
            # Galaxy10 is expected to be byte-valued. Convert safely if another
            # compatible release stores numeric pixels in a different dtype.
            array = np.clip(array, 0, 255).astype(np.uint8)
        image = Image.fromarray(array).convert("RGB")
        label = int(self.labels[item])
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_h5"] = None
        return state

    def close(self):
        if self._h5 is not None:
            self._h5.close()
            self._h5 = None
