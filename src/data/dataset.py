"""LMDB-backed face dataset."""

import io
import pickle

import lmdb
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


def default_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])


class LMDBFaceDataset(Dataset):
    """Face dataset backed by an LMDB written by prepare_data.py."""

    def __init__(self, lmdb_path: str, transform=None):
        self.lmdb_path = lmdb_path
        self.transform = transform or default_transform()
        self._env = None  # opened lazily per worker

        # Read metadata without keeping env open
        env = lmdb.open(lmdb_path, readonly=True, lock=False)
        with env.begin() as txn:
            self._len       = int(txn.get(b"__len__").decode())
            self._nclasses  = int(txn.get(b"__nclasses__").decode())
        env.close()

    def _open_env(self):
        if self._env is None:
            self._env = lmdb.open(self.lmdb_path, readonly=True, lock=False)

    @property
    def num_classes(self) -> int:
        return self._nclasses

    def get_labels(self) -> np.ndarray:
        """Return all labels as a numpy array (used by PKSampler)."""
        env = lmdb.open(self.lmdb_path, readonly=True, lock=False)
        labels = np.empty(self._len, dtype=np.int64)
        with env.begin() as txn:
            for i in range(self._len):
                value = txn.get(f"{i:09d}".encode())
                labels[i] = pickle.loads(value)["label"]
        env.close()
        return labels

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        self._open_env()
        key   = f"{idx:09d}".encode()
        with self._env.begin() as txn:
            value = txn.get(key)
        record = pickle.loads(value)
        image  = Image.open(io.BytesIO(record["jpeg"])).convert("RGB")
        return self.transform(image), record["label"]
