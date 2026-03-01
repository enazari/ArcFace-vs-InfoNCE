"""Identity-aware batch samplers for metric learning."""

import random
from collections import defaultdict
from typing import Iterator

import numpy as np
from torch.utils.data import Sampler


class PairSampler(Sampler):
    """
    Yields batches of P identities × 2 images (CLIP-style pairing).

    Images within each identity are shuffled and paired. Round-robin across
    identities ensures ~99% of all images are seen per epoch.

    Pairs are adjacent in each batch: (pair0_img0, pair0_img1, pair1_img0, ...).
    """

    def __init__(self, labels: np.ndarray, P: int):
        self.P = P
        self.id_to_indices: dict[int, list[int]] = defaultdict(list)
        for idx, label in enumerate(labels):
            self.id_to_indices[int(label)].append(idx)
        self.identities = list(self.id_to_indices.keys())

    def __iter__(self) -> Iterator[list[int]]:
        # 1. Form pairs within each identity
        identity_pairs: dict[int, list[tuple[int, int]]] = {}
        for ident in self.identities:
            indices = list(self.id_to_indices[ident])
            random.shuffle(indices)
            identity_pairs[ident] = [
                (indices[i], indices[i + 1])
                for i in range(0, len(indices) - 1, 2)
            ]

        # 2. Round-robin: one pair per identity per round, chunk into batches of P
        max_rounds = max(len(ps) for ps in identity_pairs.values())
        batches = []
        for r in range(max_rounds):
            round_pairs = []
            for ident in self.identities:
                if r < len(identity_pairs[ident]):
                    round_pairs.append(identity_pairs[ident][r])
            random.shuffle(round_pairs)
            for i in range(0, len(round_pairs) - self.P + 1, self.P):
                batch = []
                for i1, i2 in round_pairs[i : i + self.P]:
                    batch.extend([i1, i2])
                batches.append(batch)

        random.shuffle(batches)
        for batch in batches:
            yield batch

    def __len__(self) -> int:
        total_pairs = sum(len(v) // 2 for v in self.id_to_indices.values())
        return total_pairs // self.P
