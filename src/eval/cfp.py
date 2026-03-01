"""CFP 10-fold evaluation (FF and FP protocols) with LMDB caching.

Expects the CFP dataset at data/cfp-dataset/ (manual download from Kaggle).
Detects + aligns faces with RetinaFace (reuses _detect_align from lfw.py),
caches in LMDB, evaluates with 10-fold cross-validation at FAR@0.001.
"""

import csv
import os
import pickle
import shutil

import cv2
import lmdb
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from .lfw import _detect_align

LMDB_NAME_FF = "cfp_ff_10fold_retinaface.lmdb"
LMDB_NAME_FP = "cfp_fp_10fold_retinaface.lmdb"
CFP_SUBDIR = "cfp-dataset"

_LMDB_NAMES = {"FF": LMDB_NAME_FF, "FP": LMDB_NAME_FP}


# ---------------------------------------------------------------------------
# Parse CFP protocol
# ---------------------------------------------------------------------------

def _load_pair_list(path):
    """Parse Pair_list_{F,P}.txt → {1-based index: absolute image path}."""
    base_dir = os.path.dirname(path)
    index_to_path = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            idx = int(parts[0])
            rel_path = parts[1]
            index_to_path[idx] = os.path.normpath(os.path.join(base_dir, rel_path))
    return index_to_path


def _parse_cfp_protocol(cfp_dir, protocol):
    """Parse CFP protocol into 10 folds of (path_a, path_b, label) triples.

    protocol: "FF" or "FP"
    """
    proto_dir = os.path.join(cfp_dir, "Protocol")
    frontal_map = _load_pair_list(os.path.join(proto_dir, "Pair_list_F.txt"))
    map_b = frontal_map
    if protocol == "FP":
        map_b = _load_pair_list(os.path.join(proto_dir, "Pair_list_P.txt"))

    folds = []
    for fold_idx in range(1, 11):
        fold = []
        fold_dir = os.path.join(proto_dir, "Split", protocol, f"{fold_idx:02d}")
        for fname, label in [("same.txt", 1), ("diff.txt", 0)]:
            with open(os.path.join(fold_dir, fname)) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    a_str, b_str = line.split(",")
                    fold.append((frontal_map[int(a_str)], map_b[int(b_str)], label))
        folds.append(fold)
    return folds


# ---------------------------------------------------------------------------
# LMDB preparation
# ---------------------------------------------------------------------------

def prepare_cfp(data_dir="data"):
    """Detect+align CFP faces, write LMDB for FF and FP protocols."""
    cfp_dir = os.path.join(data_dir, CFP_SUBDIR)
    if not os.path.isdir(cfp_dir):
        raise FileNotFoundError(
            f"CFP dataset not found at {cfp_dir}. "
            "Download from https://www.kaggle.com/datasets/chinafax/cfpw-dataset "
            "and extract to data/cfp-dataset/"
        )

    for protocol, lmdb_name in [("FF", LMDB_NAME_FF), ("FP", LMDB_NAME_FP)]:
        lmdb_path = os.path.join(data_dir, lmdb_name)
        if os.path.exists(lmdb_path):
            print(f"CFP-{protocol} LMDB exists: {lmdb_path}")
            continue

        folds = _parse_cfp_protocol(cfp_dir, protocol)
        print(f"Detecting + aligning CFP-{protocol} faces...")

        # Cache aligned faces to avoid re-detecting the same image
        align_cache = {}
        failed_paths = set()

        def get_aligned(path):
            if path not in align_cache:
                img = np.array(Image.open(path).convert("RGB"))
                result = _detect_align(img)
                align_cache[path] = result
                if result is None:
                    failed_paths.add(path)
            return align_cache[path]

        kept_records = []
        fold_sizes = []
        failed = 0

        for fold_idx, fold in enumerate(folds):
            fold_kept = 0
            for path_a, path_b, label in tqdm(fold, desc=f"CFP-{protocol} fold {fold_idx+1}/10", leave=False):
                aligned_a = get_aligned(path_a)
                aligned_b = get_aligned(path_b)
                if aligned_a is None or aligned_b is None:
                    failed += 1
                    continue
                kept_records.append((aligned_a, aligned_b, label))
                fold_kept += 1
            fold_sizes.append(fold_kept)

        total = sum(len(f) for f in folds)
        print(f"CFP-{protocol}: aligned {len(kept_records)}/{total} pairs ({failed} failed)")

        # Save failed detection images for sanity checking
        if failed_paths:
            fail_dir = os.path.join(data_dir, f"cfp_{protocol.lower()}_failed")
            if os.path.isdir(fail_dir):
                shutil.rmtree(fail_dir)
            os.makedirs(fail_dir)
            for path in sorted(failed_paths):
                shutil.copy(path, fail_dir)
            print(f"Saved {len(failed_paths)} failed images → {fail_dir}/")

        # Write LMDB
        os.makedirs(data_dir, exist_ok=True)
        map_size = len(kept_records) * 112 * 112 * 3 * 2 * 2
        env = lmdb.open(lmdb_path, map_size=max(map_size, 1 << 30))
        with env.begin(write=True) as txn:
            for i, (fa, fb, lbl) in enumerate(kept_records):
                key = f"{i:06d}".encode()
                txn.put(key, pickle.dumps({"face_a": fa, "face_b": fb, "label": lbl}))
            txn.put(b"__len__", str(len(kept_records)).encode())
            txn.put(b"__fold_sizes__", pickle.dumps(fold_sizes))
        env.close()
        print(f"Wrote {lmdb_path} ({len(kept_records)} pairs, folds={fold_sizes})")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _load_cfp(data_dir, protocol):
    """Load CFP pairs from LMDB. Returns (faces_a, faces_b, labels, fold_sizes)."""
    lmdb_path = os.path.join(data_dir, _LMDB_NAMES[protocol])
    env = lmdb.open(lmdb_path, readonly=True, lock=False)
    with env.begin() as txn:
        n = int(txn.get(b"__len__").decode())
        fold_sizes = pickle.loads(txn.get(b"__fold_sizes__"))
        faces_a, faces_b, labels = [], [], []
        for i in range(n):
            rec = pickle.loads(txn.get(f"{i:06d}".encode()))
            faces_a.append(rec["face_a"])
            faces_b.append(rec["face_b"])
            labels.append(rec["label"])
    env.close()
    return faces_a, faces_b, np.array(labels), fold_sizes


@torch.no_grad()
def evaluate_cfp(backbone, data_dir, device, protocol, output_dir=None):
    """Evaluate backbone on CFP-FF or CFP-FP 10-fold. Returns (mean_acc, std_acc, mean_threshold).

    If output_dir is given, writes cfp_{ff,fp}_results.csv with per-fold details.
    """
    faces_a, faces_b, labels, fold_sizes = _load_cfp(data_dir, protocol)
    tag = f"CFP-{protocol}"
    n = len(labels)
    print(f"{tag} eval: {n} pairs, folds={fold_sizes}")

    # Extract embeddings
    backbone.eval()
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    def embed_faces(face_list):
        embs = []
        batch = []
        for face in face_list:
            batch.append(transform(Image.fromarray(face)))
            if len(batch) == 64:
                t = torch.stack(batch).to(device)
                e = F.normalize(backbone(t), dim=1)
                embs.append(e.cpu())
                batch = []
        if batch:
            t = torch.stack(batch).to(device)
            e = F.normalize(backbone(t), dim=1)
            embs.append(e.cpu())
        return torch.cat(embs)

    embs_a = embed_faces(faces_a)
    embs_b = embed_faces(faces_b)
    sims = (embs_a * embs_b).sum(dim=1).numpy()

    # 10-fold cross-validation
    fold_starts = np.cumsum([0] + fold_sizes)
    accs, thresholds = [], []
    n_same_correct_list, n_diff_correct_list = [], []
    n_same_list, n_diff_list = [], []

    for i in range(10):
        test_mask = np.zeros(n, dtype=bool)
        test_mask[fold_starts[i]:fold_starts[i + 1]] = True
        train_mask = ~test_mask

        # Find threshold at FAR <= 0.001 from train folds
        train_diff_sims = sims[train_mask & (labels == 0)]
        sorted_diffs = np.sort(train_diff_sims)[::-1]
        max_fa = int(np.floor(len(sorted_diffs) * 0.001))
        if max_fa == 0:
            thresh = float(sorted_diffs[0]) + 1e-6
        else:
            thresh = float(sorted_diffs[max_fa - 1])

        # Test accuracy
        test_sims = sims[test_mask]
        test_labels = labels[test_mask]
        preds = (test_sims >= thresh).astype(int)
        acc = (preds == test_labels).mean()

        # Per-class breakdown
        same_mask = test_labels == 1
        diff_mask = test_labels == 0
        n_same_list.append(int(same_mask.sum()))
        n_diff_list.append(int(diff_mask.sum()))
        n_same_correct_list.append(int((preds[same_mask] == 1).sum()))
        n_diff_correct_list.append(int((preds[diff_mask] == 0).sum()))

        accs.append(acc)
        thresholds.append(thresh)
        print(f"  fold {i+1}: acc={acc:.4f}  thresh={thresh:.4f}")

    mean_acc = np.mean(accs)
    std_acc = np.std(accs)
    mean_thresh = np.mean(thresholds)
    print(f"  mean: acc={mean_acc:.4f} ± {std_acc:.4f}  thresh={mean_thresh:.4f}")

    # Save detailed CSV
    if output_dir is not None:
        csv_name = f"cfp_{protocol.lower()}_results.csv"
        csv_path = os.path.join(output_dir, csv_name)
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["fold", "accuracy", "threshold",
                         "n_same", "same_correct", "TAR",
                         "n_diff", "diff_correct", "1-FAR"])
            for i in range(10):
                tar = n_same_correct_list[i] / n_same_list[i] if n_same_list[i] else 0
                spec = n_diff_correct_list[i] / n_diff_list[i] if n_diff_list[i] else 0
                w.writerow([
                    i + 1,
                    f"{accs[i]:.6f}", f"{thresholds[i]:.6f}",
                    n_same_list[i], n_same_correct_list[i], f"{tar:.6f}",
                    n_diff_list[i], n_diff_correct_list[i], f"{spec:.6f}",
                ])
            w.writerow([
                "mean",
                f"{mean_acc:.6f}", f"{mean_thresh:.6f}",
                "", "", f"{np.mean([sc/ns for sc, ns in zip(n_same_correct_list, n_same_list)]):.6f}",
                "", "", f"{np.mean([dc/nd for dc, nd in zip(n_diff_correct_list, n_diff_list)]):.6f}",
            ])
            w.writerow([
                "std",
                f"{std_acc:.6f}", "",
                "", "", "",
                "", "", "",
            ])
        print(f"  saved → {csv_path}")

    return mean_acc, std_acc, mean_thresh
