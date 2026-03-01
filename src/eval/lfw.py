"""LFW 10-fold evaluation with self-reliant LMDB creation.

Downloads LFW (250x250 originals) from figshare, detects + aligns faces with
RetinaFace, caches in LMDB, evaluates with 10-fold cross-validation at FAR@0.001.
"""

import csv
import os
import pickle
import tarfile
import urllib.request

import cv2
import lmdb
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from skimage.transform import SimilarityTransform
from torchvision import transforms
from tqdm import tqdm

# ArcFace reference landmarks for 112x112 alignment
ARCFACE_REF = np.array([
    [38.2946, 51.6963],   # left eye (viewer)
    [73.5318, 51.5014],   # right eye (viewer)
    [56.0252, 71.7366],   # nose
    [41.5493, 92.3655],   # left mouth (viewer)
    [70.7299, 92.2041],   # right mouth (viewer)
], dtype=np.float32)

LMDB_NAME = "lfw_10fold_original_retinaface.lmdb"

# Same figshare URLs used by scikit-learn
_LFW_TGZ_URL = "https://ndownloader.figshare.com/files/5976018"
_PAIRS_URL = "https://ndownloader.figshare.com/files/5976006"


# ---------------------------------------------------------------------------
# Download + parse
# ---------------------------------------------------------------------------

def _download_file(url, dest, desc="Downloading"):
    """Download a file with tqdm progress bar."""
    resp = urllib.request.urlopen(url)
    total = int(resp.headers.get("Content-Length", 0))
    with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=desc) as bar:
        while True:
            chunk = resp.read(1 << 16)
            if not chunk:
                break
            f.write(chunk)
            bar.update(len(chunk))


def _ensure_lfw_downloaded(cache_dir):
    """Download LFW tarball + pairs.txt if needed. Returns (lfw_dir, pairs_path)."""
    os.makedirs(cache_dir, exist_ok=True)
    lfw_dir = os.path.join(cache_dir, "lfw")
    pairs_path = os.path.join(cache_dir, "pairs.txt")

    if not os.path.exists(pairs_path):
        _download_file(_PAIRS_URL, pairs_path, "pairs.txt")

    if not os.path.isdir(lfw_dir):
        tgz_path = os.path.join(cache_dir, "lfw.tgz")
        if not os.path.exists(tgz_path):
            _download_file(_LFW_TGZ_URL, tgz_path, "lfw.tgz (~173 MB)")
        print("Extracting lfw.tgz...")
        with tarfile.open(tgz_path) as tf:
            tf.extractall(cache_dir)
        os.remove(tgz_path)

    return lfw_dir, pairs_path


def _parse_pairs(pairs_path, lfw_dir):
    """Parse pairs.txt → list of 10 folds, each a list of (path_a, path_b, label)."""
    folds = []
    with open(pairs_path) as f:
        header = f.readline().strip().split()
        n_folds, n_per_class = int(header[0]), int(header[1])
        n_same = n_per_class
        n_diff = n_per_class
        for _ in range(n_folds):
            fold = []
            for _ in range(n_same):
                parts = f.readline().strip().split("\t")
                name, i1, i2 = parts[0], int(parts[1]), int(parts[2])
                fold.append((
                    os.path.join(lfw_dir, name, f"{name}_{i1:04d}.jpg"),
                    os.path.join(lfw_dir, name, f"{name}_{i2:04d}.jpg"),
                    1,
                ))
            for _ in range(n_diff):
                parts = f.readline().strip().split("\t")
                name1, i1, name2, i2 = parts[0], int(parts[1]), parts[2], int(parts[3])
                fold.append((
                    os.path.join(lfw_dir, name1, f"{name1}_{i1:04d}.jpg"),
                    os.path.join(lfw_dir, name2, f"{name2}_{i2:04d}.jpg"),
                    0,
                ))
            folds.append(fold)
    return folds


# ---------------------------------------------------------------------------
# Face detection + alignment
# ---------------------------------------------------------------------------

def _detect_align(img_rgb, padding=150):
    """Detect face with RetinaFace and align to 112x112.

    Adds gray padding around the image before detection to improve detection
    of faces near edges (especially profile views). Landmark coordinates are
    shifted back to the original image space before alignment.

    Returns aligned uint8 ndarray (112, 112, 3) or None on failure.
    """
    from retinaface import RetinaFace

    # Pad with gray to help RetinaFace detect faces near image borders
    h, w = img_rgb.shape[:2]
    padded = np.full((h + 2 * padding, w + 2 * padding, 3), 128, dtype=np.uint8)
    padded[padding:padding + h, padding:padding + w] = img_rgb

    padded_bgr = padded[:, :, ::-1].copy()
    faces = RetinaFace.detect_faces(padded_bgr, threshold=0.5)
    if not isinstance(faces, dict) or len(faces) == 0:
        return None

    # Select face closest to original image center (in padded coordinates)
    cx, cy = padding + w / 2, padding + h / 2
    best_key, best_dist = None, float("inf")
    for key, face in faces.items():
        x1, y1, x2, y2 = face["facial_area"]
        fx, fy = (x1 + x2) / 2, (y1 + y2) / 2
        dist = (fx - cx) ** 2 + (fy - cy) ** 2
        if dist < best_dist:
            best_dist = dist
            best_key = key

    lms = faces[best_key]["landmarks"]
    # RetinaFace uses anatomical L/R; swap to viewer perspective for InsightFace
    # Shift landmarks back from padded to original coordinate space
    landmarks = np.array([
        lms["right_eye"],    # viewer left eye
        lms["left_eye"],     # viewer right eye
        lms["nose"],
        lms["mouth_right"],  # viewer left mouth
        lms["mouth_left"],   # viewer right mouth
    ], dtype=np.float32)
    landmarks -= padding

    tform = SimilarityTransform()
    tform.estimate(landmarks, ARCFACE_REF)
    M = tform.params[:2]
    aligned = cv2.warpAffine(img_rgb, M, (112, 112), borderValue=0.0)
    return aligned.astype(np.uint8)


# ---------------------------------------------------------------------------
# LMDB preparation
# ---------------------------------------------------------------------------

def prepare_lfw(data_dir="data"):
    """Download LFW, detect+align faces, write LMDB."""
    lmdb_path = os.path.join(data_dir, LMDB_NAME)
    if os.path.exists(lmdb_path):
        print(f"LFW LMDB exists: {lmdb_path}")
        return

    cache_dir = os.path.join(data_dir, "lfw_raw")
    lfw_dir, pairs_path = _ensure_lfw_downloaded(cache_dir)
    folds = _parse_pairs(pairs_path, lfw_dir)

    print("Detecting + aligning faces (250x250 originals)...")
    kept_records = []
    fold_sizes = []
    failed = 0

    for fold_idx, fold in enumerate(folds):
        fold_kept = 0
        for path_a, path_b, label in tqdm(fold, desc=f"fold {fold_idx+1}/10", leave=False):
            img_a = np.array(Image.open(path_a).convert("RGB"))
            img_b = np.array(Image.open(path_b).convert("RGB"))
            aligned_a = _detect_align(img_a)
            aligned_b = _detect_align(img_b)
            if aligned_a is None or aligned_b is None:
                failed += 1
                continue
            kept_records.append((aligned_a, aligned_b, label))
            fold_kept += 1
        fold_sizes.append(fold_kept)

    total = sum(len(f) for f in folds)
    print(f"Aligned {len(kept_records)}/{total} pairs ({failed} failed)")

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

def _load_lfw(data_dir="data"):
    """Load LFW pairs from LMDB. Returns (faces_a, faces_b, labels, fold_sizes)."""
    lmdb_path = os.path.join(data_dir, LMDB_NAME)
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
def evaluate_lfw(backbone, data_dir, device, output_dir=None):
    """Evaluate backbone on LFW 10-fold. Returns (mean_acc, std_acc, mean_threshold).

    If output_dir is given, writes lfw_results.csv with per-fold details.
    """
    faces_a, faces_b, labels, fold_sizes = _load_lfw(data_dir)
    n = len(labels)
    print(f"LFW eval: {n} pairs, folds={fold_sizes}")

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
        csv_path = os.path.join(output_dir, "lfw_results.csv")
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
