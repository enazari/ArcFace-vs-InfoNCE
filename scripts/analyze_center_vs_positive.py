"""Center vs Positive distance analysis.

For each positive pair (A, B) in LFW/CFP benchmarks, compare:
  - cos(A, B)                    — positive pair similarity
  - max_j cos(A, center_j)       — A's closest class center
  - max_j cos(B, center_j)       — B's closest class center

Reports how often the closest center is nearer than the positive pair.
"""

import csv
import os

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from src.eval.cfp import _load_cfp
from src.eval.lfw import _load_lfw
from src.backbones.factory import build_backbone
from src.data.dataset import LMDBFaceDataset

MODELS = {
    "arcface":          ("configs/arc.yaml",     "sessions/arcface/best.pth"),
    "arcface_infonce":  ("configs/arc-nce.yaml", "sessions/arcface_infonce/best.pth"),
    "nce":              ("configs/nce.yaml",     "sessions/nce/best.pth"),
}

BENCHMARKS = ["LFW", "CFP-FF", "CFP-FP"]
DATA_DIR = "data"


@torch.no_grad()
def embed_faces(backbone, face_list, device):
    """Embed a list of uint8 face arrays. Returns L2-normalized [N, 512] tensor."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    embs, batch = [], []
    for face in tqdm(face_list, desc="  Embedding faces", leave=False):
        batch.append(transform(Image.fromarray(face)))
        if len(batch) == 64:
            t = torch.stack(batch).to(device)
            embs.append(F.normalize(backbone(t), dim=1).cpu())
            batch = []
    if batch:
        t = torch.stack(batch).to(device)
        embs.append(F.normalize(backbone(t), dim=1).cpu())
    return torch.cat(embs)


@torch.no_grad()
def compute_nce_centroids(backbone, val_lmdb_path, device):
    """Compute class centroids from val LMDB for headless models."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    ds = LMDBFaceDataset(val_lmdb_path, transform=transform)
    loader = DataLoader(ds, batch_size=256, shuffle=False, num_workers=4)

    num_classes = ds.num_classes
    emb_dim = 512
    sums = torch.zeros(num_classes, emb_dim)
    counts = torch.zeros(num_classes)

    backbone.eval()
    for imgs, labels in tqdm(loader, desc="  Computing centroids", leave=False):
        e = F.normalize(backbone(imgs.to(device)), dim=1).cpu()
        for i in range(len(labels)):
            lbl = labels[i].item()
            sums[lbl] += e[i]
            counts[lbl] += 1

    # Only keep classes with at least 1 sample
    mask = counts > 0
    centroids = sums[mask] / counts[mask].unsqueeze(1)
    return F.normalize(centroids, dim=1)


def load_benchmark(benchmark):
    """Load faces and labels for a benchmark. Returns (faces_a, faces_b, labels)."""
    if benchmark == "LFW":
        faces_a, faces_b, labels, _ = _load_lfw(DATA_DIR)
    elif benchmark == "CFP-FF":
        faces_a, faces_b, labels, _ = _load_cfp(DATA_DIR, "FF")
    elif benchmark == "CFP-FP":
        faces_a, faces_b, labels, _ = _load_cfp(DATA_DIR, "FP")
    return faces_a, faces_b, labels


def max_center_sim(embs, centers, chunk_size=1000):
    """Compute max cosine similarity to any center for each embedding.

    Processes in chunks to avoid OOM on [N, 85742] matrix.
    """
    results = []
    for i in range(0, len(embs), chunk_size):
        chunk = embs[i:i + chunk_size]
        sims = chunk @ centers.T  # [chunk, C]
        results.append(sims.max(dim=1).values)
    return torch.cat(results)


def analyze(model_name, backbone, centers, device):
    """Run analysis for one model across all benchmarks."""
    backbone.eval()
    results = []

    for bench in tqdm(BENCHMARKS, desc="  Benchmarks", leave=False):
        faces_a, faces_b, labels = load_benchmark(bench)

        embs_a = embed_faces(backbone, faces_a, device)
        embs_b = embed_faces(backbone, faces_b, device)

        # Filter to positive pairs only
        pos_mask = labels == 1
        ea = embs_a[pos_mask]
        eb = embs_b[pos_mask]
        n_pos = pos_mask.sum()

        # Positive pair similarity
        sim_pos = (ea * eb).sum(dim=1)

        # Closest center similarity
        sim_a_center = max_center_sim(ea, centers)
        sim_b_center = max_center_sim(eb, centers)

        # Fraction where center is closer than positive
        a_closer = (sim_a_center > sim_pos).float().mean().item()
        b_closer = (sim_b_center > sim_pos).float().mean().item()

        row = {
            "model": model_name,
            "benchmark": bench,
            "n_positive_pairs": int(n_pos),
            "mean_sim_pos": sim_pos.mean().item(),
            "mean_sim_A_center": sim_a_center.mean().item(),
            "mean_sim_B_center": sim_b_center.mean().item(),
            "frac_A_center_closer": a_closer,
            "frac_B_center_closer": b_closer,
        }
        results.append(row)

        print(f"  {bench:8s}  pos={row['mean_sim_pos']:.4f}  "
              f"A_ctr={row['mean_sim_A_center']:.4f}  B_ctr={row['mean_sim_B_center']:.4f}  "
              f"A_closer={a_closer:.4f}  B_closer={b_closer:.4f}")

    return results


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_results = []

    for model_name, (config_path, ckpt_path) in tqdm(MODELS.items(), desc="Models", total=len(MODELS)):
        print(f"\n{'='*60}")
        print(f"Model: {model_name}")
        print(f"{'='*60}")

        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        backbone = build_backbone(cfg).to(device)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        backbone.load_state_dict(ckpt["backbone"])
        backbone.eval()

        # Get class centers
        if ckpt.get("head") is not None:
            centers = F.normalize(ckpt["head"]["weight"], dim=1)
            print(f"  Class centers from head: {centers.shape}")
        else:
            val_lmdb = os.path.join(DATA_DIR, f"ms1m_{cfg['data']['num_identities']}_val.lmdb")
            print(f"  Computing centroids from {val_lmdb}...")
            centers = compute_nce_centroids(backbone, val_lmdb, device)
            print(f"  Centroids: {centers.shape}")

        results = analyze(model_name, backbone, centers, device)
        all_results.extend(results)

        # Save per-model CSV into its session folder
        session_dir = os.path.dirname(ckpt_path)
        per_model_path = os.path.join(session_dir, "center_vs_positive.csv")
        fields = list(results[0].keys())
        with open(per_model_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(results)
        print(f"  Saved → {per_model_path}")

    # Save combined CSV into sessions/
    csv_path = os.path.join("sessions", "center_vs_positive_results.csv")
    fields = list(all_results[0].keys())
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(all_results)
    print(f"\nSaved → {csv_path}")


if __name__ == "__main__":
    main()
