"""Training script for face verification."""

import argparse
import csv
import math
import os
import shutil
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.optim as optim
from accelerate import Accelerator
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

from src.data.prepare import prepare, lmdb_paths
from src.backbones.factory import build_backbone
from src.data.dataset import LMDBFaceDataset
from src.data.sampler import PairSampler
from src.heads.arcface_head import ArcFaceHead
from src.losses.factory import build_loss


# ---------------------------------------------------------------------------
# LR scheduler: cosine with linear warmup
# ---------------------------------------------------------------------------

def cosine_lr(optimizer, epoch: int, total_epochs: int, warmup_epochs: int,
              base_lr: float) -> None:
    if epoch < warmup_epochs:
        lr = base_lr * (epoch + 1) / warmup_epochs
    else:
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        lr = base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))
    for g in optimizer.param_groups:
        g['lr'] = lr


# ---------------------------------------------------------------------------
# One training epoch
# ---------------------------------------------------------------------------

def train_epoch(backbone, head, loss_fn, loader, optimizer, accelerator, scale):
    backbone.train()
    if head is not None:
        head.train()
    total_loss = 0.0
    correct = 0
    total   = 0

    all_params = list(backbone.parameters())
    if head is not None:
        all_params += list(head.parameters())

    for imgs, labels in tqdm(loader, desc="  train", leave=False,
                             disable=not accelerator.is_main_process):
        optimizer.zero_grad()
        emb = backbone(imgs)
        emb_norm = F.normalize(emb, dim=1)

        if head is not None:
            logits = head(emb)
            loss   = loss_fn(logits, labels, emb_norm)
            accelerator.backward(loss)
            accelerator.clip_grad_norm_(all_params, max_norm=5.0)
            optimizer.step()
            correct += ((logits * scale).argmax(1) == labels).sum().item()
        else:
            loss = loss_fn(emb_norm, labels)
            accelerator.backward(loss)
            accelerator.clip_grad_norm_(all_params, max_norm=5.0)
            optimizer.step()
            # Pair accuracy: is the positive partner the top-1 nearest?
            with torch.no_grad():
                sim = emb_norm @ emb_norm.T
                sim.fill_diagonal_(float('-inf'))
                B = emb_norm.size(0)
                targets = torch.arange(B, device=sim.device)
                targets[0::2] = torch.arange(1, B, 2, device=sim.device)
                targets[1::2] = torch.arange(0, B, 2, device=sim.device)
                correct += (sim.argmax(1) == targets).sum().item()

        total_loss += loss.item()
        total += labels.size(0)

    return total_loss / len(loader), correct / total


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def val_epoch(backbone, head, loss_fn, loader, accelerator, scale):
    backbone.eval()
    head.eval()
    total_loss = 0.0
    correct = 0
    total   = 0

    for imgs, labels in tqdm(loader, desc="  val", leave=False,
                             disable=not accelerator.is_main_process):
        emb    = backbone(imgs)
        logits = head(emb)
        loss   = loss_fn(logits, labels)
        total_loss += loss.item()
        correct += ((logits * scale).argmax(1) == labels).sum().item()
        total   += labels.size(0)

    return total_loss / len(loader), correct / total


# val not meaningful for pure contrastive losses (no head, no classification pairs)
_HEADLESS_LOSSES = ("infonce",)


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

def save_checkpoint(accelerator, backbone, head, optimizer, epoch, output_dir, name):
    if not accelerator.is_main_process:
        return
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    path = os.path.join(output_dir, f"{name}.pth")
    state = {
        "backbone": accelerator.unwrap_model(backbone).state_dict(),
        "epoch": epoch,
    }
    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
    if head is not None:
        state["head"] = accelerator.unwrap_model(head).state_dict()
    torch.save(state, path)
    print(f"  saved → {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="arcface_r50")
    args = parser.parse_args()

    with open(f"configs/{args.config}.yaml") as f:
        cfg = yaml.safe_load(f)

    use_pairs = cfg["loss"]["name"] in ("arcface_infonce", "infonce")
    use_head  = cfg["loss"]["name"] not in _HEADLESS_LOSSES
    accelerator = Accelerator(
        mixed_precision=cfg["training"].get("mixed_precision", "no"),
    )
    if use_pairs:
        accelerator.even_batches = False

    output_dir = f"sessions/{cfg['session']}"
    epochs = cfg["training"]["epochs"]
    if accelerator.is_main_process:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        shutil.copy(f"configs/{args.config}.yaml", output_dir)

    # 4. Model
    backbone = build_backbone(cfg)

    if epochs == 0:
        # Eval-only: skip data loading, save initial model, jump to evaluation
        backbone = accelerator.prepare(backbone)
        save_checkpoint(accelerator, backbone, None, None, 0, output_dir, "best")
    else:
        # 1. Prepare data (no-op if LMDBs exist)
        if accelerator.is_main_process:
            prepare(cfg)
        accelerator.wait_for_everyone()

        # 2. Train dataset + loader
        train_path, val_path = lmdb_paths(cfg)
        batch_size  = cfg["training"]["batch_size"]
        num_workers = cfg["training"].get("num_workers", 4)
        train_dataset = LMDBFaceDataset(train_path)
        if accelerator.is_main_process:
            print(f"Train: {len(train_dataset)} images, {train_dataset.num_classes} classes")

        if use_pairs:
            pair_sampler = PairSampler(train_dataset.get_labels(), P=batch_size // 2)
            train_loader = DataLoader(train_dataset, batch_sampler=pair_sampler,
                                      num_workers=num_workers, pin_memory=True)
        else:
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                      num_workers=num_workers, pin_memory=True, drop_last=True)

        # 3. Val dataset + loader
        val_dataset = LMDBFaceDataset(val_path)
        if accelerator.is_main_process:
            print(f"Val:   {len(val_dataset)} images")
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                num_workers=num_workers, pin_memory=True)

        head     = ArcFaceHead(cfg["backbone"]["embedding_dim"],
                                train_dataset.num_classes) if use_head else None
        loss_fn  = build_loss(cfg).to(accelerator.device)

        # 5. Optimizer (only trainable params — matters for LoRA)
        params = [p for p in backbone.parameters() if p.requires_grad]
        if head is not None:
            params += list(head.parameters())
        optimizer = optim.SGD(params,
                              lr=cfg["training"]["lr"],
                              momentum=cfg["training"]["momentum"],
                              weight_decay=cfg["training"]["weight_decay"])

        # 6. Resume from checkpoint if session exists
        start_epoch = 1
        ckpt_path = os.path.join(output_dir, "last.pth")
        ckpt = None
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            backbone.load_state_dict(ckpt["backbone"])
            if head is not None and "head" in ckpt:
                head.load_state_dict(ckpt["head"])
            start_epoch = ckpt["epoch"] + 1

        if head is not None:
            backbone, head, optimizer, train_loader, val_loader = accelerator.prepare(
                backbone, head, optimizer, train_loader, val_loader
            )
        else:
            backbone, optimizer, train_loader, val_loader = accelerator.prepare(
                backbone, optimizer, train_loader, val_loader
            )

        if ckpt is not None and "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
            if accelerator.is_main_process:
                print(f"Resumed from epoch {ckpt['epoch']}")
            ckpt = None  # free memory

    # 7. Training loop
    warmup     = cfg["training"]["warmup_epochs"]
    eval_every = cfg["training"]["eval_every"]
    base_lr    = cfg["training"]["lr"]
    scale      = cfg["loss"].get("scale", 1.0)

    csv_file = None
    writer   = None
    start_epoch = start_epoch if epochs > 0 else 1
    resuming = start_epoch > 1
    if accelerator.is_main_process:
        csv_path = os.path.join(output_dir, "metrics.csv")
        csv_file = open(csv_path, "a" if resuming else "w", newline="")
        writer = csv.writer(csv_file)
        if not resuming:
            writer.writerow(["epoch", "lr", "train_loss", "train_acc", "val_loss", "val_acc"])

    best_loss = float("inf")

    for epoch in range(start_epoch, epochs + 1):
        cosine_lr(optimizer, epoch - 1, epochs, warmup, base_lr)
        lr = optimizer.param_groups[0]['lr']
        if accelerator.is_main_process:
            print(f"Epoch {epoch}/{epochs}  lr={lr:.5f}")

        train_loss, train_acc = train_epoch(backbone, head, loss_fn, train_loader,
                                            optimizer, accelerator, scale)
        if accelerator.is_main_process:
            print(f"  train_loss={train_loss:.4f}  train_acc={train_acc:.4f}")

        val_loss, val_acc = "", ""
        if use_head and (epoch % eval_every == 0 or epoch == epochs):
            v_loss, v_acc = val_epoch(backbone, head, loss_fn, val_loader,
                                      accelerator, scale)
            if accelerator.is_main_process:
                print(f"  val_loss={v_loss:.4f}  val_acc={v_acc:.4f}")
                val_loss, val_acc = f"{v_loss:.6f}", f"{v_acc:.6f}"
            if v_loss < best_loss:
                best_loss = v_loss
                save_checkpoint(accelerator, backbone, head, optimizer, epoch, output_dir, "best")
        elif not use_head:
            # Headless: use train_loss for best checkpoint selection
            if train_loss < best_loss:
                best_loss = train_loss
                save_checkpoint(accelerator, backbone, head, optimizer, epoch, output_dir, "best")

        if accelerator.is_main_process:
            writer.writerow([epoch, f"{lr:.6f}", f"{train_loss:.6f}", f"{train_acc:.6f}",
                             val_loss, val_acc])
            csv_file.flush()

        save_checkpoint(accelerator, backbone, head, optimizer, epoch, output_dir, "last")

    # 8. LFW 10-fold evaluation on best model
    if accelerator.is_main_process:
        from src.eval.lfw import prepare_lfw, evaluate_lfw

        data_dir = os.environ.get("DATA_DIR", "data")
        prepare_lfw(data_dir)
        ckpt = torch.load(os.path.join(output_dir, "best.pth"),
                          map_location="cpu", weights_only=True)
        accelerator.unwrap_model(backbone).load_state_dict(ckpt["backbone"])
        mean_acc, std_acc, mean_thresh = evaluate_lfw(
            accelerator.unwrap_model(backbone), data_dir, accelerator.device,
            output_dir=output_dir,
        )
        print(f"LFW 10-fold: acc={mean_acc:.4f} ± {std_acc:.4f}, "
              f"thresh@FAR0.001={mean_thresh:.4f}")
        if writer:
            writer.writerow(["lfw_eval", "", "", "", "", f"{mean_acc:.6f}"])

        # CFP-FF and CFP-FP 10-fold evaluation
        try:
            from src.eval.cfp import prepare_cfp, evaluate_cfp

            prepare_cfp(data_dir)
            for protocol in ("FF", "FP"):
                acc, std, thresh = evaluate_cfp(
                    accelerator.unwrap_model(backbone), data_dir,
                    accelerator.device, protocol, output_dir=output_dir,
                )
                print(f"CFP-{protocol} 10-fold: acc={acc:.4f} ± {std:.4f}, "
                      f"thresh@FAR0.001={thresh:.4f}")
                if writer:
                    writer.writerow([f"cfp_{protocol.lower()}_eval", "", "", "", "", f"{acc:.6f}"])
        except FileNotFoundError as e:
            print(f"Skipping CFP evaluation: {e}")

    if csv_file:
        csv_file.close()


if __name__ == "__main__":
    main()
