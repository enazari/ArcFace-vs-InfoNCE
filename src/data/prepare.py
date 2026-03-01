"""
Convert MS1M-ArcFace RecordIO dataset to LMDB (train + val split).

Reads train.rec + train.idx, selects top-N identities by image count,
splits 80/20 per identity, remaps labels to 0..N-1, writes two LMDBs.

Run once before training. Skips if LMDBs already exist.
"""

import os
import struct
import pickle
import random
from pathlib import Path
from collections import defaultdict

import lmdb
from tqdm import tqdm


RECORDIO_MAGIC = 0xCED7230A


def parse_idx(idx_path: str) -> dict[int, int]:
    """Parse train.idx → {record_id: byte_offset}."""
    offsets = {}
    with open(idx_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record_id, offset = line.split('\t')
            offsets[int(record_id)] = int(offset)
    return offsets


def read_record(f, offset: int) -> tuple[int, bytes] | None:
    """Read one RecordIO record. Returns (label_int, jpeg_bytes) or None."""
    f.seek(offset)
    magic = struct.unpack('<I', f.read(4))[0]
    assert magic == RECORDIO_MAGIC, f"Bad magic at offset {offset}: {magic:#x}"
    lrecord = struct.unpack('<I', f.read(4))[0]
    cflag  = lrecord >> 29
    length = lrecord & 0x1FFFFFFF
    if cflag != 0:
        return None                                # multi-chunk record
    flag   = struct.unpack('<I', f.read(4))[0]
    if flag > 0:
        return None                                # header/metadata record
    label  = struct.unpack('<f', f.read(4))[0]
    f.read(16)                                     # skip id (8) + id2 (8)
    jpeg   = f.read(length - 24)
    return int(label), jpeg


def select_top_n_identities(
    identity_groups: dict[int, list[int]],
    n: int,
) -> list[int]:
    """Return the N identity IDs with the most images, sorted descending."""
    sorted_ids = sorted(identity_groups, key=lambda k: len(identity_groups[k]), reverse=True)
    return sorted_ids[:n]


def split_records(
    selected_ids: list[int],
    label_map: dict[int, int],
    identity_groups: dict[int, list[int]],
    val_fraction: float,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Split images 80/20 within each identity. Returns (train_records, val_records)."""
    train_records = []
    val_records = []
    for orig_id in selected_ids:
        label = label_map[orig_id]
        rec_ids = identity_groups[orig_id].copy()
        random.shuffle(rec_ids)
        n_val = max(1, int(len(rec_ids) * val_fraction))
        for rec_id in rec_ids[:n_val]:
            val_records.append((rec_id, label))
        for rec_id in rec_ids[n_val:]:
            train_records.append((rec_id, label))
    return train_records, val_records


def write_lmdb(
    rec_path: str,
    offsets: dict[int, int],
    records: list[tuple[int, int]],
    num_classes: int,
    lmdb_path: str,
) -> None:
    """Write records to LMDB."""
    map_size = int(len(records) * 15_000 * 1.2)

    Path(lmdb_path).parent.mkdir(parents=True, exist_ok=True)
    env = lmdb.open(lmdb_path, map_size=map_size)

    skipped = 0
    written = 0
    with open(rec_path, 'rb') as rec_file:
        with env.begin(write=True) as txn:
            for rec_id, label in tqdm(records, desc=f"Writing {Path(lmdb_path).name}"):
                result = read_record(rec_file, offsets[rec_id])
                if result is None:
                    skipped += 1
                    continue
                _, jpeg = result
                if jpeg[:2] != b'\xff\xd8':
                    skipped += 1
                    continue
                key   = f"{written:09d}".encode()
                value = pickle.dumps({"jpeg": jpeg, "label": label})
                txn.put(key, value)
                written += 1

            txn.put(b"__len__",      str(written).encode())
            txn.put(b"__nclasses__", str(num_classes).encode())

    env.close()
    if skipped:
        print(f"  Skipped {skipped} invalid records")
    print(f"  {written} images → {lmdb_path}")


def lmdb_paths(cfg: dict) -> tuple[str, str]:
    """Derive train/val LMDB paths from num_identities."""
    data_dir = os.environ.get("DATA_DIR", "data")
    n = cfg["data"]["num_identities"]
    return f"{data_dir}/ms1m_{n}_train.lmdb", f"{data_dir}/ms1m_{n}_val.lmdb"


def prepare(cfg: dict) -> None:
    train_path, val_path = lmdb_paths(cfg)
    ms1m_dir   = os.environ.get("MS1M_DIR", cfg["data"]["ms1m_dir"])
    num_identities = cfg["data"]["num_identities"]
    val_fraction   = cfg["data"].get("val_fraction", 0.2)

    if os.path.exists(train_path) and os.path.exists(val_path):
        print(f"LMDBs already exist, skipping.")
        return

    rec_path = os.path.join(ms1m_dir, "train.rec")
    idx_path = os.path.join(ms1m_dir, "train.idx")
    assert os.path.exists(rec_path), f"Not found: {rec_path}"
    assert os.path.exists(idx_path), f"Not found: {idx_path}"

    print("Parsing index file...")
    offsets = parse_idx(idx_path)
    print(f"  {len(offsets)} records in index")

    print("Scanning records for identity labels...")
    identity_groups: dict[int, list[int]] = defaultdict(list)
    skipped_scan = 0
    with open(rec_path, 'rb') as rec_file:
        for rec_id, offset in tqdm(offsets.items(), desc="Scanning"):
            result = read_record(rec_file, offset)
            if result is None:
                skipped_scan += 1
                continue
            label, _ = result
            identity_groups[label].append(rec_id)

    print(f"  {len(identity_groups)} unique identities (skipped {skipped_scan} non-image records)")

    selected_ids = select_top_n_identities(identity_groups, num_identities)
    print(f"  Selected top {len(selected_ids)} identities")

    label_map = {orig: new for new, orig in enumerate(selected_ids)}
    train_records, val_records = split_records(
        selected_ids, label_map, identity_groups, val_fraction
    )
    print(f"  Split: {len(train_records)} train, {len(val_records)} val")

    write_lmdb(rec_path, offsets, train_records, len(selected_ids), train_path)
    write_lmdb(rec_path, offsets, val_records, len(selected_ids), val_path)
