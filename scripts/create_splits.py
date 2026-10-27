#!/usr/bin/env python
"""Create stable stratified 80/10/10 row-index splits and check disjointness."""
import argparse, json
from pathlib import Path
import h5py
import numpy as np
from sklearn.model_selection import train_test_split
from galaxylens.data import inspect_hdf5

parser = argparse.ArgumentParser()
parser.add_argument("--data", default="data/raw/Galaxy10_DECals.h5")
parser.add_argument("--out", default="data/splits/split_seed42.json")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--train-fraction", type=float, default=0.8)
parser.add_argument("--val-fraction", type=float, default=0.1)
args = parser.parse_args()
if not (0 < args.train_fraction < 1 and 0 < args.val_fraction < 1
        and args.train_fraction + args.val_fraction < 1):
    raise ValueError("Fractions must be positive and train + validation must be < 1.")
info = inspect_hdf5(args.data)
with h5py.File(args.data, "r") as h5:
    labels = np.asarray(h5["ans"]).reshape(-1).astype(int)
indices = np.arange(len(labels))
test_fraction = 1.0 - args.train_fraction - args.val_fraction
train_idx, remaining_idx = train_test_split(
    indices, test_size=(1.0 - args.train_fraction), random_state=args.seed,
    stratify=labels
)
remaining_labels = labels[remaining_idx]
relative_test_fraction = test_fraction / (args.val_fraction + test_fraction)
val_idx, test_idx = train_test_split(
    remaining_idx, test_size=relative_test_fraction, random_state=args.seed,
    stratify=remaining_labels
)
all_sets = [set(map(int, train_idx)), set(map(int, val_idx)), set(map(int, test_idx))]
assert not (all_sets[0] & all_sets[1] or all_sets[0] & all_sets[2] or all_sets[1] & all_sets[2])
assert len(set.union(*all_sets)) == len(labels)
payload = {
    "dataset_path_at_creation": str(Path(args.data)),
    "dataset_n_samples": int(len(labels)),
    "seed": args.seed,
    "fractions_requested": {"train": args.train_fraction, "validation": args.val_fraction, "test": test_fraction},
    "split_sizes": {"train": len(train_idx), "validation": len(val_idx), "test": len(test_idx)},
    "label_counts": info["label_counts"],
    "indices": {"train": train_idx.tolist(), "validation": val_idx.tolist(), "test": test_idx.tolist()},
    "leakage_checks": {"pairwise_disjoint": True, "covers_all_rows": True},
    "note": "Indices are row IDs in the verified HDF5 file. This split prevents row overlap; duplicate astrophysical objects require a separate ID/catalog audit if metadata supports it."
}
out = Path(args.out)
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
print(json.dumps({k:v for k,v in payload.items() if k != "indices"}, indent=2))
print(f"Saved split manifest: {out}")
