#!/usr/bin/env python
import argparse, json
from pathlib import Path
from galaxylens.data import inspect_hdf5

parser = argparse.ArgumentParser(description="Inspect Galaxy10 DECaLS HDF5 structure and labels.")
parser.add_argument("--data", default="data/raw/Galaxy10_DECals.h5")
parser.add_argument("--out", default="reports/dataset_inspection.json")
args = parser.parse_args()
result = inspect_hdf5(args.data)
print(json.dumps(result, indent=2))
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
print(f"\nSaved inspection report: {args.out}")
