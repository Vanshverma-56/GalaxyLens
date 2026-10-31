#!/usr/bin/env python
"""Train the residual multi-scale ResNet-18 GalaxyLens experiment."""

import argparse
import json
import random
import time
import platform
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from tqdm import tqdm
import torchvision

from galaxylens.data import Galaxy10H5Dataset
from galaxylens.labels import CLASS_NAMES
from galaxylens.residual_multiscale_model import build_residual_multiscale_model
from galaxylens.model import choose_device
from galaxylens.transforms import build_transform


p = argparse.ArgumentParser()

p.add_argument("--data", default="data/raw/Galaxy10_DECals.h5")
p.add_argument("--splits", default="data/splits/split_seed42.json")
p.add_argument("--out", default="artifacts/checkpoints/residual_multiscale_best.pt")
p.add_argument("--reports", default="reports/experiments/residual_multiscale")
p.add_argument("--epochs", type=int, default=12)
p.add_argument("--batch-size", type=int, default=32)
p.add_argument("--workers", type=int, default=2)
p.add_argument("--lr", type=float, default=3e-4)
p.add_argument("--weight-decay", type=float, default=1e-4)
p.add_argument("--seed", type=int, default=42)
p.add_argument("--device", default="auto")
p.add_argument("--patience", type=int, default=4)

args = p.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

device = choose_device(args.device)

print("Experiment: Residual Multi-Scale Fusion ResNet-18")
print("Device:", device)
print("Seed:", args.seed)

splits = json.loads(
    Path(args.splits).read_text(encoding="utf-8")
)

train_ds = Galaxy10H5Dataset(
    args.data,
    splits["indices"]["train"],
    build_transform(training=True)
)

val_ds = Galaxy10H5Dataset(
    args.data,
    splits["indices"]["validation"],
    build_transform(training=False)
)

generator = torch.Generator().manual_seed(args.seed)

train_loader = DataLoader(
    train_ds,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=args.workers,
    pin_memory=(device.type == "cuda"),
    generator=generator
)

val_loader = DataLoader(
    val_ds,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=args.workers,
    pin_memory=(device.type == "cuda")
)

model = build_residual_multiscale_model(pretrained=True).to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=args.lr,
    weight_decay=args.weight_decay
)

out_path = Path(args.out)
out_path.parent.mkdir(parents=True, exist_ok=True)

report_dir = Path(args.reports)
report_dir.mkdir(parents=True, exist_ok=True)

best_f1 = -1.0
best_epoch = 0
stale = 0
history = []

start = time.time()


def evaluate(loader):
    model.eval()

    truths = []
    predictions = []
    loss_sum = 0.0
    count = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = criterion(logits, y)

            loss_sum += loss.item() * len(y)
            count += len(y)

            truths.extend(y.cpu().tolist())
            predictions.extend(logits.argmax(1).cpu().tolist())

    return (
        loss_sum / max(count, 1),
        float(f1_score(
            truths,
            predictions,
            average="macro",
            zero_division=0
        ))
    )


for epoch in range(1, args.epochs + 1):
    model.train()

    running = 0.0
    seen = 0

    loop = tqdm(
        train_loader,
        desc=f"Epoch {epoch}/{args.epochs}"
    )

    for x, y in loop:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        logits = model(x)
        loss = criterion(logits, y)

        loss.backward()
        optimizer.step()

        running += loss.item() * len(y)
        seen += len(y)

        loop.set_postfix(
            train_loss=f"{running / max(seen, 1):.4f}"
        )

    val_loss, val_f1 = evaluate(val_loader)

    row = {
        "epoch": epoch,
        "train_loss": running / max(seen, 1),
        "val_loss": val_loss,
        "val_macro_f1": val_f1
    }

    history.append(row)
    print(row)

    if val_f1 > best_f1:
        best_f1 = val_f1
        best_epoch = epoch
        stale = 0

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "class_names": list(CLASS_NAMES),
                "num_classes": len(CLASS_NAMES),
                "architecture": "residual_multiscale_resnet18",
                "image_size": 224,
                "normalization": {
                    "mean": [0.485, 0.456, 0.406],
                    "std": [0.229, 0.224, 0.225]
                },
                "training_config": vars(args),
                "best_epoch": epoch,
                "best_validation_macro_f1": best_f1,
                "split_manifest": args.splits,
                "dataset_path": args.data,
                "environment": {
                    "python": platform.python_version(),
                    "torch": torch.__version__,
                    "torchvision": torchvision.__version__
                }
            },
            out_path
        )

        print("Saved new best checkpoint:", out_path)

    else:
        stale += 1

    if stale >= args.patience:
        print(
            f"Early stopping after {args.patience} "
            "epochs without validation macro-F1 improvement."
        )
        break


(report_dir / "history.json").write_text(
    json.dumps(history, indent=2),
    encoding="utf-8"
)

summary = {
    "experiment": "Experiment 7 - Residual Multi-Scale Fusion",
    "checkpoint": str(out_path),
    "best_epoch": best_epoch,
    "best_validation_macro_f1": best_f1,
    "epochs_completed": len(history),
    "elapsed_seconds": round(time.time() - start, 2),
    "device": str(device),
    "seed": args.seed,
    "test_set_used_for_selection": False,
    "history_path": str(report_dir / "history.json")
}

(report_dir / "training_summary.json").write_text(
    json.dumps(summary, indent=2),
    encoding="utf-8"
)

print("\nTRAINING SUMMARY")
print(json.dumps(summary, indent=2))
