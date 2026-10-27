#!/usr/bin/env python
"""Calibrate/selective evaluation: choose confidence cutoffs on validation, apply on test."""
import argparse, json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from galaxylens.data import Galaxy10H5Dataset
from galaxylens.model import choose_device, load_checkpoint
from galaxylens.transforms import build_transform
from galaxylens.uncertainty import (
    expected_calibration_error, brier_score_multiclass,
    threshold_for_coverage, selective_metrics
)
from galaxylens.metrics import write_json

p=argparse.ArgumentParser()
p.add_argument("--data",default="data/raw/Galaxy10_DECals.h5")
p.add_argument("--splits",default="data/splits/split_seed42.json")
p.add_argument("--checkpoint",default="artifacts/checkpoints/baseline_best.pt")
p.add_argument("--out",default="reports/baseline/selective_metrics.json")
p.add_argument("--batch-size",type=int,default=32)
p.add_argument("--workers",type=int,default=2)
p.add_argument("--device",default="auto")
p.add_argument("--coverages",type=float,nargs="+",default=[0.95,0.90,0.80,0.70,0.50])
args=p.parse_args()
device=choose_device(args.device)
splits=json.loads(Path(args.splits).read_text(encoding="utf-8"))
model,checkpoint=load_checkpoint(args.checkpoint,device)

def predict(indices):
    ds=Galaxy10H5Dataset(args.data,indices,build_transform(training=False))
    loader=DataLoader(ds,batch_size=args.batch_size,shuffle=False,num_workers=args.workers)
    ys=[];probs=[]
    with torch.no_grad():
        for x,y in loader:
            p=model(x.to(device)).softmax(1).cpu().numpy()
            ys.extend(y.numpy().tolist());probs.extend(p.tolist())
    return np.asarray(ys),np.asarray(probs)

y_val,p_val=predict(splits["indices"]["validation"])
y_test,p_test=predict(splits["indices"]["test"])
conf_val=p_val.max(axis=1)
conf_test=p_test.max(axis=1)
pred_test=p_test.argmax(axis=1)
rows=[]
for coverage in args.coverages:
    threshold=threshold_for_coverage(conf_val,coverage)
    row=selective_metrics(y_test,pred_test,conf_test,threshold)
    row["target_coverage_from_validation"]=coverage
    rows.append(row)
result={
    "checkpoint":args.checkpoint,
    "threshold_selection_split":"validation",
    "evaluation_split":"test",
    "validation_n":int(len(y_val)),
    "test_n":int(len(y_test)),
    "validation_ece":expected_calibration_error(y_val,p_val),
    "test_ece":expected_calibration_error(y_test,p_test),
    "validation_multiclass_brier":brier_score_multiclass(y_val,p_val),
    "test_multiclass_brier":brier_score_multiclass(y_test,p_test),
    "selective_results":rows,
    "interpretation":"ECE and Brier score are calibration diagnostics, not proof of calibrated probabilities. Thresholds were selected using validation confidence only; the test set was used only for final evaluation."
}
write_json(args.out,result)
print(json.dumps(result,indent=2))
