#!/usr/bin/env python
"""Run clean and controlled synthetic image-quality stress tests on the frozen test IDs."""
import argparse, json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import h5py
from galaxylens.labels import CLASS_NAMES
from galaxylens.model import choose_device, load_checkpoint
from galaxylens.transforms import build_transform, apply_corruption
from galaxylens.metrics import summarize_predictions, write_json

class H5CorruptedDataset(Dataset):
    def __init__(self,path,indices,kind,severity,seed,transform):
        self.path=path;self.indices=indices;self.kind=kind;self.severity=severity;self.seed=seed;self.transform=transform
        with h5py.File(path,"r") as h5: self.labels=np.asarray(h5["ans"]).reshape(-1).astype(int)[indices]
        self.h5=None
    def __len__(self): return len(self.indices)
    def __getitem__(self,i):
        if self.h5 is None: self.h5=h5py.File(self.path,"r")
        idx=int(self.indices[i]); image=Image.fromarray(np.asarray(self.h5["images"][idx]).astype(np.uint8)).convert("RGB")
        image=apply_corruption(image,self.kind,self.severity,seed=self.seed+idx)
        return self.transform(image),int(self.labels[i])

p=argparse.ArgumentParser()
p.add_argument("--data",default="data/raw/Galaxy10_DECals.h5")
p.add_argument("--splits",default="data/splits/split_seed42.json")
p.add_argument("--checkpoint",default="artifacts/checkpoints/baseline_best.pt")
p.add_argument("--out",default="reports/baseline/robustness_metrics.json")
p.add_argument("--batch-size",type=int,default=32);p.add_argument("--workers",type=int,default=2)
p.add_argument("--device",default="auto");p.add_argument("--seed",type=int,default=2026)
args=p.parse_args()
device=choose_device(args.device); split=json.loads(Path(args.splits).read_text(encoding="utf-8"))
model,ckpt=load_checkpoint(args.checkpoint,device); results=[]
conditions=[("clean",0)]+[(k,s) for k in ("blur","resolution","noise","brightness") for s in (1,2,3)]
for kind,severity in conditions:
    ds=H5CorruptedDataset(args.data,split["indices"]["test"],kind,severity,args.seed,build_transform(training=False))
    loader=DataLoader(ds,batch_size=args.batch_size,shuffle=False,num_workers=args.workers)
    ys=[];ps=[]
    with torch.no_grad():
        for x,y in loader:
            logits=model(x.to(device));ys.extend(y.tolist());ps.extend(logits.argmax(1).cpu().tolist())
    metrics=summarize_predictions(ys,ps,CLASS_NAMES)
    results.append({"condition":kind,"severity":severity,"n_samples":len(ys),
                    "accuracy":metrics["accuracy"],"macro_f1":metrics["macro_f1"],
                    "per_class":{name:metrics["classification_report"][name] for name in CLASS_NAMES},
                    "confusion_matrix":metrics["confusion_matrix"]})
    print(f"{kind:10s} severity={severity} accuracy={metrics['accuracy']:.4f} macro_f1={metrics['macro_f1']:.4f}")
write_json(args.out,{"checkpoint":args.checkpoint,"split":"test","seed":args.seed,
                     "conditions":results,
                     "note":"Synthetic stress tests only; results do not establish cross-survey generalization."})
print(f"Saved robustness report: {args.out}")
