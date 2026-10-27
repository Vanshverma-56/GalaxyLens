#!/usr/bin/env python
"""Evaluate a selected checkpoint on the held-out test split exactly once per frozen protocol."""
import argparse, json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from galaxylens.data import Galaxy10H5Dataset
from galaxylens.labels import CLASS_NAMES
from galaxylens.model import choose_device, load_checkpoint
from galaxylens.transforms import build_transform
from galaxylens.metrics import summarize_predictions, write_json

p=argparse.ArgumentParser()
p.add_argument("--data",default="data/raw/Galaxy10_DECals.h5")
p.add_argument("--splits",default="data/splits/split_seed42.json")
p.add_argument("--checkpoint",default="artifacts/checkpoints/baseline_best.pt")
p.add_argument("--out",default="reports/baseline/test_metrics.json")
p.add_argument("--batch-size",type=int,default=32)
p.add_argument("--workers",type=int,default=2)
p.add_argument("--device",default="auto")
args=p.parse_args()
device=choose_device(args.device)
splits=json.loads(Path(args.splits).read_text(encoding="utf-8"))
ds=Galaxy10H5Dataset(args.data,splits["indices"]["test"],build_transform(training=False))
loader=DataLoader(ds,batch_size=args.batch_size,shuffle=False,num_workers=args.workers)
model,checkpoint=load_checkpoint(args.checkpoint,device)
truths=[];preds=[];confidences=[]
with torch.no_grad():
    for x,y in loader:
        logits=model(x.to(device))
        probs=logits.softmax(1)
        truths.extend(y.tolist());preds.extend(probs.argmax(1).cpu().tolist())
        confidences.extend(probs.max(1).values.cpu().tolist())
result=summarize_predictions(truths,preds,CLASS_NAMES)
result.update({"split":"test","checkpoint":args.checkpoint,"split_manifest":args.splits,
               "n_samples":len(truths),"mean_max_softmax_confidence":float(sum(confidences)/max(1,len(confidences))),
               "confidence_note":"Maximum softmax is a confidence score, not a calibrated probability."})
write_json(args.out,result)
print(json.dumps({"accuracy":result["accuracy"],"macro_f1":result["macro_f1"],"n_samples":len(truths),"out":args.out},indent=2))
