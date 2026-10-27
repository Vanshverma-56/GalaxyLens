#!/usr/bin/env python
"""Build a resumable-safe NumPy cache from the verified Galaxy10 HDF5 source."""
from pathlib import Path
import argparse, hashlib, json, time
import h5py
import numpy as np

EXPECTED_SHA256 = "19AEFC477C41BB7F77FF07599A6B82A038DC042F889A111B0D4D98BB755C1571"

def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/raw/Galaxy10_DECals.h5")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    data = Path(args.data).resolve()
    if not data.exists(): raise FileNotFoundError(data)
    actual = sha256(data)
    print("SHA-256:", actual)
    if actual != EXPECTED_SHA256: raise RuntimeError("Dataset checksum mismatch; refusing to cache.")
    cache = data.parent.parent / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    image_tmp, label_tmp = cache / "galaxy10_images.npy.tmp", cache / "galaxy10_labels.npy.tmp"
    image_out, label_out = cache / "galaxy10_images.npy", cache / "galaxy10_labels.npy"
    meta_path = cache / "cache_complete.json"
    meta_path.unlink(missing_ok=True); image_tmp.unlink(missing_ok=True); label_tmp.unlink(missing_ok=True)
    start=time.time()
    with h5py.File(data, "r") as h5:
        if "images" not in h5 or "ans" not in h5: raise ValueError(f"Unexpected HDF5 keys: {list(h5.keys())}")
        images, labels = h5["images"], h5["ans"]
        if len(images) != len(labels): raise ValueError("Image/label counts differ.")
        n=len(images)
        image_map=np.lib.format.open_memmap(image_tmp, mode="w+", dtype=images.dtype, shape=images.shape)
        label_map=np.lib.format.open_memmap(label_tmp, mode="w+", dtype=labels.dtype, shape=labels.shape)
        for start_i in range(0,n,args.batch_size):
            end_i=min(start_i+args.batch_size,n)
            image_map[start_i:end_i]=images[start_i:end_i]
            label_map[start_i:end_i]=labels[start_i:end_i]
            if end_i == n or end_i % 2048 == 0: print(f"Cached {end_i:,}/{n:,}")
        image_map.flush(); label_map.flush(); del image_map, label_map
        image_shape=list(images.shape[1:])
    image_tmp.replace(image_out); label_tmp.replace(label_out)
    meta={"complete":True,"num_images":int(n),"image_shape":image_shape,"source_sha256":actual,"cache_format":"numpy_memmap_v1"}
    meta_path.write_text(json.dumps(meta,indent=2))
    print(f"Cache complete in {(time.time()-start)/60:.1f} minutes: {cache}")
if __name__ == "__main__": main()
