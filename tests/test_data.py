import json
import h5py
import numpy as np
from PIL import Image
from galaxylens.data import inspect_hdf5, Galaxy10H5Dataset

def make_tiny_h5(path):
    images = np.zeros((20, 16, 16, 3), dtype=np.uint8)
    for i in range(20): images[i] = i
    labels = np.array([i % 10 for i in range(20)], dtype=np.uint8)
    with h5py.File(path, "w") as h5:
        h5.create_dataset("images", data=images)
        h5.create_dataset("ans", data=labels)
        h5.create_dataset("ra", data=np.zeros(20))
    return images, labels

def make_cache(path, images, labels):
    cache = path.parent.parent / "cache"
    cache.mkdir()
    np.save(cache / "galaxy10_images.npy", images)
    np.save(cache / "galaxy10_labels.npy", labels)
    (cache / "cache_complete.json").write_text(json.dumps({"complete": True, "num_images": len(labels)}))

def test_inspector_reports_schema_and_counts(tmp_path):
    path = tmp_path / "tiny.h5"; make_tiny_h5(path)
    info = inspect_hdf5(path)
    assert info["n_samples"] == 20
    assert info["images_shape"] == [20, 16, 16, 3]
    assert info["labels_valid_0_to_9"] is True
    assert info["label_counts"]["0"] == 2

def test_cache_dataset_returns_image_and_original_label(tmp_path):
    path = tmp_path / "raw" / "tiny.h5"; path.parent.mkdir()
    images, labels = make_tiny_h5(path); make_cache(path, images, labels)
    ds = Galaxy10H5Dataset(path, [3, 11])
    image, label = ds[0]
    assert isinstance(image, Image.Image)
    assert image.size == (16, 16)
    assert label == int(labels[3])
    assert image.getpixel((0, 0)) == (3, 3, 3)
    ds.close()
