# GalaxyLens-Robust — Live Handoff (updated 24 October 2026)

## What has actually been completed

1. Recovered the project source from `GalaxyLens_Robust_Implementation_Stage1.zip` after the Colab runtime reset.
2. Downloaded `Galaxy10_DECals.h5` into `/content/galaxylens/data/raw/` using resumable `wget`.
3. Verified SHA-256: `19AEFC477C41BB7F77FF07599A6B82A038DC042F889A111B0D4D98BB755C1571`.
4. Confirmed HDF5 keys include `ans`, `dec`, `images`, `pxscale`, `ra`, `redshift`; image array shape `(17736, 256, 256, 3)`, dtype `uint8`; label array shape `(17736,)`, dtype `uint8`.
5. Built `data/cache/galaxy10_images.npy`, `galaxy10_labels.npy`, and `cache_complete.json` in about 2.3 minutes.
6. Replaced the loader in the active Colab project with a NumPy memmap cache-backed loader and saved `data_before_cache_loader.py` backup.
7. Smoke-tested four real images through ResNet-18 on an NVIDIA Tesla T4 (`cuda`): input `(4,3,224,224)`, output `(4,10)`, finite loss `2.4121`, backward pass succeeded.

This smoke test is evidence that basic data loading and model execution work. It is not evidence of trained model quality. Four labels were all class 0; validate split and class distribution before training.

## Immediate next steps (in order)

1. **Persist files before runtime reset.** Copy the project, HDF5 dataset, cache, split manifest if present, and any outputs to Google Drive or another durable location. Dataset + cache need several GB.
2. **Sync the updated source.** This ZIP contains a cache-backed `src/galaxylens/data.py` and reusable `scripts/build_cache.py`. Copy the updated source/docs into the active Colab project or use this ZIP as the new canonical project folder. Do not blindly overwrite an existing split manifest.
3. Run `python scripts/inspect_dataset.py` and check label counts.
4. Check `data/splits/split_seed42.json`. If absent, run `python scripts/create_splits.py`; if present, inspect it and confirm it is disjoint and stratified before reusing it.
5. Run `pytest -q`. Fix failures before proceeding. This package has not yet been claimed as having a verified post-change test-suite pass.
6. Run a short training pilot (e.g., a few batches / one small epoch) with model and batches on CUDA. Verify finite loss, gradients, checkpoint save/load and validation metric calculation.
7. Only after pilot passes, run baseline training and freeze its checkpoint. Then run degradation robustness evaluation, quality-augmentation comparison, calibration diagnostics, and selective prediction.
8. Update this handoff, README and handbook with actual test/pilot outcomes; repackage them together.

## Runtime / paths

Current Colab path during successful smoke test: `/content/galaxylens`. Dataset: `data/raw/Galaxy10_DECals.h5`. Cache: `data/cache/`. Colab `/content` is ephemeral; runtime reset deletes it. GPU: NVIDIA Tesla T4; CUDA available.

## Research question and constraints

How does image quality affect galaxy-classifier predictions, and can training with realistic image-quality variation plus uncertainty-based rejection reduce unreliable classifications? Preserve original labels 0–9. Use a fixed split for all comparisons. Validation data alone selects checkpoints and rejection thresholds. Synthetic degradations do not establish cross-survey generalization. No full training results or improvement claims exist yet.

---

## Brightness Robustness Correction — 2026-10-27

The brightness corruption mapping in `src/galaxylens/transforms.py`
was corrected to use factors `(0.8, 0.6, 0.4)` for severities 1, 2, and 3.
Previously, severity 3 used factor `1.25`, which brightened images rather
than strongly darkening them.

Corrected brightness results on the frozen test split:

| Severity | Brightness factor | Accuracy | Macro-F1 |
|---|---:|---:|---:|
| 1 — mild darkening | 0.8 | 0.7813 | 0.7762 |
| 2 — moderate darkening | 0.6 | 0.7097 | 0.7039 |
| 3 — strong darkening | 0.4 | 0.5620 | 0.5657 |

The completed corrected evaluation results were recovered from the prior
notebook output; inference was not rerun during the report repair. The
aggregate report was repaired by replacing only the invalid severity-3
entry. The original report was backed up before modification.

Relevant files:
- `reports/baseline/robustness_metrics.json` — repaired aggregate report.
- `reports/baseline/robustness_metrics_brightness_corrected.json` — corrected
  brightness summary.
- `reports/baseline/brightness_severity3_historical_brightening.json` —
  preserved historical severity-3 brightening result.
- `reports/baseline/robustness_metrics.pre_brightness_repair_20261027_182902.json`
  — pre-repair aggregate report backup.
- `src/galaxylens/transforms.pre_brightness_fix_20261027_182418.py` —
  pre-correction transform backup.

**Known limitation:** The corrected severity-3 accuracy and macro-F1 are
available, but the corrected per-class metrics and confusion matrix were
not recovered in the restored project. Do not substitute the historical
brightening details for corrected darkening details.

The existing baseline checkpoint and frozen test split were not changed.
This synthetic stress test does not establish cross-survey generalization.
Do not claim a model improvement without a valid comparison experiment.
