# Reviewer checklist

This checklist summarizes what a reviewer can inspect in this repository without downloading the full image datasets.

## 1. Repository structure

Check that the repository contains:

```text
configs/
data/
docs/
figures/
manifests/
notebooks/
results/
scripts/
src/sasc/
tests/
```

The repository is designed to contain code, configurations, documentation, cleaned notebooks, small result tables, selected figures, and manifests.

Large datasets, cached logits, checkpoints, and generated arrays are intentionally excluded from GitHub.

## 2. Main configuration files

Check the frozen protocol files:

```text
configs/frozen_v3_sasc.json
configs/protocol.json
```

The final SASC/Frozen V3 configuration should specify:

```text
8 entropy bins
0.25 standard-deviation support margin
absolute temperature cap 1.45
additive temperature cap 0.35
maximum extrapolation slope 0.75
lower fallback enabled
source-bank-only fitting
no target labels used for fitting
no target metrics used for hyperparameter selection
```

## 3. Label-use rule

The main label-use rule is:

```text
SASC/Frozen V3 does not use target labels for fitting.
```

Target labels are used only for:

```text
evaluation metrics
target-oracle diagnostics
post-hoc analysis tables
```

This distinction is important for the information-regime claims in the paper.

## 4. Review-time cached-logit workflow

The recommended review-time workflow does not rerun image-model inference.

At minimum, the cached-logit workflow requires:

```text
source temperature CSV
source-bank descriptor CSV
target-stream descriptor CSV
target logits manifest
cached target logits and labels
final result CSVs
```

Large cached artifacts should be provided through an external artifact link, not committed directly to GitHub.

Expected local layout:

```text
artifacts/
├── source_temperature/
├── source_bank_descriptors/
├── target_stream_descriptors/
├── target_logits/
├── manifests/
└── results/
```

Then run:

```bash
python scripts/run_sasc_from_cached_logits.py
python scripts/make_tables_from_outputs.py
```

## 5. Notebook order

The final cleaned notebooks should be reviewed in this order:

```text
01_vit_training_logits_and_source_bank.ipynb
02_vit_sasc_analysis_and_diagnostics.ipynb
03_resnet_baseline_protocol_rerun.ipynb
04_resnet_sasc_analysis_and_diagnostics.ipynb
05_close_baselines_and_method_comparison.ipynb
06_extra_diagnostics_and_final_exports.ipynb
```

Notebook 1 and notebook 3 are the heaviest because they start from image data and checkpoints.

Notebook 2 and notebook 4 are the main SASC analysis notebooks.

Notebook 5 and notebook 6 depend on already generated CSVs, manifests, and cached-logit artifacts.

## 6. Tests

After installing the repository requirements, run:

```bash
pytest
```

The tests check core SASC behavior, including positive temperatures, support-region handling, and safety constraints.

## 7. Data and artifact policy

Do not expect the following files to be stored in GitHub:

```text
ImageNet image folders
ImageNet-C image folders
cached logits
source-bank logits
model checkpoints
large .pt / .pth / .ckpt files
large .npz / .npy files
large .zip / .tar / .tar.gz files
```

These artifacts should be provided separately through an external artifact link.

## 8. What to verify in the paper artifact

A reviewer can check the following:

```text
The frozen SASC configuration is fixed before final evaluation.
The source-bank fitting uses source-side degraded streams only.
Target-stream descriptors are unlabeled.
Target labels are not used for deployable temperature fitting.
Target-oracle temperatures are diagnostic only.
SASC is presented as a conservative support-aware protocol, not as a universal best average method.
Close baselines and ablations are separated from the frozen final method.
```

## 9. Expected final release step

Before formal journal submission, create a GitHub release such as:

```text
v1.0-paper-submission
```

If possible, archive the release on Zenodo and cite the DOI in the manuscript and `CITATION.cff`.

## 10. Final reviewer note

This repository is intended to support the SASC paper by making the code, configuration, cleaned notebooks, and small outputs auditable.

The full image-level rerun is possible but expensive. The recommended review-time path is the cached-logit workflow.
