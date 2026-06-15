# Reproducibility guide

This repository supports two reproducibility paths:

1. a **review-time cached-logit workflow**, which avoids rerunning image-model inference;
2. a **full image-level rerun**, which requires ImageNet-100, ImageNet-C, trained checkpoints, and substantial compute.

For manuscript review, the recommended path is the cached-logit workflow.

## Review-time cached-logit workflow

The full image-level rerun is expensive because it requires downloading ImageNet-100 and ImageNet-C, loading trained backbones, running inference over all corruption families and severities, and regenerating cached logits.

To make the paper easier to review, this repository is organized around a cached-logit workflow. This workflow starts from saved logits, stream descriptor tables, source-bank descriptor tables, source-temperature summaries, and final result tables.

Large cached logits are **not stored directly in this GitHub repository**. They will be made available through an external artifact link for review or public release.

## Required cached artifacts

Place cached artifacts under:

```text
artifacts/
├── source_temperature/
├── source_bank_descriptors/
├── target_stream_descriptors/
├── target_logits/
├── manifests/
└── results/
```

At minimum, the cached-logit workflow expects:

```text
source temperature CSV
source-bank descriptor CSV
target-stream descriptor CSV
target logits manifest
final result CSVs
```

To fully recompute target metrics such as NLL, Brier, ECE, AECE, and accuracy, the workflow also requires:

```text
cached target logits and labels
```

This is because these metrics cannot be recomputed from descriptor tables alone.

## What is included in this GitHub repository

This repository includes:

```text
source code
configuration files
documentation
notebook workflow
selected paper figures
small result/manifest files when available
```

This repository does not include:

```text
ImageNet image files
ImageNet-C image files
large cached logits
large checkpoint files
large .pt / .pth / .ckpt files
large .npz / .npy arrays
large .zip / .tar / .tar.gz archives
```

These large or restricted artifacts are intentionally excluded from GitHub.

## Recommended review path

For reviewers who do not want to rerun image-model inference:

1. Download the cached artifacts from the external artifact link.
2. Place them under the local `artifacts/` directory.
3. Check or update the paths in `configs/protocol.json`.
4. Run the cached-logit SASC script:

```bash
python scripts/run_sasc_from_cached_logits.py
```

5. Generate compact result tables:

```bash
python scripts/make_tables_from_outputs.py
```

6. Compare the regenerated outputs with the manuscript tables and figures.

## Cached-logit workflow inputs

The cached-logit workflow is based on the following artifact types:

### Source temperature summaries

These files contain the fitted clean-source global temperature values for each backbone and seed.

They are used as the lower fallback and reference temperature for SASC.

### Source-bank descriptor tables

These files contain source-bank stream descriptors and source-bank fitted target temperatures.

They are used to fit the support-aware entropy-to-temperature mapping.

### Target-stream descriptor tables

These files contain unlabeled target-stream descriptors such as entropy, confidence, margin, and free-energy statistics.

They are used to predict target temperatures without using target labels.

### Target logits and labels

These files are needed only when recomputing target evaluation metrics such as NLL, Brier, ECE, AECE, and accuracy.

The SASC temperature prediction step itself does not use target labels.

### Manifests

Manifest files record paths, backbones, seeds, corruptions, severities, and conditions.

They make the cached-logit workflow auditable and help connect outputs back to the manuscript tables.

## Full image-level rerun path

A full rerun from images requires the following steps:

1. Download ImageNet-100.
2. Download ImageNet-C.
3. Filter ImageNet-C to the same 100 ImageNet classes.
4. Prepare or load the fixed train/calibration split.
5. Load trained ResNet and ViT checkpoints.
6. Regenerate clean-source logits.
7. Regenerate ImageNet-C target logits.
8. Regenerate source-bank logits.
9. Run the ResNet and ViT SASC analysis notebooks.
10. Run the close-baseline comparison notebook.
11. Run the final diagnostics and export notebook.

The notebooks were originally run in Google Colab using Google Drive paths. Before rerunning locally or in a different environment, update the dataset, checkpoint, and artifact paths in the notebook configuration cells.

## Notebook order

The recommended notebook order is:

```text
01_vit_training_logits_and_source_bank.ipynb
02_vit_sasc_analysis_and_diagnostics.ipynb
03_resnet_baseline_protocol_rerun.ipynb
04_resnet_sasc_analysis_and_diagnostics.ipynb
05_close_baselines_and_method_comparison.ipynb
06_extra_diagnostics_and_final_exports.ipynb
```

## Important reproducibility note

SASC/Frozen V3 is fitted using source-bank information and unlabeled target-stream descriptors.

Target labels are not used for fitting SASC temperatures. Target labels are used only for downstream evaluation metrics and diagnostics.

The final SASC configuration is fixed in `configs/frozen_v3_sasc.json`.

