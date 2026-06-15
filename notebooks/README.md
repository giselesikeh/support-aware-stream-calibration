# Final notebook index

This folder contains the final cleaned notebooks used for the paper artifact.

The notebooks are organized in execution order. Some notebooks require large external artifacts such as ImageNet-100, ImageNet-C, cached logits, source-bank logits, checkpoints, and previously exported CSV tables. These large artifacts are not stored directly in this GitHub repository.

## Recommended final names

| Notebook                                         | Role                                                                                                                                                |
| ------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| `01_vit_training_logits_and_source_bank.ipynb`   | ViT-B/16 training, clean/target logits, source TS, target-oracle diagnostics, and source-bank logits.                                               |
| `02_vit_sasc_analysis_and_diagnostics.ipynb`     | ViT-B/16 SASC/Frozen V3 analysis, Safe V2, support diagnostics, LOCO/LOSO, ablations, and bootstrap tables.                                         |
| `03_resnet_baseline_protocol_rerun.ipynb`        | Earlier repaired ResNet-18 baseline protocol: baseline, TS, TS-P, PseudoCal, MM-NLL-TS, regime-aware audit.                                         |
| `04_resnet_sasc_analysis_and_diagnostics.ipynb`  | Main ResNet-18/ResNet-50 SASC/Frozen V3 analysis and final ResNet deliverables.                                                                     |
| `05_close_baselines_and_method_comparison.ipynb` | Close-baseline comparison, including UTS-style, SBTS-style/source-bank mapping, entropy-mean, Safe V2, and Frozen V3 comparison tables.             |
| `06_extra_diagnostics_and_final_exports.ipynb`   | Final supervisor-feedback diagnostics: paired bootstrap, support decomposition, source NLL curves, entropy confound diagnostics, and final exports. |

## Important artifact note

The notebooks were originally executed in Google Colab using Google Drive paths.

Before rerunning them, update the root paths inside the first setup cells to match your own environment.

Large datasets, cached logits, checkpoints, and generated intermediate arrays are intentionally excluded from this GitHub repository.

## Required files before running each notebook

### `01_vit_training_logits_and_source_bank.ipynb`

This notebook requires:

```text
ImageNet-100 dataset
ImageNet-C dataset
ImageNet-100 class mapping or class folders
Google Drive or local output directory
ViT-B/16 training configuration
```

Expected large inputs include:

```text
ImageNet-100 train/val folders
ImageNet-C corruption folders
noise.tar
blur.tar
weather.tar
```

This notebook generates:

```text
ViT-B/16 trained checkpoints
clean source logits
target ImageNet-C logits
source temperature summaries
target-oracle diagnostic tables
source-bank logits
source-bank manifests
```

### `02_vit_sasc_analysis_and_diagnostics.ipynb`

This notebook requires outputs from notebook 1:

```text
ViT clean source logits
ViT target ImageNet-C cached logits
ViT source-bank logits
ViT source temperature CSV
ViT source-bank descriptor CSV
ViT target-stream descriptor CSV
ViT cached target logits manifest
```

It generates:

```text
ViT Safe V2 outputs
ViT Frozen V3/SASC outputs
ViT support diagnostics
ViT LOCO/LOSO tables
ViT ablation tables
ViT bootstrap confidence intervals
ViT final export tables
```

### `03_resnet_baseline_protocol_rerun.ipynb`

This notebook requires:

```text
ImageNet-100 dataset
ImageNet-C dataset
saved ImageNet-100 train/calibration split files
ResNet-18 checkpoints for seeds 1, 2, and 3
```

Expected checkpoint rule:

```text
best_by_nll.pt
```

Expected split files:

```text
imagenet100_train_idx.json
imagenet100_calib_idx.json
imagenet100_split_meta.json
```

It generates:

```text
ResNet-18 baseline tables
TS tables
TS-P tables
PseudoCal tables
MM-NLL-TS tables
regime-specific comparison tables
reliability diagrams
confidence histograms
leakage/fairness audit tables
```

### `04_resnet_sasc_analysis_and_diagnostics.ipynb`

This notebook requires:

```text
ResNet-18 cached target logits
ResNet-50 cached target logits
ResNet-18 source-bank logits
ResNet-50 source-bank logits
clean/source calibration logits
source temperature summaries
target logits manifests
```

Expected cached-logit structure includes:

```text
target logits for 3 seeds
10 corruption families
5 severity levels
ResNet-18 and ResNet-50
```

It generates:

```text
ResNet source TS refit table
ResNet source-bank descriptor table
ResNet target-stream descriptor table
Safe V2 outputs
Frozen V3/SASC outputs
support/extrapolation diagnostics
LOCO/LOSO tables
V3 ablation tables
bootstrap confidence intervals
final ResNet export tables
```

### `05_close_baselines_and_method_comparison.ipynb`

This notebook requires outputs from notebooks 2 and 4:

```text
combined source-bank descriptor table
combined target-stream descriptor table
combined target logits manifest
source TS condition-level tables
Safe V2 condition-level tables
Frozen V3/SASC condition-level tables
ResNet and ViT final export tables
```

It may also require cached logits if close-baseline metrics are recomputed.

It generates:

```text
close-baseline feasibility table
UTS-style baseline tables
SBTS-style/source-bank mapping tables
entropy-mean baseline tables
close-baseline delta summaries
close-baseline bootstrap confidence intervals
compact comparison against Safe V2 and Frozen V3
```

### `06_extra_diagnostics_and_final_exports.ipynb`

This notebook requires outputs from notebooks 2, 4, and 5:

```text
ResNet final deliverables
ViT final deliverables
close-baseline tables
combined source-bank descriptor table
combined target-stream descriptor table
combined target logits manifest
source temperature CSVs
source calibration logits
Frozen V3 support-region tables
condition-level method metric tables
```

It generates:

```text
paired bootstrap tables
Frozen V3 vs Source TS comparison
Frozen V3 vs entropy-mean comparison
Frozen V3 vs Safe V2 comparison
support-region decomposition tables
source-temperature NLL curves
source-temperature flatness summaries
entropy confound regression diagnostics
entropy partial correlation diagnostics
final artifact manifests
supervisor-ready export checklists
```

## Minimal review-time cached-logit workflow

For reviewers who do not want to rerun image-model inference, the recommended workflow starts after logits have already been generated.

At minimum, the cached-logit workflow needs:

```text
source temperature CSV
source-bank descriptor CSV
target-stream descriptor CSV
target logits manifest
cached target logits and labels
final result CSVs
```

The expected local artifact layout is:

```text
artifacts/
├── source_temperature/
├── source_bank_descriptors/
├── target_stream_descriptors/
├── target_logits/
├── manifests/
└── results/
```

Then reviewers can run:

```bash
python scripts/run_sasc_from_cached_logits.py
python scripts/make_tables_from_outputs.py
```

## Full rerun order

A full rerun should follow this order:

```text
01_vit_training_logits_and_source_bank.ipynb
02_vit_sasc_analysis_and_diagnostics.ipynb
03_resnet_baseline_protocol_rerun.ipynb
04_resnet_sasc_analysis_and_diagnostics.ipynb
05_close_baselines_and_method_comparison.ipynb
06_extra_diagnostics_and_final_exports.ipynb
```

## Practical rerun note

Notebook 1 and notebook 3 are the heaviest because they start from images and model checkpoints.

Notebook 2 and notebook 4 are the main SASC analysis notebooks.

Notebook 5 and notebook 6 depend on already generated CSVs, manifests, and cached-logit artifacts.

For review, it is usually enough to provide cached logits and final CSV tables rather than requiring a full image-level rerun.

## Files that should not be committed to GitHub

Do not commit:

```text
ImageNet image folders
ImageNet-C image folders
cached logits
source-bank logits
model checkpoints
large .pt files
large .pth files
large .ckpt files
large .npz files
large .npy files
large .tar files
large .zip files
```

These artifacts should be provided through an external artifact link such as Zenodo, Google Drive, Hugging Face Dataset, or another approved storage location.

## Files that can be committed to GitHub

It is safe to commit:

```text
clean notebooks with outputs cleared
README files
configuration JSON files
small CSV result tables
small manifest CSV files
selected paper figures
source code
scripts
tests
```

## Label-use note

Deployable SASC/Frozen V3 fitting does not use target labels.

Target labels are used only for evaluation metrics, target-oracle diagnostics, and analysis tables.



