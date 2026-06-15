# Results directory

This directory is reserved for small final result tables used by the paper artifact.

Large outputs are intentionally not stored in this GitHub repository.

## What can be stored here

Small files may be included here, such as:

```text
summary CSV tables
compact comparison tables
bootstrap confidence interval tables
paper-ready metric summaries
small diagnostic CSV files
```

## What should not be stored here

Do not commit large artifacts such as:

```text
cached logits
source-bank logits
model checkpoints
ImageNet image files
ImageNet-C image files
large .pt / .pth / .ckpt files
large .npz / .npy files
large .zip / .tar / .tar.gz files
```

Large artifacts should be provided through an external artifact link, such as Zenodo, Google Drive, Hugging Face Dataset, or another approved storage location.

## Reproducibility note

The review-time workflow starts from cached logits and descriptor tables. This directory may contain small final outputs, but the full cached-logit artifact package is stored externally.
