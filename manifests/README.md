# Manifests directory

This directory is reserved for small manifest files that describe the paper artifact workflow.

Manifest files help reviewers understand which datasets, cached logits, backbones, seeds, corruptions, severities, and output tables were used.

## What can be stored here

Small manifest files may be included here, such as:

```text
cached logits manifest
source-bank manifest
target-stream descriptor manifest
artifact inventory
notebook output checklist
paper table checklist
reproducibility checklist
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

## Purpose

The purpose of this directory is to make the repository auditable without storing restricted or large data files directly in GitHub.
