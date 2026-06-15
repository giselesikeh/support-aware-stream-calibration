# Data directory

This directory is intentionally kept empty in the public repository.

The experiments in this paper use **ImageNet-100** as the clean source distribution and **ImageNet-C** as the corrupted target distribution. Because ImageNet and ImageNet-C are large datasets with their own distribution terms, the image files are not committed to this repository.

## Datasets

### ImageNet / ILSVRC2012

The clean source data is based on the ImageNet ILSVRC2012 classification dataset.

Users should obtain access through the official ImageNet download page or another official ImageNet access route:

* Official ImageNet download page: https://image-net.org/download.php
* ILSVRC2012 download page: https://image-net.org/challenges/LSVRC/2012/2012-downloads.php

In this work, we use a 100-class subset of ImageNet, referred to as **ImageNet-100**. The selected class list and class-to-index mapping are saved by the notebooks into the metadata and manifest outputs.

### ImageNet-C

The corrupted target data is based on ImageNet-C, a benchmark for evaluating robustness to common image corruptions.

Users can obtain ImageNet-C from the official public sources:

* ImageNet-C Zenodo record: https://zenodo.org/records/2235448
* Robustness benchmark repository: https://github.com/hendrycks/robustness

This work filters ImageNet-C to the same 100 ImageNet classes used in the clean source data.

## Expected local layout

The notebooks expect the user to configure local dataset paths. A typical local layout is:

```text
data/
├── imagenet100/
│   ├── train/
│   └── val/
└── imagenet-c/
    ├── brightness/
    │   ├── 1/
    │   ├── 2/
    │   ├── 3/
    │   ├── 4/
    │   └── 5/
    ├── defocus_blur/
    ├── glass_blur/
    ├── motion_blur/
    ├── zoom_blur/
    ├── gaussian_noise/
    ├── shot_noise/
    ├── impulse_noise/
    ├── fog/
    └── snow/
```

The original experiments were run in Google Colab with Google Drive paths. Before rerunning locally, update the dataset roots in the notebook configuration cells or in the relevant config files.

## Do not commit large or restricted files

Do not commit the following files or folders to GitHub:

```text
ImageNet image folders
ImageNet-C image folders
large cached logits
model checkpoints
.pt / .pth / .ckpt files
.npz / .npy arrays
.zip / .tar / .tar.gz archives
```

These files are intentionally ignored by `.gitignore`.

## Reproducibility note

The public repository provides the code, configuration files, documentation, selected figures, and notebook workflow needed to reproduce the method and analysis. The datasets must be downloaded separately by users who have the appropriate access rights.

