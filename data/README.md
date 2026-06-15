# Data directory

This directory is intentionally kept empty in the public repository.

The experiments in this paper use **ImageNet-100** as the clean source distribution and **ImageNet-C** as the corrupted target distribution. Because these datasets are large and have their own distribution terms, the image files are not committed to this repository.

## Datasets used

### 1. ImageNet-100

The clean source dataset is **ImageNet-100**, downloaded from Kaggle.

In the notebooks, the dataset was downloaded using the Kaggle dataset `ambityga/imagenet100`.

```python
# Upload kaggle.json first, then run:
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

# Download ImageNet-100
!mkdir -p /content/data
!kaggle datasets download -d ambityga/imagenet100 -p /content/data --unzip
```

In some runs, the already-downloaded archive was copied from Google Drive and extracted in Colab:

```python
from google.colab import drive
drive.mount('/content/drive')

!cp "/content/drive/MyDrive/imagenet 100/archive (1).zip" /content/
%cd /content
!unzip -q "archive (1).zip"
```

The expected extracted clean dataset structure is:

```text
imagenet100/
├── train/
└── val/
```

The notebooks then use a fixed 100-class subset and save the class list, class-to-index mapping, and split metadata into the generated metadata/manifests.

### 2. ImageNet-C

The corrupted target dataset is **ImageNet-C**, downloaded from Zenodo.

In the notebooks, the ImageNet-C tar files were downloaded into Google Drive using:

```python
from google.colab import drive
drive.mount('/content/drive')

!mkdir -p "/content/drive/MyDrive/imagenet-c"
%cd /content/drive/MyDrive/imagenet-c

!wget -O noise.tar   "https://zenodo.org/records/2235448/files/noise.tar"
!wget -O blur.tar    "https://zenodo.org/records/2235448/files/blur.tar"
!wget -O weather.tar "https://zenodo.org/records/2235448/files/weather.tar"
```

To check the downloaded files:

```python
!ls -lh "/content/drive/MyDrive/imagenet-c"
```

Expected approximate file sizes:

```text
noise.tar    ≈ 22.57 GB
blur.tar     ≈ 7.11 GB
weather.tar  ≈ 12.80 GB
```

The notebooks extract the tar files and then filter ImageNet-C to the same 100 classes used in ImageNet-100.

## Corruptions used in the paper

The paper uses the following 10 ImageNet-C corruption families:

```text
brightness
defocus_blur
glass_blur
motion_blur
zoom_blur
gaussian_noise
shot_noise
impulse_noise
fog
snow
```

Each corruption is evaluated at severities:

```text
1, 2, 3, 4, 5
```

## Expected local layout

A typical local or Colab layout after preparing the data is:

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

The original experiments were run in Google Colab using Google Drive paths. Before rerunning locally or in another environment, update the dataset root paths in the notebook configuration cells or in the relevant config files.

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

These files are intentionally excluded from the public repository and should remain ignored by `.gitignore`.

## Reproducibility note

This repository provides the code, configurations, documentation, selected figures, and notebook workflow needed to reproduce the method and analysis. The datasets must be downloaded separately by users with the appropriate access rights and sufficient storage.


