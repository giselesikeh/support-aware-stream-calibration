# Support-Aware Stream Calibration under Corruption Shift

This repository contains the reproducibility package for the manuscript **Support-Aware Stream Calibration under Corruption Shift**.

The project studies post-hoc calibration for image classifiers under corruption shift using clean ImageNet-100 source data and filtered ImageNet-C target streams. The main method is **Support-Aware Stream Calibration (SASC)**, corresponding to the frozen V3 protocol used in the final experiments.

## What is included

- A clean Python implementation of the core SASC/Frozen V3 method.
- Shared metric, stream-descriptor, temperature-scaling, evaluation, and bootstrap utilities.
- Frozen protocol configuration files.
- Scripts for running SASC from cached logits and generating paper-ready tables.
- A notebook index describing the final notebooks used to produce the manuscript results.
- Documentation for data preparation, reproducibility, and review-time artifact use.

## Main method

SASC fits a monotone entropy-to-temperature mapping on a source bank of labeled pseudo-domain streams. At deployment, it uses only unlabeled target-stream logits to compute stream entropy and choose a temperature. The method also reports whether the target stream lies inside the calibrated source-bank support, below support, or in high-entropy extrapolation.

The frozen SASC configuration is:

- 8 entropy bins
- monotone isotonic entropy-to-temperature mapping
- source-bank-only fitting
- support margin: 0.25 source-bank entropy standard deviations
- lower fallback to source temperature
- capped high-entropy extrapolation with maximum slope 0.75
- upper cap: `min(1.45, source_T + 0.35)`
- no target labels or target metrics used for fitting or deployment

## Repository structure

```text
support-aware-stream-calibration/
├── configs/                 # Frozen protocol and method configuration
├── src/sasc/                # Reusable implementation
├── scripts/                 # Command-line experiment scripts
├── notebooks/               # Final notebook index and naming guide
├── docs/                    # Reproducibility and review documentation
├── data/                    # Data placement instructions only; no dataset redistribution
├── results/                 # Generated result tables; keep small final CSVs only
├── figures/                 # Generated figures; keep paper figures only
├── manifests/               # Run manifests and artifact metadata
└── tests/                   # Lightweight checks for core utilities
```
## Data policy

This repository does **not** redistribute ImageNet, ImageNet-100, ImageNet-C, trained checkpoints, or large cached logits. Reviewers should obtain the datasets from their official sources and place them according to `docs/DATA.md`.

Small final CSV tables and paper figures may be included in `results/` and `figures/` if allowed by the target venue.

## Cached artifact availability

Large cached artifacts are not stored directly in this GitHub repository.

The cached-logit workflow requires external artifacts such as cached target logits, source-bank descriptors, target-stream descriptors, source temperature summaries, and manifests.

For review and public release, these artifacts will be provided through an external artifact link:

```text
External cached artifacts: https://drive.google.com/drive/folders/1w9WKsWdtmtO0BocCXexKKuSRB0lmLr3h
```

The external artifact package is expected to contain:

```text
cached target logits and labels
source-bank descriptor tables
target-stream descriptor tables
source temperature CSV files
target logits manifests
small final result CSV files
```

The GitHub repository contains the code, configurations, cleaned notebooks, scripts, tests, documentation, selected figures, and small result/manifest files only.


## Quick setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
```

## Quick sanity tests

After installing the repository requirements, run the test suite from the repository root:

```bash
pytest
```

These tests check the core SASC/Frozen V3 behavior, including:

```text
positive temperature predictions
support-region labels
lower-fallback behavior
high-entropy extrapolation behavior
temperature cap behavior
basic argmax invariance under positive scalar temperature scaling
```

The tests are intended as a lightweight sanity check for the reusable code in `src/sasc/`. They do not require ImageNet, ImageNet-C, cached logits, or model checkpoints.



## Minimal SASC run from cached logits

Prepare three CSV files:

1. `source_ts.csv` with columns: `backbone,seed,temperature`
2. `source_bank_descriptors.csv` with columns: `backbone,seed,corruption,severity,condition,entropy_mean,source_bank_target_temperature`
3. `target_stream_descriptors.csv` with columns: `backbone,seed,corruption,severity,condition,entropy_mean`

Then run:

```bash
python scripts/run_sasc_from_cached_logits.py \
  --source-ts results/source_ts.csv \
  --source-bank results/source_bank_descriptors.csv \
  --target-stream results/target_stream_descriptors.csv \
  --out-dir results/sasc_run
```

This produces:

- `sasc_temperature_predictions.csv`
- `sasc_model_info.csv`
- `sasc_support_summary_by_backbone.csv`
- `sasc_support_summary_by_backbone_severity.csv`

## Final notebooks used for the manuscript

The original Colab notebooks are large and should be cleaned before committing. The final recommended names are:

```text
01_vit_training_logits_and_source_bank.ipynb
02_vit_sasc_analysis_and_diagnostics.ipynb
03_resnet_baseline_protocol_rerun.ipynb
04_resnet_sasc_analysis_and_diagnostics.ipynb
05_close_baselines_and_method_comparison.ipynb
06_extra_diagnostics_and_final_exports.ipynb
```

See `notebooks/README.md` for what each notebook does and which ones are essential for reviewers.

## Citation

A `CITATION.cff` file is included. Update the manuscript title, author list, DOI, and publication venue after acceptance.

## License

Code is released under the MIT License. Dataset use remains governed by the licenses and terms of the original datasets.
