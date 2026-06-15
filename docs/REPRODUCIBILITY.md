# Reproducibility guide

## Recommended review-time path

For reviewers who do not want to rerun image-model inference, use the cached-logit workflow:

1. Prepare source temperature CSV.
2. Prepare source-bank descriptor CSV.
3. Prepare target-stream descriptor CSV.
4. Run `scripts/run_sasc_from_cached_logits.py`.
5. Compare generated tables with the manuscript tables.

## Full rerun path

The full rerun requires:

- ImageNet-100 clean source data
- ImageNet-C corruptions filtered to the same 100 classes
- ResNet-18, ResNet-50, and ViT-B/16 checkpoints
- enough compute to generate logits for all seeds, corruptions, and severities

The full rerun notebooks are indexed in `notebooks/README.md`.

## Leakage policy

SASC/Frozen V3 deployment uses:

- source temperature fitted on clean labeled source calibration data;
- source-bank labels only to fit source-bank target temperatures;
- unlabeled target stream entropy for deployment.

It does not use target labels or target metrics for fitting or method selection.
