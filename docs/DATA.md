# Data and artifact placement

This repository does not redistribute ImageNet, ImageNet-C, checkpoints, or cached logits.

Recommended local layout:

```text
data/
├── raw/
│   ├── imagenet100/
│   └── imagenet-c/
├── processed/
│   └── imagenet-c-100/
└── cache/
    ├── logits/
    ├── source_bank_logits/
    └── descriptors/
```

Required derived CSVs for the lightweight SASC script:

- `source_ts.csv`
- `source_bank_descriptors.csv`
- `target_stream_descriptors.csv`

Required columns are documented in the main `README.md`.

For review submission, include only small tables and figures when allowed by the venue. Do not include raw datasets, `.pt` checkpoints, or large `.npz`/`.pt` logits.
