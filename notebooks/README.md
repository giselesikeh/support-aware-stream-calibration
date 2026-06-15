# Final notebook index

Only the cleaned final notebooks should be committed here. Do not commit old attempts, broken Colab tests, duplicate versions, or notebooks with large outputs.

Recommended final names:

| Notebook | Role |
|---|---|
| `01_vit_training_logits_and_source_bank.ipynb` | ViT-B/16 training, clean/target logits, source TS, target-oracle diagnostics, source-bank logits. |
| `02_vit_sasc_analysis_and_diagnostics.ipynb` | ViT-B/16 SASC/Frozen V3 analysis, Safe V2, support diagnostics, LOCO/LOSO, ablations, bootstrap tables. |
| `03_resnet_baseline_protocol_rerun.ipynb` | Earlier repaired ResNet-18 baseline protocol: baseline, TS, TS-P, PseudoCal, MM-NLL-TS, regime-aware audit. |
| `04_resnet_sasc_analysis_and_diagnostics.ipynb` | Main ResNet-18/ResNet-50 SASC/Frozen V3 analysis and final ResNet deliverables. |
| `05_close_baselines_and_method_comparison.ipynb` | Close-baseline comparison: UTS-style, SBTS-style, Safe V2, Frozen V3/SASC, bootstrap CIs. |
| `06_extra_diagnostics_and_final_exports.ipynb` | Final supervisor-feedback diagnostics: paired bootstrap, support decomposition, source NLL curves, entropy confound checks, export manifests. |

Before pushing notebooks:

1. Clear large cell outputs.
2. Remove personal Google Drive paths where possible, or move them into config variables.
3. Ensure notebooks run top-to-bottom or clearly state which cached files are required.
4. Keep final CSVs and figures separate from raw data and checkpoints.
