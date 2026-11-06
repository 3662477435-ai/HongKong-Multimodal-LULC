# Urban-context residual learning for multimodal LULC mapping

Research assets for **Urban-context residual learning for multimodal land use/land cover mapping in high-density mountainous coastal cities**.

**Authors:** Yufeng Liu — Faculty of Geographical Science, Beijing Normal University; Hongsheng Zhang — Department of Geography, The University of Hong Kong.

**Code repository: https://github.com/3662477435-ai/HongKong-Multimodal-LULC . The data/model deposit is not yet published; no DOI has been assigned.**

For full reproduction, download and extract all four Figshare archives into one directory when the deposit is published. This Git repository contains code, documentation and registries, not the multi-gigabyte inputs and checkpoints. `verify` expects the complete extracted deposit, not a code-only Git clone.
The authoritative manuscript is the author's `最终/GISscience` submission directory. Original manuscripts and correspondence are not bundled into this public draft.

## Contents

- `src/hk_landuse/`, `scripts/`: study code, model architectures, training/evaluation and plotting programs.
- `main_models.json`: the nine trained model settings in final Table 3.
- `model_registry.json`: 83 reported-study runs, linking checkpoints, seeds, modality sets and metrics. 0 runs currently have no automatically matched checkpoint.
- `models/`: original selected checkpoints (no optimizer or weight conversion).
- `data_manifest.csv`: exact used-input sizes, SHA-256 checksums, sources and sharing status.
- `artifacts/full/prepared/`: saved train/validation/test manifests, train-only vocabularies and normalization statistics; see access notes for rasters.
- `results/`: archived metrics, tables, attribution and external-validation source data.
- `outputs/`: the six full-region classification/confidence products.
- `figures/`: revised 600-dpi English figures.
- `docs/index.html`: a local catalogue suitable for a future repository landing page.

## Final Table 3

| Model/setting | OA | macro mIoU | macro-F1 |
|---|---:|---:|---:|
| Proposed model | 0.8789 | 0.5946 | 0.7338 |
| FCNN (20 epochs) | 0.8733 | 0.5796 | 0.7211 |
| FCNN (47 epochs) | 0.8794 | 0.5885 | 0.7295 |
| Transformer | 0.8714 | 0.5775 | 0.7185 |
| CNN | 0.8679 | 0.5698 | 0.7123 |
| FCN-ResNet50 (48 to 3) | 0.8189 | 0.4239 | 0.5724 |
| FCN-ResNet50 (direct 48) | 0.8124 | 0.4143 | 0.5647 |
| VGG-UNet-Transformer (48 to 3) | 0.8233 | 0.4255 | 0.6351 |
| VGG-UNet-Transformer (direct 48) | 0.8640 | 0.5467 | 0.6940 |

## Environment and execution

Python 3.11; install PyTorch/torchvision compatible with your hardware, then `pip install -e .`. The observed packaging environment is recorded separately; it is not a validated original-training lockfile. The project's existing broad requirements may need platform-specific resolution.

From this directory run:

```text
python run_release.py verify
python run_release.py metrics
python run_release.py preflight
python run_release.py evaluate --model all
```

`verify` checks published-package file hashes; `metrics` recomputes main OA/mIoU/F1/kappa from the archived integer confusion matrices (no GPU or raw imagery required). `preflight` lists every required input. `evaluate` runs frozen checkpoints on the saved test patches and needs all exact original inputs, including those withheld pending rights verification. Missing imagery is never silently replaced or filled with zeros.

Training entry points are `python -m hk_landuse.train_comparison_8class` and `python -m hk_landuse.train_hierarchical_fcnn_8class`, both with `--config config_release.yaml`. Use the saved run registries and original runner scripts for the seed, stage schedule, modality, initialization and component mode. The proposed model is initialized from its corresponding FCNN checkpoint. Full retraining was not performed during packaging.

## Evaluation definitions

Eight targets exclude commercial/office, barren/rocky-shore and BLU=0. Original 128×128 patches share no spatial groups across splits: train 627/62 groups, validation 123/13, test 125/14. The valid main test support is 1,518,543 pixels. The main 5.12-km protocol is spatially blocked and is not a claim of fully removed spatial dependence.

`metrics.py` uses NaN for undefined per-class precision/F1 and `nanmean` across classes with true support. Thus a class receiving no predictions can be excluded from macro-F1; this matters for the original VGG baseline. This implementation is preserved and documented rather than silently changing the reported study statistic.

## Data access and rights

See `DATA_AVAILABILITY.md`, `DATA_ACCESS.md` and `LICENSE_STATUS.md`. Google Earth RGB and mixed-source building features are not present in public archives while exact redistribution rights remain unresolved. Their original local file paths are kept only in a separate author-only staging file, outside this release directory. Provider links, checksums and required filenames are public metadata. Consequently frozen-matrix metric reproduction is self-contained; exact full input-to-prediction reproduction is conditional on obtaining these same rasters lawfully.

## Reproducibility checks and known issues

Main-table numerical provenance checks are recorded in `audit/`. The later 100-draw spatial-bootstrap CSV is retained as history; `table_spatial_robustness_5000_restored.csv` restores the 5,000-draw table matching every final Table 9 interval. An exhaustive audit of all supplementary/narrative claims and a clean-machine GPU retraining run remain outstanding.

The final manuscript Table 1 does not explicitly list the RGB input present in full-model checkpoints and original loaders. Its source/version description needs author reconciliation. No manuscript text or analysis result was changed during packaging.

## Publication

Planned: a public GitHub code repository linked to a Figshare dataset/model deposit. Large binary files are separated from the Git source bundle. `metadata/record_draft.json` contains verified author/title metadata and approved license scopes and deliberately empty account and DOI fields. Supply actual identifiers only after successful deposit. See `PUBLISHING.md` for remaining steps.
