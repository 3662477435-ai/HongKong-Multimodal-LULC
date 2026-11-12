# Reproduction guide

## Code and data are separate

The code is in this repository. Download the four whole ZIP archives listed in
the current Zenodo deposit and extract them into one directory, preserving paths:

- `01_results_tables_metadata.zip`: saved tables, integer confusion matrices,
  run registries, provenance and audit records;
- `02_selected_model_checkpoints.zip`: 83 original selected checkpoints;
- `03_open_provider_inputs_and_splits.zip`: shareable inputs and saved splits;
- `04_prediction_products_and_final_figures.zip`: regional outputs and
  shareable figures (including 600-dpi raster exports).

Use the deposit's current `README_DATA.md` for sizes, SHA-256 checksums and
access restrictions. Historical references to Figshare or previous drafts
inside unchanged archives are not current publication links. The public DOI
of the current deposit has not yet been verified.

The source tree no longer contains old downloaders, combined-package checksum
manifests, publishing logs or a `verify` subcommand that depended on the obsolete
combined package. This does not alter the dataset archives.

## Environment

Use Python 3.11, a compatible PyTorch/torchvision pair, and `pip install -e .`.
Attribution/feature analyses additionally need `pip install -e ".[analysis]"`.
For tests install `pip install -e ".[test]"`.

[environment-observed.txt](environment-observed.txt) describes only the observed
packaging environment, not a validated original-training lockfile. Platform-
specific dependency resolution and a clean-machine retraining audit remain
outstanding. No full training was executed during repository organization.

## Saved-matrix reproduction

From the repository directory:

```bash
python run_release.py metrics --data-dir /path/to/extracted_data
```

This recomputes nine main-model OA, mIoU, macro-F1 and kappa values from archived
integer confusion matrices. It writes `reproduced_metrics.json` in the data
directory and compares supported headline metrics to their archived values.
It is not retraining or independent accuracy validation.

## Frozen-checkpoint evaluation

```bash
python run_release.py preflight --data-dir /path/to/extracted_data
python run_release.py evaluate --model all --data-dir /path/to/extracted_data
```

Select an individual model using an exact `id` from
[main_models.json](../metadata/main_models.json). Checkpoints are loaded strictly,
without weight conversion. Missing inputs are reported rather than silently
replaced. Exact inference needs the two withheld rasters documented in
[DATA_ACCESS.md](DATA_ACCESS.md).

`python run_release.py smoke --data-dir /path/to/extracted_data` is only a
synthetic structural forward check, not a held-out accuracy test.

## Training and supplementary analyses

The main training entry points are:

```bash
python -m hk_landuse.train_comparison_8class --help
python -m hk_landuse.train_hierarchical_fcnn_8class --help
```

Use `--config config_release.yaml` or an external copy with input/output paths
pointing to the data directory. Shared configuration is not a complete
training recipe: select architecture, run name, epochs/stages, seed, modality
set, pretrained initialization and component mode using
[model_registry.json](../metadata/model_registry.json) and the corresponding
runner in `scripts/`. Proposed-model training starts from its FCNN checkpoint.
The optional `--data-dir` flag belongs to `run_release.py`; it is not a flag
supported by every historical module or runner.

- `scripts/run_eight_class_experiments.py`: comparison-model experiments.
- `scripts/run_hfcnn_eight_class_experiments.py`: proposed-model experiments.
- `scripts/run_reviewer_revision_experiments.py`: seeds, modality ablations
  and spatial-protocol experiments.
- `scripts/run_fairness_supplement_20260802.py` and
  `scripts/run_direct48_strong_baselines.py`: equal-budget/direct-input baselines.
- `scripts/run_building_lineage_ablations.py`: building-component experiments.
- `src/hk_landuse/prepare_height_buildings.py`: building-stack preparation.
- `src/hk_landuse/model_expected_gradients_fullpatch_8class.py`: attribution.
- `src/hk_landuse/export_eight_class_figures.py`,
  `scripts/export_seed_spatial_robustness_english.py` and
  `scripts/plot_detailed_local_comparison_8class.py`: main figure families.

Runners retain historical experiment names and output layouts. Use a separate
working/data copy for retraining so new runs do not overwrite the archived
results. Some secondary scripts require provider-native inputs not included
in the prepared-data package; inspect their CLI/configuration before running.
Do not run every retained runner as an unreviewed one-command retraining recipe.

## Evaluation conventions and known limits

Eight targets exclude commercial/office, barren/rocky-shore and BLU=0.
The main 128 × 128-pixel split has 627 training, 123 validation and 125 test
patches, with 62/13/14 spatial groups and no group overlap. Main valid test
support is 1,518,543 pixels. Train-only vocabularies and normalization statistics
must be reused; the 5.12-km blocked protocol does not imply zero residual spatial
dependence.

The existing metrics implementation uses NaN for undefined per-class
precision/F1 and `nanmean` across eligible classes. A class with no predictions
can therefore be excluded from macro-F1, which matters for the original VGG
baseline. This convention is preserved rather than silently changing results.

Use the archived `table_spatial_robustness_5000_restored.csv` for the 5,000-draw
intervals matching final Table 9; the later 100-draw CSV is historical. Audit
records and numerical outputs remain in archive 01 rather than being
duplicated in the code repository.

RGB interfaces and original model structures are retained. The manuscript's
Table 1 currently needs an explicit RGB source row. Exact imagery dates,
component release provenance and the CSDI BLU accession still need author
reconciliation. Full retraining and an exhaustive audit of supplementary
claims were not performed by this cleanup.

## Licences and journal policy

The authors chose CC BY 4.0 for their current original code/documentation
([LICENSE](../LICENSE)) and original dataset contributions. This does not
relicense third-party material; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
Previous MIT grants remain valid for the versions covered by those grants.

Taylor & Francis identifies GIScience & Remote Sensing among the journals
using its [Open & FAIR policy](https://authorservices.taylorandfrancis.com/data-sharing-policies/fair-data-copdess/):
supporting data should be archived in a FAIR-aligned repository with a persistent
identifier and CC BY, CC0 or equivalent reuse terms; a Data Availability
Statement and data citations are required. Where ownership prevents sharing,
the restriction must be disclosed at submission and considered by the editors.

Its [code-sharing guidance](https://authorservices.taylorandfrancis.com/data-sharing/share-your-data/share-code/)
is separate from the data policy: share code, identify the relevant version,
provide a software-availability statement and an open licence, preferably an
OSI-approved software licence. It does not specifically mandate CC BY for code.
The current author-selected CC BY licence has not been silently replaced with MIT.
Before publication, archive the exact code version with a persistent identifier
and cite it separately from the data DOI.

Policy sources checked 2026-11-12. The journal's author-instructions page
returned HTTP 403 during this check; the policy statements above are based on
the accessible official publisher pages. Repository cleanup alone is not
certification of full journal compliance or approval of third-party exceptions.

## Tests

```bash
python -m pytest -q
```

Tests cover unit behavior and package/layout contracts. Passing them does not
replace full-data inference, GPU training or editorial review.
