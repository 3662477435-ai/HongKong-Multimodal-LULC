# Hong Kong multimodal LULC: data, trained models and final products

Authors: Yufeng Liu (Beijing Normal University) and Hongsheng Zhang (The University of Hong Kong).
README revision: 2026-11-12.

This dataset supports “Urban-context residual learning for multimodal land use/land cover mapping in high-density mountainous coastal cities”. It contains shareable study inputs, saved spatial splits, original trained checkpoints, evaluation source tables and final shareable products. Executable analysis code is maintained separately on [GitHub](https://github.com/3662477435-ai/HongKong-Multimodal-LULC).

Zenodo upload workspace: https://zenodo.org/uploads/24140176 . This is a draft-management link, not a public dataset DOI. Replace it with the verified published record DOI when the deposit is published. Do not cite the earlier Zenodo draft or the unpublished Figshare record as this dataset's public location.

## 1. Files

| File | Contents | Size (bytes) |
|---|---|---:|
| `01_results_tables_metadata.zip` | Main and supplementary source tables, metrics, 83-run registry, nine main-model settings, input manifest, configuration, audits and access notes; no executable code | 3,756,180 |
| `02_selected_model_checkpoints.zip` | 83 original study checkpoints, including proposed, comparison and supplementary experiment models | 3,197,087,401 |
| `03_open_provider_inputs_and_splits.zip` | Shareable provider inputs, aligned derived rasters, saved train/validation/test splits and train-only preparation metadata | 2,608,711,155 |
| `04_prediction_products_and_final_figures.zip` | Six models' full-region classification/confidence products and final shareable English figures, including 600-dpi raster exports | 270,008,767 |
| `README_DATA.md` | Current deposit instructions, third-party sources and access limitations | This file |

The four ZIPs are complete archives, not numbered parts. No joining is required. This README supersedes older repository-location and availability statements inside the archived metadata; it does not change archived model weights, data or numerical results. No separately uploaded `data_asset_manifest.json` or checksum file is required: whole-archive SHA-256 values are provided below.

## 2. Use and reproduction

1. Download the four ZIP files and verify their SHA-256 values.
2. Clone the code repository separately, then extract all four ZIPs into the same code-repository directory, preserving paths. Allow at least 14 GB for the archives and extracted data.
3. Use Python 3.11, install a compatible PyTorch/torchvision pair, and follow the repository's environment instructions. The packaging environment is not a validated original-training lockfile.
4. Run `python run_release.py metrics` to recompute main metrics from archived integer confusion matrices. No GPU or withheld imagery is needed for this step.
5. Run `python run_release.py preflight` before inference. Frozen-model evaluation requires all original inputs, including the two withheld rasters described below.

The historical `run_release.py verify` command checks the earlier combined-package manifest, not this separated deposit. The historical `scripts/download_zenodo_data.py` targets an earlier draft and must not be used to download this deposit. Use the current Zenodo file list and the hashes in this README.

Study rasters are aligned to a common 10 m grid over Hong Kong; use each GeoTIFF's stored coordinate reference system, transform, nodata and band descriptions. The main saved split contains 627 training, 123 validation and 125 test patches of 128 × 128 pixels, with no spatial-group overlap. Saved training-only vocabularies and normalization metadata must be reused rather than refitted on validation/test data. Per-file input paths and hashes are in `data_manifest.csv`; checkpoint/run mappings are in `model_registry.json` and `main_models.json`.

## 3. Inputs used but not included

RGB remains part of the original model inputs. The public archives omit the following third-party-derived rasters because their redistribution or combined relicensing conditions have not been cleared for this release:

### Google Earth RGB

Required file: `artifacts/full/prepared/highres_rgb_10m.tif` (three bands).

Source: [Google Earth](https://earth.google.com/). Earlier study provenance describes imagery mainly from 2021; precise acquisition dates and imagery-provider credits still require author confirmation. See [Google Earth terms](https://www.google.com/help/terms_maps-earth/) and [Geo permissions guidance](https://about.google/brand-resource-center/products-and-services/geo-guidelines/). Permission to redistribute the imagery as a standalone raster has not been established, so neither the RGB raster nor the RGB-based local-comparison Figure 8 is included.

Exact local raster SHA-256: `2c5780d0914b32ab38c0e51ed81ab2e7565994bbb8548b2aac7c6e9d8b64d2d8`.

### Mixed-source building height/function stack

Required file: `artifacts/full/prepared/building_height_function_14band_10m.tif` (14 bands).

- Height: Che et al., **3D-GloBFP**, [dataset record](https://doi.org/10.5281/zenodo.15459025). The record describes 2020 building heights in metres.
- Function: **SinoBF-1**, [dataset record](https://doi.org/10.5281/zenodo.17844789); Li et al. (2026), [associated paper](https://doi.org/10.1038/s41467-026-69589-5).

The derived stack contains nine building-function indicators (Residential, Commercial, Industrial, Educational, Healthcare, Administrative, Public service, Sport and art, Unknown), followed by coverage fraction, mean height (m), maximum height (m), floor-area-ratio proxy (3 m per floor), and height range (m). Its preparation is implemented in `src/hk_landuse/prepare_height_buildings.py` in the code repository.

Exact local raster SHA-256: `3ad2f82264dc0e6e8ca337d66da19f565a1724c9794dfa9f99151337a8337214`.

Obtain component datasets from the original providers under their own terms. We do not redistribute or relicense the combined stack. This does not imply that all upstream datasets prohibit redistribution; the exact source releases used locally still need confirmation from acquisition records.

### What can be reproduced without these files?

Metrics from archived confusion matrices are self-contained. Exact input-to-prediction inference with the frozen checkpoints requires lawful access to both exact withheld study rasters. Different imagery vintages or newly reconstructed building stacks are not guaranteed to reproduce the archived outputs. The original RGB interfaces, checkpoint structures and numerical results are retained; missing inputs are not silently substituted.

## 4. Other sources and attribution

Sentinel-1 GRD and Sentinel-2 Level-2A are Copernicus Sentinel products accessed through Google Earth Engine. VIIRS nighttime-light composites are from the Earth Observation Group catalogue. OSM-derived POI/AOI, roads and semantic features require attribution to OpenStreetMap contributors and retain ODbL 1.0. BLU supervisory and building-outline validation inputs originate from the HKSAR Government/CSDI Geoportal and retain applicable provider terms. See `data_manifest.csv`, `DATA_ACCESS.md` and `THIRD_PARTY_NOTICES.md` in archive 01 and the current GitHub documentation. The exact CSDI BLU source accession/version remains an acquisition-metadata item to complete.

## 5. Licence

Copyright 2026 Yufeng Liu and Hongsheng Zhang. The authors' original data contributions, trained-model contributions, tables and results are licensed under [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/). Retain attribution, link the licence and indicate changes. Current original code and documentation on GitHub also use CC BY 4.0.

CC BY 4.0 covers only rights held by the authors. It does not replace third-party source-data, database, dependency or pretrained-component terms, does not relicense OSM data, and does not authorize access to or redistribution of the withheld inputs. Earlier MIT grants remain valid for the corresponding historical code versions.

## 6. Archive integrity

These hashes identify the four currently uploaded whole ZIP archives; the small README update does not alter their bytes.

```text
a959c48094daf4b2ad64a4fffcb4893ee534e5a1819951c5e9d0f71346731a62  01_results_tables_metadata.zip
20180a0b5536dc1b79b8d3dbb6df65f581cf842465d9beee5d464f6c23522d04  02_selected_model_checkpoints.zip
e611cf0f91f76ddcfee11f720ff7791d4181f2aca31e954e0fba373ffacf0657  03_open_provider_inputs_and_splits.zip
d0456890881bd49c397f8ba5bd9fea25c8fa974499e583b807ced3252c1bf6e6  04_prediction_products_and_final_figures.zip
```

## 7. Citation and known limits

Cite Yufeng Liu and Hongsheng Zhang, this dataset title and the verified dataset DOI/version once published; cite the code repository separately. Do not use a draft-management URL as a DOI.

Packaging checks did not include full retraining or exhaustive validation of every supplementary claim. Archived audit notes document remaining provenance limitations. The final manuscript's Table 1 currently omits an explicit row for the RGB input present in the code and checkpoints; reconcile that source description before submission. No numerical result or model weight has been changed by this documentation revision.
