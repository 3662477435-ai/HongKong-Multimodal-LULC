# Data, trained models and results — GitHub Release v1.0.0

Repository: https://github.com/3662477435-ai/HongKong-Multimodal-LULC

Download page: https://github.com/3662477435-ai/HongKong-Multimodal-LULC/releases/tag/v1.0.0

The release page is the authoritative indicator that remote publication has
completed. These archives are prepared before publication. A code-only clone
or GitHub's automatically generated Source code ZIP does not contain the
research inputs or checkpoints. No DOI is claimed: the Figshare record
34937653 remained an unpublished draft after a platform publishing error.

## Download and use

Download `download_research_assets.py` from the release or use
`scripts/download_research_assets.py` in a clone, then run with Python 3.11:

```text
python download_research_assets.py --output research_package
cd research_package
python run_release.py verify
python run_release.py metrics
```

No external libraries are needed for downloading, SHA-256 verification,
merging or extracting. The evaluation commands require the study Python
dependencies (see README.md). Retain at least 35 GB of free disk space to
keep the downloaded parts, reconstructed archives and extracted files.

If you downloaded all assets manually into `release_assets/`, run:

```text
python download_research_assets.py --offline --assets-dir release_assets --output research_package
```

To check downloaded files without extraction:

```text
python download_research_assets.py --offline --assets-dir release_assets --verify-only
```

## Archive contents

| Archive | Contents |
|---|---|
| 01_code_docs_source_tables.zip | Code, environment information, 83-run registry, final source tables, metrics and audit notes |
| 02_selected_model_checkpoints.zip.part001 / .part002 | Two byte-for-byte parts of one ZIP containing 83 selected checkpoints, including nine final main-table model settings |
| 03_open_provider_inputs_and_splits.zip.part001 / .part002 | Two byte-for-byte parts of one ZIP containing exact shareable Sentinel, VIIRS, OSM and CSDI inputs, spatial partitions, normalization and vocabularies |
| 04_prediction_products_and_final_figures.zip | Six-model full-region predictions/confidence products and shareable revised 600-dpi figures |
| release_manifest.json | Every part's size and SHA-256, plus each reconstructed original archive's size and SHA-256 |

The parts are NOT independent ZIP files. Do not unzip a single `.part001`.
The downloader checks every part, concatenates them in numeric order,
checks the reconstructed ZIP, and extracts all four archives into one folder.
It refuses to overwrite existing extracted files. Original byte content of
model and input archives is unchanged. Files were split because GitHub
requires each release attachment to be under 2 GiB:
https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases .

## Rights and reproduction limits

Original code: MIT. Original model contributions and results: CC BY 4.0.
Third-party material retains its source terms, including ODbL for OSM-derived
databases and Government/CSDI attribution and terms for reference rasters.
See LICENSE_STATUS.md, THIRD_PARTY_NOTICES.md and data_manifest.csv.

Google Earth RGB and mixed 3D-GloBFP/SinoBF-1 building features are NOT in
these public files: component redistribution permissions remain unconfirmed.
Their exact filenames, hashes and acquisition routes are documented. Metrics
can be reproduced from the saved integer confusion matrices without these
inputs; complete frozen-model inference requires both obtained lawfully.
The archived 329 main-table checks passed and nine main checkpoint synthetic
forward passes succeeded. This is not a certification of exhaustive
supplementary auditing, clean-machine reproduction or full retraining.

## 中文说明

数据、模型和结果下载在同一仓库的 Releases 页面，不在普通代码文件列表中。
请下载全部附件，使用上述 Python 脚本自动校验、合并和解压，不要单独解压分片。
模型和数据内容未因分片而改变；两项权限未确认的第三方输入仍不公开。
GitHub Release 不是 DOI 存档；Figshare 草稿未成功公开，不应作为已发布数据引用。
