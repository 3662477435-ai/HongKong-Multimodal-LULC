# Hong Kong multimodal LULC — analysis code

Code for **Urban-context residual learning for multimodal land use/land cover
mapping in high-density mountainous coastal cities**.

Yufeng Liu (Beijing Normal University) and Hongsheng Zhang (The University of Hong Kong).

This repository contains the proposed model, comparison models and the code
needed for data preparation, training, evaluation and figures. Data, trained
weights and final products are maintained separately in Zenodo.

## Repository layout

```text
src/hk_landuse/       Models, datasets, preprocessing, training and inference
scripts/             Experiment runners, evaluation and figure scripts
tests/               Code and release-layout checks
metadata/            Input provenance, nine main settings and 83-run registry
docs/                Data access, third-party notices and reproduction guide
config_release.yaml  Paths and shared settings
run_release.py       Archived metrics and frozen-model evaluation
```

## Install

Use Python 3.11 and install a hardware-compatible PyTorch/torchvision pair first.
Then run:

```bash
pip install -e .
# Optional attribution/feature analyses and tests:
pip install -e ".[analysis,test]"
```

The dependency declarations are not a locked original-training environment.
See [reproduction guidance](docs/REPRODUCIBILITY.md) for commands and limitations.

## Data and model weights

The current Zenodo preparation workspace is
[upload 24140176](https://zenodo.org/uploads/24140176). It is an author-management
link, not a verified public record or DOI. This repository does not yet claim
a published DOI for that deposit. Replace it with the actual public DOI after
publication; reviewers need an appropriate access link while the deposit is private.

The deposit contains four whole ZIP archives: result tables/metadata, 83
selected checkpoints, shareable inputs/splits, and prediction products/final
shareable figures. Its `README_DATA.md` explains archive hashes and extraction.
Executable code is here, not in the dataset.

Extract all four ZIPs into one **separate data directory**, preserving their
paths. From this repository, run:

```bash
python run_release.py metrics --data-dir /path/to/extracted_data
python run_release.py preflight --data-dir /path/to/extracted_data
python run_release.py evaluate --model all --data-dir /path/to/extracted_data
```

Use a quoted absolute Windows path where appropriate. The default, when
`--data-dir` is omitted, is this repository directory.

## Inputs not redistributed

The trained models retain **Google Earth RGB** and **14-band building
height/function** inputs. Their rasters are not included in the public package
because redistribution or combined relicensing conditions have not been cleared.
The building components originate from 3D-GloBFP and SinoBF-1. The RGB-based
local-comparison Figure 8 is also withheld.

See [exact filenames, checksums and provider routes](docs/DATA_ACCESS.md).
Archived-matrix metric reproduction works without these inputs. Exact
frozen-checkpoint inference requires lawful access to both exact study rasters;
there is no silent substitution, channel removal or checkpoint conversion.

## Code entry points

| Task | Entry point |
|---|---|
| Proposed model | `src/hk_landuse/hierarchical_fcnn_model.py` |
| Comparison architectures | `src/hk_landuse/comparison_models.py` |
| Comparison-model training | `python -m hk_landuse.train_comparison_8class --config config_release.yaml` |
| Proposed-model training | `python -m hk_landuse.train_hierarchical_fcnn_8class --config config_release.yaml` |
| Frozen-model metrics/evaluation | `python run_release.py --help` |
| Prediction maps | `python -m hk_landuse.predict_experiment_models_8class --help` |

Training needs the run-specific options and all inputs, not just the shared
configuration. Follow [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

## Licence and citation

The authors' original code and documentation are licensed under
[CC BY 4.0](LICENSE), as requested by the authors. Preserve attribution, link
the licence and indicate changes. [Third-party terms](docs/THIRD_PARTY_NOTICES.md)
remain applicable; earlier MIT grants are not revoked.

Use [CITATION.cff](CITATION.cff) to cite the code. Cite the separately archived
dataset and original data providers independently. A version-specific code DOI
has not been assigned; GitHub's commit history provides the code version record.

Historical publishing records, duplicate explanations, manuscript-editing tools
and unused calibration experiments were removed from the current tree during
the 2026-11-12 cleanup; they remain recoverable from Git history. Existing
Releases and Zenodo data files have not been deleted or modified.
