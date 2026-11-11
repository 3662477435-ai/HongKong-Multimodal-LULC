# Separate data deposit (upload in progress)

Code: https://github.com/3662477435-ai/HongKong-Multimodal-LULC

Reserved dataset DOI: **10.5281/zenodo.24137833**. This is not yet a published
record or a working public download route. Do not cite it as available until
publication is confirmed. The historical GitHub Release remains the download
fallback during migration.

## Planned Zenodo contents

1. `01_results_tables_metadata.zip`: source tables, metrics, model registries,
   data manifest, configuration, audits and access notes; no executable code.
2. `02_selected_model_checkpoints.zip`: 83 selected study checkpoints.
3. `03_open_provider_inputs_and_splits.zip`: shareable provider inputs, derived
   rasters and saved spatial split/preparation metadata.
4. `04_prediction_products_and_final_figures.zip`: regional predictions,
   confidence products and public final English 600-dpi figures.

The three large archives retain their previously verified byte content. The
Zenodo ZIPs are whole archives, not numbered GitHub parts. Clone this code
repository, then extract all four archives into the clone. See README_DATA.md
and SHA256SUMS_ZENODO.txt in the deposit for data access and integrity checks.
`python run_release.py metrics` uses the extracted saved confusion matrices.
The old `verify` command targets the historical combined package manifest,
not the new separated data package.

CC BY 4.0 applies to the authors' original contributions. Third-party source
terms and pretrained-component terms remain applicable. Two input rasters and
the RGB-based Figure 8 remain withheld pending redistribution verification;
see DATA_ACCESS.md and THIRD_PARTY_NOTICES.md. Full-input inference requires
lawful access to the same withheld rasters; none are silently substituted.

GitHub data attachments will only be removed after the Zenodo replacement is
public and verified. Historical Git history and earlier license grants remain
unchanged.
