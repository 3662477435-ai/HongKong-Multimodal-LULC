# Input sources and files not redistributed

The original model code and checkpoint structures retain the three-channel high-resolution RGB input and the 14-band building input. Neither input raster is included in the public data archives. Their absence does not mean that these modalities were absent from the trained models.

## Google Earth RGB

- Required file: `artifacts/full/prepared/highres_rgb_10m.tif`.
- Source: Google Earth imagery service, https://earth.google.com/ .
- As reported in the final manuscript, imagery over the study area was predominantly acquired in 2025. Exact tile-level acquisition dates and imagery-provider credits should be documented from the original acquisition records.
- Provider conditions: https://www.google.com/help/terms_maps-earth/ and https://about.google/brand-resource-center/products-and-services/geo-guidelines/ .
- SHA-256 of the exact local study raster: `2c5780d0914b32ab38c0e51ed81ab2e7565994bbb8548b2aac7c6e9d8b64d2d8` (108,732,513 bytes).
- The raster and RGB-based local-comparison Figure 8 are not redistributed because permission to redistribute the imagery as a standalone research raster has not been established. Obtain imagery through the provider under its applicable conditions; this repository does not grant those rights or provide a bulk-download workaround.

## Building height and function

- Required file: `artifacts/full/prepared/building_height_function_14band_10m.tif`.
- Building height source: Che et al., 3D-GloBFP, https://doi.org/10.5281/zenodo.15459025 . The linked record describes 2020 building heights in metres; the exact release used locally should be confirmed from the original acquisition records.
- Building function source: SinoBF-1, https://doi.org/10.5281/zenodo.17844789 . Associated paper: Li et al. (2026), https://doi.org/10.1038/s41467-026-69589-5 .
- Derived 10 m stack: nine function-indicator bands (Residential, Commercial, Industrial, Educational, Healthcare, Administrative, Public service, Sport and art, Unknown), followed by building coverage fraction, mean height (m), maximum height (m), floor-area-ratio proxy (3 m per floor), and height range (m). See `src/hk_landuse/prepare_height_buildings.py` for the implemented matching and rasterization.
- SHA-256 of the exact local study raster: `3ad2f82264dc0e6e8ca337d66da19f565a1724c9794dfa9f99151337a8337214` (25,403,880 bytes).
- This mixed-source stack is not redistributed because the combined redistribution and relicensing conditions have not been cleared for this release. Obtain the component datasets from their original providers and follow their attribution and use conditions. This is a release-specific rights limitation, not a claim that all upstream datasets prohibit redistribution. The licence of an associated paper or code repository is not automatically the licence of the dataset.

## Reproduction boundary

Read `metadata/data_manifest.csv` for the required paths and hashes. Saved-confusion-matrix metric reproduction does not need either withheld raster. Exact frozen-checkpoint inference requires lawful access to both exact study rasters. A reconstructed raster or a different imagery vintage is not guaranteed to reproduce the archived predictions or scores. No replacement imagery, zero filling, channel removal or checkpoint conversion is performed.

The shareable inputs comprise Sentinel-1/2, VIIRS, OSM-derived features and HKSAR Government/CSDI supervisory and validation rasters. Sources and original terms are documented in `metadata/data_manifest.csv` and `docs/THIRD_PARTY_NOTICES.md`. Prepared patch manifests retain model-required OSM semantic tokens; raw POI name/address dumps and raw validation-building attribute dumps are not included.
