# Exact inputs and acquisition

Read data_manifest.csv before running inference. Rows marked local_only_permission_pending identify exact necessary files whose redistribution permission has not been established. They have not been uploaded or included in the public-ready archives. Obtain them from the rights holder/provider under its conditions and place the exact file at release_path; verify sha256. A newly reconstructed or different-vintage raster is not guaranteed to reproduce frozen-model scores.

Sentinel source collection: COPERNICUS/S1_GRD and COPERNICUS/S2_SR_HARMONIZED, dates in the manifest. The exact joint mosaic used by config_hierarchical.yaml is included, not the similarly named older mosaic in the parent data directory. OSM-derived data require attribution to OpenStreetMap contributors and the ODbL. Building height: https://doi.org/10.5281/zenodo.15459025 ; building function: https://doi.org/10.1038/s41467-026-69589-5 ; Hong Kong reference/OP validation: https://portal.csdi.gov.hk/geoportal/ .

The prepared manifests contain patch-level OSM semantic tokens required by the model. No original POI name/address dump or raw OP-building attribute dump is included. OP validation summary statistics are included; its raw geometries remain available through the provider.
