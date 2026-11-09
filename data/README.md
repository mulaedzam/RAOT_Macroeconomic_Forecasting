# Data

This repository is designed around the journal paper's data-availability statement: public source links are retained, processed modelling datasets should be deposited where redistribution is permitted, and restrictions should be stated explicitly.

## Folders

- `raw/south_africa/` and `raw/brazil/`: source downloads. Raw third-party files are intentionally not bundled by default; use `metadata/source_portals.csv` and `metadata/brazil_source_file_manifest.csv` to reconstruct the source set.
- `interim/`: extraction/harmonisation intermediates used by the South African pipeline.
- `processed/`: exact model-ready artefacts from the archived experiment environment should be copied here before public release.
- `metadata/`: feature maps, release-shift assumptions, source manifests and source portals.

## Important reproducibility point

The release shifts are operational publication-lag proxies, not historical real-time data vintages. Do not describe these files as a true real-time vintage database.

## Redistribution

Before uploading raw downloads, check each provider's terms. For third-party exports (for example Eulerpool or Investing.com candidate files), prefer download instructions and identifiers over redistribution unless the licence clearly permits redistribution.
