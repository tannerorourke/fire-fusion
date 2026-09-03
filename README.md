# FireFusion

ML modeling pipeline purpose-built for sourcing, analyzing, and predicting wildfire burn arrival and its cause in Washington State, at resolutions down to 500 m.

Ten geospatial products spanning terrain, fuels, weather, human activity, lightning, and fire history are aggregated onto a single daily grid spanning 2003-2020 continuously.

As a case study, we train a spatiotemporal ConvFormer on the datacube to predict, for every cell not already alight, the probability it burns within the next 7 days by new ignition or spread, and which of three cause classes (human, lightning, industrial) is responsible.

**Stats & Features**:

- Full daily coverage from 2003-2020, supervised over the May-October fire season.
- 39 input channels (26 from source processors, 13 derived), normalized, resampled and interpolated into trainable feature layers.
- Custom derived features: Per-cause ignition KDE's, 3x3 cell 3-day rolling active fire, NDVI anomalies, 2 and 5-day cumulative precipitation, 100 and 1000-hr dead fuel moisture, decayed lightning load, Fosberg FWI.
- Circular quantities (N/S and E/W aspect and wind-direction components, day-of-year) decomposed into orthogonal components, so no channel carries 0/360 discontinuities.
- Burn day and cause fused from ignition points, MODIS burned area, FIRMS detections and perimeters; KDE by cause, 3x3 rolling active fire, and days since last burn.
- Water, active fire, burned, and valid cause masks

## Datasets

Four datacubes, available on HuggingFace, are the intended way to use this work.

Each includes a raw `dataset.zarr` including normalized raw and derived features, as well as compiled train/eval/test splits. The `wa****` datasets cover the full state. `cascades500` narrows to a 272 x 272 km (73,984 km$^2$) window over the eastern Cascades and Okanogan NF corridor, which holds the bulk of the state's recorded ignitions. A cell of any tier nests exactly inside one cell of every coarser tier and snaps to a 4 km lattice in EPSG:32610.

| Dataset | Resolution | Grid (y, x) | Coverage | Dataset |
| --- | --- | --- | --- | --- |
| `wa4000` | 4000 m | 103 x 109 | Washington State | [torq1/fire-fusion-wa-4000m](https://huggingface.co/datasets/torq1/fire-fusion-wa-4000m) |
| `wa2000` | 2000 m | 205 x 217 | Washington State | [torq1/fire-fusion-wa-2000m](https://huggingface.co/datasets/torq1/fire-fusion-wa-2000m) |
| `wa1000` | 1000 m | 410 x 433 | Washington State | [torq1/fire-fusion-wa-1000m](https://huggingface.co/datasets/torq1/fire-fusion-wa-1000m) |
| `cascades500` | 500 m | 548 x 544 | Eastern Cascades-Okanogan corridor | [torq1/fire-fusion-cascades-500m](https://huggingface.co/datasets/torq1/fire-fusion-cascades-500m) |

## Sources

| Source | Type | Contributes | Native resolution |
| --- | --- | --- | --- |
| [FPA-FOD, 6th edition](https://www.fs.usda.gov/rds/archive/Catalog/RDS-2013-0009.6) | Fire | ignition time and NWCG cause, per-cause ignition density (KDE, 20 km), burn-event cause | daily, point |
| [USFS Perimeter Layer](https://data-usfs.hub.arcgis.com/datasets/usfs::national-usfs-fire-perimeter-feature-layer/about) | Fire | fire extent through time, burn day for small undetected polygons | daily, <10 m |
| [NASA FIRMS MODIS C6.1](https://firms.modaps.eosdis.nasa.gov/download/) | Fire | active-fire detections, burn-day pinning, active-fire state, 3x3 rolling active fire | daily, ~1 km |
| [PRISM AN81d](https://prism.oregonstate.edu/documents/PRISM_downloads_web_service.pdf) | Meteorology | temperature (mean/min/max), dewpoint, vapour-pressure deficit (min/max), precipitation, 2 and 5-day cumulative precipitation, 100 and 1000-hr dead fuel moisture | daily, 800 m |
| [NOAA AORC v1.1](https://registry.opendata.aws/noaa-nws-aorc/) | Meteorology | relative humidity, wind speed, wind direction decomposed E-W and N-S, Fosberg Fire Weather Index | daily, ~800 m |
| [NCEI SWDI NLDN](https://www.ncei.noaa.gov/pub/data/swdi/database-csv/v2/) | Meteorology | same-day CG strike count, decayed lightning load as a holdover proxy | daily, ~11 km |
| [MODIS MCD15A2H](https://ladsweb.modaps.eosdis.nasa.gov/missions-and-measurements/products/MCD15A2H) | Vegetation | leaf area index | 8-day, 500 m |
| [MODIS MOD13Q1](https://ladsweb.modaps.eosdis.nasa.gov/missions-and-measurements/products/MOD13Q1) | Vegetation | NDVI, NDVI anomaly against day-of-year climatology, land/water mask | 16-day, 250 m |
| [MODIS MCD64A1](https://ladsweb.modaps.eosdis.nasa.gov/missions-and-measurements/products/MCD64A1) | Fire | per-cell burn day and its reported uncertainty, days since last burn | monthly, 500 m |
| [LANDFIRE](https://www.landfire.gov/viewer/) | Geography | elevation, slope, aspect decomposed E-W and N-S | annual, 30 m |
| [NLCD](https://www.mrlc.gov/viewer/) | Geography | fractional impervious surface, canopy cover, cropland fraction | annual, 30 m |
| [USDA CONUS WUI v4](https://www.fs.usda.gov/rds/archive/catalog/RDS-2015-0012-4) | Human | housing density, WUI class index, distance to WUI interface | decadal, block-level |
| [NASA GPWv4](https://doi.org/10.7927/H4F47M65) | Human | population density | 5-year, ~1 km |
| [Census TIGER/Line](https://www.census.gov/cgi-bin/geo/shapefiles/index.php) | Human | distance to nearest road | exact, vector |

See `fire_fusion/dataset/SOURCING.md` for authoritative info on products, versions, links, and per-feature aggregation.

### Fire state, labels and masks

A cell's burn day is fused from five sources, in priority order: an FPA-FOD discovery point (`point`); an MCD64A1 burn day with a FIRMS detection inside its uncertainty window, the detection setting the day (`pinned`); an unpinned MCD64A1 day, uncertainty clipped to 1-7 days (`mcd64`); a FIRMS detection where MCD64A1 saw nothing, either two consecutive detections or one with a discovery point within 2 km in the last 3 days (`firms`); the start day of an undetected USFS polygon of at most 4 km² (`polygon`). MCD64A1 burns in cells more than half cropland count as agricultural burning and are dropped. A cell burns at most once per 30 days.

Physics-driven masking constrains what the model is asked to predict to improve accuracy. The model is never supervised/scored on water, cells already alight, cells already burned in a given year, or burning cells with unknown-cause pixels.

| Name | Meaning |
| --- | --- |
| `burn_next` | 1 if a cell that is neither alight nor already burned this year burns on any of T+1 to T+7 |
| `burn_next_early`, `burn_next_late` | the same label with unpinned MCD64A1 days shifted by minus and plus their uncertainty; scored beside the headline, never trained on |
| `burn_next_cause` | cause of the earliest burn event in the window. `dataset.zarr` carries four classes (natural/lightning, human, industrial, debris). The splits fold debris into industrial, leaving three |
| `land_mask` | 1 on land |
| `no_act_fire_mask` | 1 where the cell is neither alight nor burned earlier this year |
| `valid_cause_mask` | 1 where `burn_next_cause` carries a class |
| `burns`, `burn_src`, `active`, `burned` | the day's fused fire state, carried for scoring and visualization; never model inputs |

The target is burn arrival by ignition or spread, not ignition alone. Each burn event takes the cause of the nearest discovery point within 20 km over the preceding 60 days.

## Modeling

We train a two-path spatiotemporal ConvFormer: 26 dynamic channels (weather, land state) flow through the attention trunk, and 13 static maps (terrain, human footprint, plus the date plane) condition it through FiLM.

- Per-group CNN stems (ResNet MLPs), one per dynamic channel group, each with a learned missing-modality token
- Windowed spatial attention over the grid
- SDPA over the feature axis
- SDPA over the time axis
- A zero-initialized static branch producing per-level, spatially-varying FiLM, applied through encoder and decoder
- CNN decoder upsampling to per-cell 7-day burn probability and cause over 3 classes, each head with a fitted calibrator

### Training

Burn arrival is a heavily imbalanced target, dependent on resolution (the positive rate implies a pos_weight of 944.85 at 4 km and 2683.80 at 2 km). Instead of reweighting, we absorb it by using **unit-weight BCE** and subsampling negatives in the loss. This keeps the per-cell loss proper under extreme imbalance.

The primary term is a **per-day allocation loss**: a softmax over the day's supervised cells and the negative log-likelihood of the cells that burn, at weight `alpha_alloc`. The per-day softmax cancels any constant added to that day's logits; the term scores placement alone and is blind to the overall rate. The per-cell BCE stays as an anchor at `alpha_ign`; the cause head is unchanged. Model selection is on validation allocation ignorance (`select_by: val_alloc`).

### Experiments

Two seed profiles per tier at a uniform embed width (configured in `fire_fusion/model/params.json`), plus a wider `cascades500-optimal` headline model. Reported ladder results are seed means.

- Build: each tier's datacube is built from the raw sources, validated, and published (HuggingFace hosts the cube, B2 the training splits).
- A 4 km width comparison fixes the ladder's embed width to ensure val ignorance doesn't degrade as dimension scales.
- Ladder: Three tiers (wa2000, wa1000, cascades500) each at two seeds, plus the `cascades500-optimal` headline, re-run per rolling-origin fold.

## Pipeline

Building from raw data requires gated API tokens, bulk-download requests, and patience waiting on rate limits. Please reach out me personally if you'd like to rebuild/reproduce the data.

Three staged transforms, selected by `--stage`. This keeps peak RAM to one feature layer plus a few dask chunks rather than the full cube.

1. `extract`: Raw sources $\rightarrow$ `cube.zarr`.
2. `publish`: `cube.zarr` $\rightarrow$ `dataset.zarr`.
3. `compile`: `dataset.zarr` $\rightarrow$ `{train,eval,test}.zarr` + `manifest.json` + train-fitted statistics. `--fold` picks the split years; `full` writes at the dataset root and every other fold writes under `data/processed/<dataset>/<fold>/`.

*Note on transforms*: All data transforms done in the `publish` step are a fixed function of the grid (e.g., `clip`, `log1p`, `to_sin`, and `per_area`). Transforms whose parameters are estimated from data (e.g., `z_score`, `minmax`, and `scale_max`) are done in `compile`. This ensures statistics that are dependent on the train-split choice are separated, and lets `dataset.zarr` act as a redistributable product, irrespective of train/eval/test choice.

*Transferring to other states*: Raw data is reprojected to EPSG:32610 (UTM Zone 10N) CRS and **does not** transfer cleanly to other states without rebuilding data from scratch.

## Setup

[uv](https://docs.astral.sh/uv/) or plain pip. The linux `torch` wheel carries CUDA 12.1, GPU host needs no extra CUDA setup.

```bash
uv venv --python 3.11
uv pip install -r requirements.txt
uv run python -m fire_fusion.<...>
```

To reproduce an exact version, use `requirements.lock.txt` to pin transitive dependencies to exact versions and hashes:

```bash
uv pip sync requirements.lock.txt # reproduce
uv pip compile requirements.txt -o requirements.lock.txt --generate-hashes # regenerate
```

## Commands

- `python -m fire_fusion.dataset.build --dataset <name> --stage <extract|publish|compile|validate|all> --[fold] --[sources] --[splits]`: build a datacube, or read the written splits back. Examples: `--dataset wa2000 --stage compile --fold fold3`, `--stage extract --sources MODIS PRISM` to rewrite only those processors' variables into the existing cube.
- `python -m fire_fusion.dataset.processors.proc_ignitions`: reduce the FPA-FOD SQLite release to `data/raw/fpa_fod/fires.parquet`, once, before the first extract.
- `python -m fire_fusion.training.train --[experiment] --[dataset] --[seed] --[init-from] --[freeze] --[alpha-ign] --[alpha-cause] --[export-b2]`: train the ConvFormer. Requires a built dataset under `data/processed`.
- `python -m fire_fusion.predict --[experiment] --[dataset] --[checkpoint] --[calib] --[split] --[batches]`: turn a trained checkpoint into per-cell burn probabilities.
- `python -m fire_fusion.analysis.extract --[experiment] --[split] --[dataset] --[checkpoint]`: run a checkpoint over one split into a prediction archive plus its sidecar.
- `python -m fire_fusion.analysis.score <native|compare|pooled|allocation|cause|localization|extract> --experiments <name...> --[split] --[footprint] --[coarse-res] --[label-source] --[n-boot]`: score archives into one JSON report per stage.
- `python -m fire_fusion.analysis.reference --dataset <name> --[fold] --[bandwidth-km]`: build the climatology reference a scored stage compares against.
- `python -m fire_fusion.analysis.rate --dataset <name> --[fold] --[family]`: fit and score the domain ignition rate.
- `python -m fire_fusion.analysis.viewer --dataset <name> --[store] --[split] --[year] --[archive] --[export]`: step through a tier's store day by day.

Run output (prediction archives, reports, references, rate reports, viewer exports and TensorBoard runs) is written outside the repository, under `../artifacts/` by default. Set `FF_ARTIFACTS_DIR` to relocate the whole tier, or `FF_PRED_DIR`, `FF_REPORTS_DIR`, `FF_REF_DIR`, `FF_RATE_DIR`, `FF_VIEWER_DIR`, `FF_RUNS_DIR` individually.
