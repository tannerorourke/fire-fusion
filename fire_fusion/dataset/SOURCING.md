# Sourcing

Raw product native resolution and cadence, spatial resample, temporal alignment, QA masking, and normalization chain.

Feature names, resampling modes, interpolation modes and normalization chains below are declared in `fire_fusion/config/feature_config.py` and executed by the processors under `fire_fusion/dataset/processors/`.

## Target grid

- CRS `EPSG:32610` (UTM zone 10N), square cells at the dataset's configured resolution: 4000 m, 2000 m, 1000 m or 500 m.
- Every tier's west and north edges snap outward onto multiples of `LATTICE_M` = 4000 m, the least common multiple of the tier resolutions; a cell of any tier nests inside exactly one cell of every coarser tier. The grid runs from those edges to the smallest cell count covering the requested bounds: 103 x 109 at 4 km, 205 x 217 at 2 km, 410 x 433 at 1 km, 548 x 544 for `cascades500`. Encoder-stride alignment is the loader's concern; it trims the far edge.
- Daily time step over 2003-01-01 to 2020-12-31. 2003 is the first fully clean season, since MCD15A2H starts mid-2002.
- Seasonal datasets supervise months 5 to 10 and extract a 40-day lead / 10-day trail halo around each season. Halo days give the temporal recursions real history and are dropped before the splits are written, so they are never supervised.
- Year splits come from a named fold in `FOLDS`. `full` is train 2003-2016, eval 2017-2018, test 2019-2020, compiled at the dataset root. `fold1`, `fold2` and `fold3` are rolling-origin: each trains only on years before its test block, the three test blocks are disjoint, and calibration years are interleaved. Every statistical normalization is fit on the chosen fold's train years alone.

## Aggregation stages

Each layer passes the same sequence. Nothing reaches the cube on a foreign grid or a foreign time index; the builder raises if a layer's shape, coordinates or time axis diverge from the master.

1. **Preclip.** The master extent is transformed into the source CRS and the native array is clipped to it plus a margin (3 native pixels, or 0.05 degrees for geographic sources). Reprojection then runs on a state-sized window rather than a CONUS or global one.
2. **Spatial resample.** `rio.reproject_match` onto the master grid with the per-feature resampling mode. Float sources with gaps get `nodata=NaN` first, so out-of-extent cells stay NaN instead of picking up the float32 sentinel.
3. **Source transform.** Unit conversion, QA masking and scale factors, in the source's own terms. Per-feature detail is under each source below.
4. **Temporal alignment onto the master daily index**, one of:
   - `broadcast`: a static or single-vintage layer repeated over every day.
   - `existing` + `linear` / `nearest`: interpolation between the source timestamps. Days outside source coverage hold the nearest edge observation rather than going NaN.
   - none: the processor already lands the feature on the master index (rasterized fire layers, lightning, forward-filled MODIS).
5. **Deterministic normalization at publish.** `clip`, then `log1p` and `per_area` in the declared order. These are functions of the value alone, so they are safe to ship in the redistributable cube.
6. **Statistical normalization at compile.** `z_score` and `minmax`, fit on finite train-split cells only and re-fit after each preceding step in the chain. Missing cells stay NaN through this stage and are zero-filled afterwards, which puts them at the post-normalization mean. A deterministic step ordered after a statistical one is rejected.

Table columns below: **Native** is the source resolution and cadence, **Resample** the rasterio mode used in stage 2, **Time** the stage 4 mode, **Norms** the stage 5 and 6 chain in order.

## PRISM AN81d (800 m daily)

Terrain-aware daily meteorology, 1981 to present at 800 m. Downloaded and consolidated to annual per-variable NetCDF by `python -m fire_fusion.dataset.processors.proc_prism`.

- [Web service docs (PDF)](https://prism.oregonstate.edu/documents/PRISM_downloads_web_service.pdf)
- Download endpoint: `https://services.nacse.org/prism/data/get/us/800m/{var}/{YYYYMMDD}`, returning a zipped COG in native EPSG:4326. Throttled under 2 requests/s.
- The service answers an over-quota or transient failure with HTTP 200 and a text body, so the fetcher validates zip magic before parsing. The per-file daily cap is fatal and never retried.

| Feature | Source var | Native | Resample | Time | Norms |
| --- | --- | --- | --- | --- | --- |
| `temp_avg` | `tmean` | 800 m daily | bilinear | linear | z |
| `temp_min` | `tmin` | 800 m daily | bilinear | linear | z |
| `temp_max` | `tmax` | 800 m daily | bilinear | linear | z |
| `dewpoint` | `tdmean` | 800 m daily | bilinear | linear | z |
| `vpd_min` | `vpdmin` | 800 m daily | bilinear | linear | clip(0, 200), log1p, z |
| `vpd_max` | `vpdmax` | 800 m daily | bilinear | linear | clip(0, 200), log1p, z |
| `precip_mm` | `ppt` | 800 m daily | bilinear | linear | log1p, z |

Temperatures arrive in Celsius and are converted to Fahrenheit on load, because the NFDRS and Fosberg equations downstream are hard-wired to Fahrenheit. Temperatures are then clipped to -40 to 120 F, dewpoint to -60 to 90 F, precipitation to 0 to 150 mm. VPD keeps its native hPa.

## NOAA AORC v1.1 (~800 m, hourly native)

Humidity and wind, the fire-weather fields PRISM does not carry. Streamed from S3 and reduced to daily at native resolution by `python -m fire_fusion.dataset.processors.proc_aorc`.

- [AWS Open Data registry](https://registry.opendata.aws/noaa-nws-aorc/)
- Store: `s3://noaa-nws-aorc-v1-1-1km/{year}.zarr`, anonymous access, consolidated Zarr, native EPSG:4326 at ~0.008333 degrees with ascending latitude. The "1km" in the bucket name is a misnomer for 800 m.
- Source variables: `SPFH_2maboveground` (kg/kg), `TMP_2maboveground` (K), `PRES_surface` (Pa), `UGRD_10maboveground` and `VGRD_10maboveground` (m/s).

Hourly to daily reduction happens at fetch time, before anything touches the master grid:

- Relative humidity is recovered per hour from specific humidity, temperature and surface pressure. Vapour pressure `e = q p / (0.622 + 0.378 q)`, saturation vapour pressure over liquid water by Bolton 1980, `es = 611.2 exp(17.67 Tc / (Tc + 243.5))`, and `RH = 100 e / es` clipped to 0-100.
- `rel_humidity` is the daily *minimum* RH and `rh_max` the daily maximum. The pair are the diurnal endpoints the NFDRS fuel model expects.
- `wind_mph` is the daily mean of hourly wind speed, converted from m/s.
- `wind_dir` is the wind-from direction of the daily *vector* mean, `degrees(atan2(-u_bar, -v_bar)) mod 360`. Averaging the components rather than the angles avoids the wraparound artefact.

| Feature | Source var | Native | Resample | Time | Norms |
| --- | --- | --- | --- | --- | --- |
| `rel_humidity` | daily-min RH | ~800 m daily | bilinear | linear | z |
| `rh_max` | daily-max RH | ~800 m daily | bilinear | linear | consumed, then dropped |
| `wind_mph` | daily-mean speed | ~800 m daily | bilinear | linear | clip(0, 100), log1p, z |
| `wind_dir` | daily vector-mean direction | ~800 m daily | nearest | nearest | consumed, then dropped |

RH is clipped to 0-100 and direction to 0-360 on load. Wind direction resamples and interpolates by nearest neighbour only: a raw angle blends through 180 degrees on a 359 to 1 wraparound under bilinear, so it is decomposed to sin/cos components before any smoothing happens. `rh_max` feeds the dead fuel moisture derivation and `wind_dir` the east/west and north/south components; both are dropped once consumed.

## Fire occurrence and perimeter layers

Ignition points come from FPA-FOD, the all-agency US wildfire occurrence record: one point per fire at its discovery location and date, carrying an NWCG general cause. Perimeter polygons come from the USFS national layer. Both are read with geopandas, reprojected to the grid CRS, clipped to the master extent and rasterized directly onto the master grid with `all_touched=False`, so a cell is marked only when its centre falls inside the geometry. Neither takes any temporal interpolation; the rasterizer writes straight to master index positions.

- Short, K.C., 2022. [Spatial wildfire occurrence data for the United States, 1992-2020, 6th edition](https://www.fs.usda.gov/rds/archive/Catalog/RDS-2013-0009.6). USDA Forest Service Research Data Archive, RDS-2013-0009.6.
- [Fire Perimeter Feature Layer](https://data-usfs.hub.arcgis.com/datasets/usfs::national-usfs-fire-perimeter-feature-layer/about)

FPA-FOD ships as a national SQLite release around 1 GB. `python -m fire_fusion.dataset.processors.proc_ignitions` reduces it once to `data/raw/fpa_fod/fires.parquet`, a few columns over the Washington box, which then syncs like any other raw source. FPA-FOD covers every reporting agency: 23,749 fires in the box over 2003-2020, against 5,216 in the USFS occurrence layer. Some agencies report a location at the PLSS section centre, up to about 1 km from the true origin.

| Feature | Source | Native | Aggregation | Norms |
| --- | --- | --- | --- | --- |
| `ign_occ` | FPA-FOD points | vector, point | point burned into its cell on `DISCOVERY_DATE`, every cause | consumed by labels, then dropped |
| `ign_cause` | FPA-FOD points | vector | one-hot `(time, burn_cause, y, x)` over 4 causes, mapped causes only | consumed by labels, then dropped |
| `perimeter_id` | perimeter polygons | vector | int32 fire id painted into every day the polygon is active | consumed by labels, then dropped |
| `kde_natural_lightning`, `kde_human`, `kde_industrial`, `kde_debris` | FPA-FOD points | Gaussian smoothing plus exponential decay, below | per_area, z |

**Occurrence and cause.** Rows without a parseable discovery date or outside the master date range are dropped. Everything else rasterizes into `ign_occ`; only fires whose `NWCG_GENERAL_CAUSE` maps to one of the four classes reach `ign_cause`. A fire whose cause is "missing data/not specified/undetermined" counts as an ignition and carries no cause plane.

| Class | NWCG general causes |
| --- | --- |
| `NATURAL_LIGHTNING` | Natural |
| `HUMAN` | Arson/incendiarism, Firearms and explosives use, Fireworks, Misuse of fire by a minor, Recreation and ceremony, Smoking, Other causes |
| `INDUSTRIAL` | Equipment and vehicle use, Power generation/transmission/distribution, Railroad operations and maintenance |
| `DEBRIS` | Debris and open burning |

**Perimeter activity window.** A perimeter is active from `DISCOVERYD` through `PERIMETERD` inclusive, painted at its final extent for every one of those days, larger polygons first; a small fire inside a complex keeps its own id. An end timestamp landing exactly on midnight is rolled back one day, since it denotes the end of the previous day. Both bounds are clamped to the master index, and end is raised to start where the record has them inverted. Downstream, the layer paints the active state for polygons no satellite detected, and supplies a start-day burn event for such a polygon's cells when the polygon is at most 4 km² (one 2 km cell).

**Ignition KDE.** Per cause, each day's occurrence raster is smoothed with a Gaussian of sigma = 20 km / cell size, then accumulated as an exponentially decayed running sum with a 365-day half-life:

```
load[t] = smoothed[t] + alpha ** dt * load[t-1],   alpha = 0.5 ** (1 / 365)
```

`dt` is the real number of days between consecutive index entries, not one step per entry. On a seasonally windowed index the winter gap spans months, and a per-entry decay would collapse it to a single step and carry a multi-year prior across nearly undecayed. The decay keeps the accumulator stationary, so a z-score fit on the train years stays calibrated on the chronologically later eval and test years; a raw cumulative sum drifts upward by construction and breaks that calibration.

`per_area` runs before the z-score. A spatial kernel normalized in pixel units deposits the same total mass per event whatever the cell size, so the raw layer is mass per cell and rescales with resolution. Dividing by cell area in km squared gives a density that means the same thing on every grid.

## USDA Wildland-Urban Interface

Census-block WUI classification with 1990-2020 vintages; the build uses 2000, 2010 and 2020, reads WA blocks only, and drops blocks flagged as water. Block attributes are rasterized to the master grid by polygon value.

- [Download](https://www.fs.usda.gov/rds/archive/catalog/RDS-2015-0012-4)
- [Metadata](https://www.fs.usda.gov/rds/archive/products/RDS-2015-0012-4/_metadata_RDS-2015-0012-4.html)

| Feature | Source column | Native | Aggregation | Time | Norms |
| --- | --- | --- | --- | --- | --- |
| `usda_hs_density_km2` | `HUDEN{2000,2010,2020}` | census block | rasterize by block, clip at the 99th percentile | linear | log1p, z |
| `usda_wui_index` | `WUICLASS{2000,2010,2020}` | census block | rasterize the remapped ordinal | nearest | z |
| `usda_dist_to_wui_km` | `usda_wui_index` | census block | Euclidean distance transform to the nearest interface or intermix cell, in km | linear | z |

The index interpolates nearest rather than linearly, since it is ordinal and a linear blend lands between classes.

The source's 14 classes collapse onto a 5-level ordinal, and `usda_dist_to_wui_km` measures distance to any cell at level 3 or above:

| Index | Meaning | Source classes |
| --- | --- | --- |
| 0 | uninhabited or water | `Uninhabited_Veg`, `Uninhabited_NoVeg`, `Water` |
| 1 | rural | `Very_Low_Dens_Veg`, `Very_Low_Dens_NoVeg` |
| 2 | built, non-WUI | `Low_Dens_NoVeg`, `Med_Dens_NoVeg`, `High_Dens_NoVeg` |
| 3 | WUI intermix | `Low_Dens_Intermix`, `Med_Dens_Intermix`, `High_Dens_Intermix` |
| 4 | WUI interface | `Low_Dens_Interface`, `Med_Dens_Interface`, `High_Dens_Interface` |

Source class definitions, in housing units per square km and percent wildland vegetation:

- **Intermix** classes are wildland vegetation above 50%. **Interface** classes are at or below 50% but within 2.414 km of an area at or above 75% wildland vegetation. **NoVeg** classes are at or below 50% with no such neighbour.
- Density bands: `Uninhabited` exactly 0, `Very_Low` below 6.177635, `Low` 6.177635 to 49.42108, `Med` 49.42108 to 741.3162, `High` at or above 741.3162.
- `WUIFLAG{year}` in the source is 0 non-WUI, 1 intermix, 2 interface. The build reads `WUICLASS{year}` instead, which carries the density band as well.

## GPWv4 population density

Gridded Population of the World v4.11, population density adjusted to the 2015 UN WPP country totals, at 30 arcsec (~1 km). Five-year vintages 2000 through 2020 are on disk, each stamped at July 1 of its year and linearly interpolated between.

- [Search: all GPW files](https://search.earthdata.nasa.gov/search?q=CIESIN%20ESDIS&hdr=500%2Bto%2B1000%2Bmeters&fpj=GPW&fsm0=Population&fst0=Human%20Dimensions)
- CIESIN, Columbia University, [GPWv4.11 population density](https://doi.org/10.7927/H4F47M65), ESDIS, released 2018-12-31.

| Feature | Native | Resample | Time | Norms |
| --- | --- | --- | --- | --- |
| `pop_density` | 30 arcsec, 5-yearly | nearest | linear | clip(0, inf), log1p, z |

Negative values are dropped to NaN on load. The `log1p` is what makes this trainable: population density is heavily skewed toward a handful of urban cells, and the raw values dominate any shared scale.

## LANDFIRE topography

30 m CONUS topography from the LF 2.2.0 (2020) vintage. Topography does not move over the record, so a single vintage is broadcast across every day.

- [Data viewer](https://www.landfire.gov/viewer/)
- [Data dictionary (PDF)](https://www.landfire.gov/sites/default/files/documents/LF_Data_Dictionary.pdf)

| Feature | Source layer | Native | Resample | Time | Norms |
| --- | --- | --- | --- | --- | --- |
| `lf_elevation` | `_Elev` | 30 m static | bilinear | broadcast | z |
| `lf_slope` | `_SlpD` | 30 m static | bilinear | broadcast | z |
| `lf_aspect` | `_Asp` | 30 m static | bilinear | broadcast | consumed, then dropped |

Elevation is in metres and clipped to 0-5000. Slope is in degrees. Aspect is the compass direction of the slope face in degrees and is decomposed to east/west and north/south components, since a raw bearing has a discontinuity at north that no model should have to learn around.

## MODIS (NASA LAADS)

HDF-EOS2 granules over the five sinusoidal tiles covering the Pacific Northwest (`h08v04`, `h08v05`, `h09v04`, `h09v05`, `h10v04`), collection 061, day granules only. Granules are reprojected per-tile onto the master grid and combined across tiles for a given timestamp by max, which is exact given each tile contributes NaN outside its own footprint. The MCD64A1 burn date combines by min, keeping the earliest burn day.

- [earthaccess API](https://earthaccess.readthedocs.io/en/latest/) for search and download; requires an Earthdata token.
- [NASA data explorer](https://ladsweb.modaps.eosdis.nasa.gov/search/order/1/MYD11A1--61,MCD15A2H--61)

| Feature | Product | Native | Resample | Time | Norms |
| --- | --- | --- | --- | --- | --- |
| `modis_lai` | MCD15A2H | 500 m, 8-day | nearest | nearest | clip(0, 10), z |
| `modis_ndvi` | MOD13Q1 | 250 m, 16-day | nearest | forward fill | consumed, then dropped |
| `modis_water_mask` | MOD13Q1 | 250 m, 16-day | nearest | forward fill | mask, no norm |
| `modis_burn` | MCD64A1 | 500 m, monthly | min | already daily | consumed by labels, then dropped |
| `modis_burn_unc` | MCD64A1 | 500 m, monthly | max | already daily | consumed by labels, then dropped |
| `days_since_last_burn` | MCD64A1 | 500 m, monthly | min | already daily | log1p, minmax |

Every MODIS layer resamples nearest, apart from the MCD64A1 burn date and its uncertainty. These are quality-gated categorical or step-function quantities, and bilinear blending would invent values across a QA boundary or smooth out the sharp drop that a burn is supposed to produce. The burn date takes the minimum over its source pixels (earliest burn day) and the uncertainty the maximum (a burned pixel's value survives the unburned zeros around it).

### MCD15A2H leaf area index

[Product page](https://ladsweb.modaps.eosdis.nasa.gov/missions-and-measurements/products/MCD15A2H) and [file spec](https://ladsweb.modaps.eosdis.nasa.gov/filespec/MODIS/6/MCD15A2H). Combined Terra and Aqua 8-day L4 composite. `Lai_500m` is kept where all of the following hold:

- `FparLai_QC` bit 0 (MODLAND) is 0, good quality.
- `FparLai_QC` bits 3-4 (cloud state) are 0 (clear) or 3 (undefined, assumed clear).
- `FparLai_QC` bits 5-7 (SCF_QC retrieval confidence) are below 4, so the pixel was actually produced.
- `FparExtra_QC` bits 0-1 (land/sea) are 0 or 1, bit 2 (snow/ice) is 0, bit 3 (aerosol) is 0, bit 4 (cirrus) is 0, bit 5 (internal cloud) is 0, bit 6 (cloud shadow) is 0.

Fill values 249-255 are dropped by keeping the DN range 0-100, then the 0.1 scale factor puts LAI in its physical 0-10 range.

### MOD13Q1 NDVI and water mask

[Product page](https://ladsweb.modaps.eosdis.nasa.gov/missions-and-measurements/products/MOD13Q1) and [file spec](https://ladsweb.modaps.eosdis.nasa.gov/filespec/MODIS/6/MOD13Q1). `250m 16 days NDVI` is kept where `250m 16 days VI Quality` satisfies:

- bits 0-1 (MODLAND_QA) at or below 1, so VI was produced and is at worst "check other QA".
- bits 2-5 (VI usefulness) below 13.
- bit 8 (adjacent cloud) is 0 and bit 10 (mixed cloud) is 0.
- bits 11-13 (land/water) equal 1, land.

NDVI carries a 1e-4 scale factor. The water mask is the same quality gate with bits 11-13 in {0, 2, 5, 6, 7}, the ocean and deep inland water codes. QA code 0 decodes as both "good quality" and "shallow ocean", so cells a granule never covered are tracked from the fill value separately rather than folded in as zeros.

Both layers forward-fill from 16-day composites to daily and hold the last composite through 31 December, since resampling alone stops at the final composite date.

### MCD64A1 burned area

[Product page](https://ladsweb.modaps.eosdis.nasa.gov/missions-and-measurements/products/MCD64A1) and [file spec](https://ladsweb.modaps.eosdis.nasa.gov/filespec/MODIS/6/MCD64A1). `Burn Date` gives a single first-burn day of year per pixel per month, with 0 unburned, -1 unmapped and -2 water, and `Burn Date Uncertainty` gives that date's reported error in days. A pixel counts as burned where `QA` bits 0-1 are both set (land, valid) and `Burn Date` falls in 1-366.

The product resolves to a burn *day*. Each granule-month reduces to one master-grid burn date and one uncertainty; the sparse (day, cell) events on the master index become three daily layers:

- `modis_burn`: 1 on the cell's burn day.
- `modis_burn_unc`: the reported uncertainty on that day, clipped to 0-100.
- `days_since_last_burn` (int16): days since the cell's most recent burn at or before the day.

The age walk is causal: a day sees only burns dated on or before itself. It runs over the *full* record from the November 2000 granules forward, not just the extracted years, so the counter enters the window with real history and winter burns outside the seasonal index still reset it. A cell with no burn yet carries the days since 2000-11-01, capped.

That ceiling is why this feature normalizes `log1p` then `minmax` rather than z-scoring. The sentinel is not a duration; z-scoring against it throws recent burns past -10 sigma. Bounding the range keeps the recent days legible.

`days_since_last_burn` is the only MCD64A1 channel the model sees. `modis_burn` and `modis_burn_unc` are label-side, feeding the burn-day fusion below, and are dropped at compile.

## NASA FIRMS active fire

MODIS Collection 6.1 active-fire detections: one point per detected fire pixel, with the along-scan pixel size that pixel was observed at. The archive arrives as yearly country CSVs under `data/raw/firms/modis_<year>_United_States.csv`.

- [FIRMS archive download](https://firms.modaps.eosdis.nasa.gov/download/)
- Columns read: `latitude`, `longitude`, `acq_date`, `confidence`, `scan`, `type`.

| Feature | Native | Aggregation | Time | Norms |
| --- | --- | --- | --- | --- |
| `firms_detect` | ~1 km, daily | detection footprints rasterized per acquisition day | none | consumed by labels, then dropped |

Rows are kept where `type` is 0, a presumed vegetation fire, against 1 to 3 for volcanoes, other static sources and offshore, and where `confidence` is at least 30. Each kept detection paints a circle of radius `scan * 500 m` around its reported centre, `scan` being the along-scan pixel width in km, 1.0 at nadir. A cell is set on the acquisition date when its centre falls inside a footprint.

VIIRS is excluded: its record starts in 2012 and would put a step change in the label part-way through the record.

The layer observes fire already burning and never feeds the model. It pins MCD64A1 burn days, contributes burn events where MCD64A1 saw nothing, and drives most of the active state.

## NLCD (MRLC)

Annual 30 m Collection 1 products, 2000-2020.

- [Data viewer](https://www.mrlc.gov/viewer/)
- [Science product user guide (PDF)](https://www.mrlc.gov/sites/default/files/docs/LSDS-2103%20Annual%20National%20Land%20Cover%20Database%20(NLCD)%20Collection%201%20Science%20Product%20User%20Guide%20-v1.1%202025_06_11.pdf)
- Dewitz, J., 2023. [National Land Cover Database (NLCD) 2021 Products.](https://doi.org/10.5066/P9KZCM54)

| Feature | Source layer | Native | Resample | Time | Norms |
| --- | --- | --- | --- | --- | --- |
| `frac_imp_surface` | `Annual_NLCD_FctImp` | 30 m, annual 2000-2020 | bilinear | linear | clip(0, 1) |
| `canopy_cover_pct` | `nlcd_tccconus` | 30 m, single vintage | bilinear | held constant | clip(0, 1) |
| `cropland_frac` | `Annual_NLCD_LndCov` | 30 m, annual 2000-2020 | average | linear | clip(0, 1) |

The first two arrive as integer percent. Values above 100 (250 is the no-data code) are masked, the rest divided by 100 to a 0-1 fraction, and remaining gaps zero-filled. All three stay in that fraction and take no statistical normalization; they are already on a bounded, physically meaningful scale.

**Cropland fraction.** The share of a cell in NLCD classes 81 (pasture/hay) and 82 (cultivated crops). The class raster is binarized at its native 30 m, then reprojected by average to an area fraction; no-data pixels stay NaN through the average, and the result is zero-filled and clipped to 0-1. It is a model channel and also gates the burn fusion: an MCD64A1 burn in a cell more than half cropland is agricultural residue burning and does not become a burn event.

**Land cover is otherwise not a feature.** Beyond `cropland_frac`, the `Annual_NLCD_LndCov` class raster is unused: the processor retains a one-hot path for it, but the model uses fractional impervious surface and canopy cover in its place, and the water mask comes from MOD13Q1 rather than from class 11. The grouping the one-hot would use, kept for reference, is 250 no-data with 11 water, 12 snow, 21/22 developed low (under 49% impervious), 23/24 developed high (50% and above), 31 barren, 41/42/43 forest, 52/71 shrub and herbaceous, 81/82 farmland, 90/95 wetlands.

The impervious *descriptor* layer (`Annual_NLCD_ImpDsc`: 0 non-urban, 1 roads, 2 urban, 250 no-data) is also unused. Road proximity comes from TIGER/Line vectors, which are exact rather than 30 m rasterized.

## NCEI SWDI NLDN lightning

Daily cloud-to-ground flash counts per 0.1 degree tile (~11 km), derived from Vaisala's National Lightning Detection Network and published by NOAA NCEI.

- [Lightning products overview](https://www.ncei.noaa.gov/products/lightning-products)
- [SWDI documentation and web services](https://www.ncdc.noaa.gov/swdiws/)
- [Bulk per-year CSV archive](https://www.ncei.noaa.gov/pub/data/swdi/database-csv/v2/), files `nldn-tiles-YYYY.csv.gz`, 1986 to present
- Schema `#ZDAY,CENTERLON,CENTERLAT,TOTAL_COUNT`: UTC day as `YYYYMMDD`, the 0.1 degree tile centroid in degrees, and the CG strike count in that tile that day.

| Feature | Native | Aggregation | Time | Norms |
| --- | --- | --- | --- | --- |
| `lightning_strikes` | 0.1 degree, daily | tile lattice gathered to master cell centres | none | log1p, z |

A tile-day is written only when at least one strike was detected, so an **absent tile-day is a true zero, not a gap**. That is what lets this feature skip temporal interpolation entirely: the product already carries a value for every day.

Rather than reproject the full daily stack, tile centroids are snapped to a regular 0.1 degree lattice, counts are scattered onto it by master-index position, and every master cell centre is mapped once to its containing tile. Gathering through that map is an exact nearest-neighbour resample from the coarse tile grid onto the fine master grid: it preserves each tile's true count and invents no sub-tile gradient. Strike days are matched against the master index by membership rather than a range test, since a seasonally windowed index has gaps inside its own bounds.

## TIGER/Line roads (Census)

Primary and secondary road vectors, 2012 vintage, for WA plus the ID and OR borders so distance near the state line is not truncated. Positional accuracy is better than 10 m, far finer than any grid resolution here.

- [Shapefile index](https://www.census.gov/cgi-bin/geo/shapefiles/index.php)

| Feature | Native | Aggregation | Time | Norms |
| --- | --- | --- | --- | --- |
| `d_to_road` | vector, static | Euclidean distance transform from the rasterized road network | broadcast | clip(0, 10000), log1p, z |

Roads ship in NAD83 degrees and are reprojected to the grid CRS before clipping. The network is rasterized to a 0/1 mask, and a Euclidean distance transform gives distance in pixels, scaled by cell size to metres. The result is clipped at 10 km; past that the distinction stops carrying ignition signal.

## Derived features

Computed at publish from features already on the grid, in the declared order, since later derivations consume earlier output. A derivation that estimates a statistic *of the record* cannot ship split-agnostic and is rebuilt against the train years at compile time.

| Feature | Inputs | Aggregation | Norms |
| --- | --- | --- | --- |
| `burns`, `burns_early`, `burns_late`, `burn_src`, `burned`, `active` | `ign_occ`, `modis_burn`, `modis_burn_unc`, `firms_detect`, `perimeter_id`, `cropland_frac` | burn-day fusion, see *Labels and masks* | none |
| `precip_2d`, `precip_5d` | `precip_mm` | trailing 2-day and 5-day rolling sums | log1p, z |
| `dead_fmo_100hr`, `dead_fmo_1000hr` | `temp_min`, `temp_max`, `rel_humidity`, `rh_max`, `precip_mm` | NFDRS 1978 dead fuel moisture recursions | z |
| `lightning_load` | `lightning_strikes` | decayed running strike sum, 4-day half-life | log1p, z |
| `fosberg_fwi` | `temp_avg`, `rel_humidity`, `wind_mph` | Fosberg Fire Weather Index | z |
| `ndvi_anomaly` | `modis_ndvi` | NDVI minus its day-of-year climatology | clip(-1, 1), z |
| `fire_spatial_roll` | `active` | 3-day rolling max, then 3x3 spatial max | log1p, z |
| `wind_dir_ew`, `wind_dir_ns` | `wind_dir` | `-sin` and `-cos` of the bearing | none |
| `lf_aspect_ew`, `lf_aspect_ns` | `lf_aspect` | `sin` and `cos` of the bearing | none |
| `doy_sin` | time index | `sin(2 pi (doy - 1) / 365)` broadcast over the grid | none |

**Dead fuel moisture** follows NFDRS 1978 (Bradshaw and Deeming, GTR INT-169), with response factors and boundary constants matching the WIMS and FireFamilyPlus operational code. Equilibrium moisture content has three humidity branches, all hard-wired to Fahrenheit:

```
H < 10:       EMC = 0.03229 + 0.281073 H - 0.000578 H T
10 <= H < 50: EMC = 2.22749 + 0.160107 H - 0.014784 T
H >= 50:      EMC = 21.0606 + 0.005565 H^2 - 0.00035 H T - 0.483199 H
```

`EMCbar` is the simple average of the hot-dry endpoint (`temp_max` with daily-min RH) and the cool-moist endpoint (`temp_min` with daily-max RH). Precipitation duration is stepped from the daily amount and capped at the 8-hour state-of-weather reporting limit, which is a modelling choice rather than an NFDRS constant, since NFDRS ingests observed duration and the daily record carries amount only. The boundary conditions and per-day response fractions are then:

```
d100  = ((24 - pdur) EMCbar + pdur (0.5 pdur + 41)) / 24
d1000 = ((24 - pdur) EMCbar + pdur (2.7 pdur + 76)) / 24
fm100[t]  = fm100[t-1]  + (d100[t] - fm100[t-1])  * 0.315634
fm1000[t] = fm1000[t-1] + (dbar[t] - fm1000[t-1]) * 0.306811
```

`dbar` is a 7-day running mean of `d1000` held within a contiguous block, which gives the 1000-hour class roughly six weeks of memory against the 100-hour class's two. Both are clipped to 0-60%. On a seasonally windowed index the recursions reseed at the first day of each contiguous block rather than carrying the prior season's fuel state across a winter gap; the 1000-hour class reseeds at the operational 30% default.

**Fosberg FWI** uses the same three-branch EMC on daily mean temperature and daily-min RH:

```
x    = clip(EMC, 0, 30) / 30
eta  = 1 - 2x + 1.5x^2 - 0.5x^3
FFWI = eta sqrt(1 + U^2) / 0.3002
```

**Lightning load** is `load[t] = strikes[t] + alpha load[t-1]` with `alpha = 0.5 ** (1/4)`, so a strike's contribution halves every four days. That approximates how long a lightning-lit fire can hold before it is discovered. The filter carries state across the year boundary on a windowed index, and the 40-day pre-season halo is sized so the carry-over has decayed below 0.1% by the first supervised day.

**NDVI anomaly** subtracts a day-of-year climatology computed over the **train years only**, and is the one derivation recomputed at compile time for that reason. A climatology spanning the whole record would carry eval and test vegetation into every training sample and let each held-out day contribute to the mean it is measured against. Days the train split never observed, such as 29 February when no train year is a leap year, fall back to the nearest observed day of year.

**Fire spatial rolling** answers "is anything burning near me, recently": a 3-day trailing max over `active`, then a 3x3 spatial max, so a cell sees its own and its neighbours' recent activity. `active` carries only what a forecaster has on the day; this channel is causal.

## Fire state, labels and masks

The horizon is 7 days. One day leaves the positive class too sparse to supervise; a week still reads as a short-range forecast. Everything in this section is built at publish from sparse `(day, cell)` event tables, not dense daily cubes; the 500 m tier fits in memory.

### Burn-day fusion

Five products are fused to one burn day per cell. Where they disagree on a cell-day the lowest code below wins; `burn_src` records which supplied it.

| Code | Source | Day taken from |
| --- | --- | --- |
| 1 | `point` | an FPA-FOD discovery, the ignition itself |
| 2 | `pinned` | an MCD64A1 burn day with a FIRMS detection inside that pixel's uncertainty window; the detection sets the day |
| 3 | `mcd64` | an unpinned MCD64A1 day, carrying its pixel's uncertainty clipped to 1-7 days |
| 4 | `firms` | a FIRMS detection where MCD64A1 saw nothing: the first day of a run of at least two consecutive detections, or a detection with a discovery point within 2 km in the last 3 days |
| 5 | `polygon` | the start day of a USFS perimeter no satellite detected, when the polygon is at most one 2 km cell (4 km²) in area |

MCD64A1 burns in cells with `cropland_frac` above 0.5 are agricultural residue burning and are dropped. A cell burns at most once per 30 days; a later day inside that window folds into `active`.

### State layers

| Name | Kind | Definition |
| --- | --- | --- |
| `burns` | mask | 1 on the cell's fused burn day |
| `burns_early`, `burns_late` | - | the same events with unpinned MCD64A1 days shifted by minus and plus their uncertainty; consumed by the bracketing labels, then dropped |
| `burn_src` | mask | the source code 1-5 that supplied the burn day, else 0 |
| `active` | mask | 1 where a FIRMS detection or a discovery point landed in the cell on any of `t-3 .. t`, or an undetected small polygon spans `t` |
| `burned` | mask | 1 where the cell burned earlier in the same calendar year by any non-point source |

`active` and `burned` hold only what a forecaster has on the day; neither reads a retrospective layer. Both ship as masks for scoring and visualization and are never model inputs.

### Labels and masks

| Name | Kind | Definition |
| --- | --- | --- |
| `burn_next` | label | 1 where `no_act_fire_mask` is 1 at `t` and `burns` is 1 on any of `t+1 .. t+7` |
| `burn_next_early`, `burn_next_late` | label | the same from `burns_early` and `burns_late`; scored beside the headline, never trained on |
| `burn_next_cause` | label | cause id of the earliest burn event in that window where `burn_next` is 1, else -1 |
| `no_act_fire_mask` | mask | 1 where the cell is neither active nor already burned this year |
| `valid_cause_mask` | mask | 1 where `burn_next_cause` is 0 or above |
| `land_mask` | mask | 1 where MODIS did not flag deep water |

The target is burn arrival, not ignition alone: a cell scores positive whether a fire starts in it or spreads into it. A cell already alight or already burned cannot be where a fire next arrives, so the positive is gated on `no_act_fire_mask` and supervision is withheld everywhere else. `burn_next_early` and `burn_next_late` bracket the date error of unpinned MCD64A1 days.

Each burn event takes the cause of the nearest discovery point within 20 km over the preceding 60 days, and -1 where no discovery is in reach; a `point` event finds its own discovery at distance zero. Cause ids index `[NATURAL_LIGHTNING, HUMAN, INDUSTRIAL, DEBRIS]`, folded to three at compile. Anything MODIS does not flag as deep water, including cells it never observed, is treated as land.

## File loaders

- [xr.open_dataset](https://xarray.pydata.org/en/v2023.08.0/generated/xarray.open_dataset.html)
- [rasterio.open](https://rasterio.readthedocs.io/en/stable/topics/reading.html)
- [rioxarray.open_rasterio](https://corteva.github.io/rioxarray/html/rioxarray.html)
- [geopandas.read_file](https://geopandas.org/en/v1.1.0/docs/reference/api/geopandas.read_file.html)

MODIS granules are HDF-EOS2 (HDF4), and the GDAL bundled into the rasterio wheels carries no HDF4 driver. `build_utils.read_hdf4_dataset` reads them through `pyhdf` and georeferences each granule from its own `StructMetadata.0` envelope, so no host needs a system GDAL.
