from dataclasses import dataclass
import numpy as np
from rasterio.enums import Resampling
from typing import Dict, List, Optional, Sequence, Tuple
from xarray.core.types import InterpOptions

"""
A clear cell is positive if it burns within this IGN_HORIZON_DAYS days.
- A single day leaves the positive class much too sparse to supervise; a week 
  still reads as a short-range forecast.
- The causal label and validity mask naturally pick this up downstream.
"""
IGN_HORIZON_DAYS = 7


"""
Every channel in the cube belongs to one CHANNEL_GROUPS group. The model conditions on these: 
- STATIC and QUASI_STATIC carry no per-day signal and steer the dynamic path
- MET (meteorology) and STATE evolve daily
- SCALAR is the spatially constant date term

CAUSAL_CLASSES determine output shape of prediction head
- DEBRIS and INDUSTRIAL are merged into INDUSTRIAL at compile time. DEBRIS is a few
  hundred labelled cell-days vs five-figs for lightning.
"""
CHANNEL_GROUPS = ("STATIC", "QUASI_STATIC", "MET", "STATE", "SCALAR")

CAUSAL_CLASSES = [
    "NATURAL_LIGHTNING",
    "HUMAN",
    "INDUSTRIAL",
    "DEBRIS"
]

CAUSE_MERGE = {"DEBRIS": "INDUSTRIAL"}

def compiled_cause_classes() -> List[str]:
    """ CAUSAL_CLASSES after merging, in output order; index is the label value. """
    out: List[str] = []
    for c in CAUSAL_CLASSES:
        target = CAUSE_MERGE.get(c, c)
        if target not in out:
            out.append(target)
    return out


def cause_index_remap() -> Dict[int, int]:
    """ Maps extracted class index to compiled class index. """
    compiled = compiled_cause_classes()
    return {i: compiled.index(CAUSE_MERGE.get(c, c)) for i, c in enumerate(CAUSAL_CLASSES)}

# -- FPA-FOD NWCG general-cause vocabulary, lower-cased. Anything else, including
#    'missing data/not specified/undetermined', is an ignition of unknown cause:
#    it still counts as an ignition and carries no cause plane.
CAUSE_RAW_MAP = {
    "NATURAL_LIGHTNING": [
        "natural",
    ],
    "HUMAN": [
        "arson/incendiarism",
        "firearms and explosives use",
        "fireworks",
        "misuse of fire by a minor",
        "recreation and ceremony",
        "smoking",
        "other causes",
    ],
    "INDUSTRIAL": [
        "equipment and vehicle use",
        "power generation/transmission/distribution",
        "railroad operations and maintenance",
    ],
    "DEBRIS": [
        "debris and open burning",
    ],
}

"""
Scaling down the WUI classes to 5 classes provides a more consistent
and interpretable signal.
"""
WUI_CLASS_MAP: dict[str, int] = {
    # uninhabited
    "Uninhabited_Veg": 0,
    "Uninhabited_NoVeg": 0,
    "Water": 0,
    # low density (rural)
    "Very_Low_Dens_Veg": 1,
    "Very_Low_Dens_NoVeg": 1,
    # non-WUI urban / town (built, no veg)
    "Low_Dens_NoVeg": 2,
    "Med_Dens_NoVeg": 2,
    "High_Dens_NoVeg": 2,
    # WUI intermix
    "Low_Dens_Intermix": 3,
    "Med_Dens_Intermix": 3,
    "High_Dens_Intermix": 3,
    # WUI interface
    "Low_Dens_Interface": 4,
    "Med_Dens_Interface": 4,
    "High_Dens_Interface": 4,
}

# purelyy for readability
WUI_CLASSES = {
    "UNINHABITED_WATER",
    "RURAL",
    "NOWUI_URBAN",
    "WUI_INTERMIX",
    "WUI_INTERFACE"
}

"""
NLCD land cover classes
"""
LAND_COVER_RAW_MAP = {
    0: [11], # water
    1: [12], # snow
    2: [21, 22], # developed, < 49%
    3: [23, 24], # developed >= 50%
    4: [31], # barren
    5: [41, 42, 43], # forest
    6: [52, 71], # farmland
    7: [90, 95], # wetlands
}

@dataclass
class Feature:
    name: str = ""
    key: Optional[str] = ""                         # unique key to access data
    group: Optional[str] = None                     # CHANNEL_GROUPS membership; untagged never reaches the cube
    clip: Optional[Tuple[float, float]] = None
    resampling: Optional[Resampling] = None         # strategy for filling missing pixels in feature grid
    time_interp: Optional[Tuple[str, InterpOptions]] = None # "time" = broadcast over time D, "existing" = fill missing
    
    # OHEs
    num_classes: Optional[int] = 0
    one_hot_encode: Optional[bool] = False

    # Special attrs
    kde_smooth_radius_km: Optional[float] = None
    kde_half_life_days: Optional[float] = None      # decay half-life for the KDE accumulator
    expand_names: Optional[List[str]] = None        # names of new features base feature is expanded into
    
    # labels and masks
    inputs: Optional[List[str]] = None
    is_label: Optional[bool] = False
    is_mask: Optional[bool] = False
    # derived features
    
    func: Optional[str] = ""                        # DerivedProcessor function signature
    # -- the derivation estimates something from the record; it is rebuilt
    # -- against the train years at compile time. That rebuild sees supervised
    # -- days only: a temporal-window operator loses its halo history there and
    # -- must not set this.
    train_dependent: Optional[bool] = False
    drop_inputs: Optional[List[str] | None] = None
    ds_clip: Optional[Tuple[float, float]] = None   # clip values after processing
    ds_norms: Optional[List[str]] = None            # sequence of normalizations


def get_labels():
    return [l for l in drv_feat_config() if l.is_label==True]

def get_masks():
    """ Mask feature declarations, derived and source; every mask equals 1 where
        the cell is usable for the head it gates. """
    return (
        [f for f in drv_feat_config() if f.is_mask==True] +
        [f for feats in base_feat_config().values() for f in feats if f.is_mask==True]
    )

def channel_group_indices(names: Sequence[str]) -> Dict[str, List[int]]:
    """ Group name -> positions of that group's channels in `names`. """
    tagged: Dict[str, str] = {}
    for f in [f for feats in base_feat_config().values() for f in feats] + drv_feat_config():
        if f.group:
            # -- an expanded feature never reaches the cube under its own name
            for n in (f.expand_names or [f.name]):
                tagged[n] = f.group

    out: Dict[str, List[int]] = {}
    for i, n in enumerate(names):
        if n not in tagged:
            raise KeyError(f"channel '{n}' carries no feature group")
        out.setdefault(tagged[n], []).append(i)
    return {g: sorted(idx) for g, idx in out.items()}


def base_feat_config():
    """ Returns the raw-source feature definitions, keyed by source group name. """
    return {
        "PRISM": [
            Feature(
                name = "temp_avg",
                group = "MET",
                key = "tmean",
                clip = (-40.0, 120.0),
                resampling = Resampling.bilinear,
                time_interp = ("existing", "linear"),
                ds_norms = ["z_score"],
            ),
            Feature(
                name = "temp_min",
                group = "MET",
                key = "tmin",
                clip = (-40.0, 120.0),
                resampling = Resampling.bilinear,
                time_interp = ("existing", "linear"),
                ds_norms = ["z_score"],
            ),
            Feature(
                name = "temp_max",
                group = "MET",
                key = "tmax",
                clip = (-40.0, 120.0),
                resampling = Resampling.bilinear,
                time_interp = ("existing", "linear"),
                ds_norms = ["z_score"],
            ),
            Feature(
                name = "dewpoint",
                group = "MET",
                key = "tdmean",
                clip = (-60.0, 90.0),
                resampling = Resampling.bilinear,
                time_interp = ("existing", "linear"),
                ds_norms = ["z_score"],
            ),
            Feature(
                name = "vpd_min",
                group = "MET",
                key = "vpdmin",
                resampling = Resampling.bilinear,
                time_interp = ("existing", "linear"),
                ds_clip = (0.0, 200.0),
                ds_norms = ["log1p", "z_score"],
            ),
            Feature(
                name = "vpd_max",
                group = "MET",
                key = "vpdmax",
                resampling = Resampling.bilinear,
                time_interp = ("existing", "linear"),
                ds_clip = (0.0, 200.0),
                ds_norms = ["log1p", "z_score"],
            ),
            Feature(
                name = "precip_mm",
                group = "MET",
                key = "ppt",
                clip = (0, 150),
                resampling = Resampling.bilinear,
                time_interp = ("existing", "linear"),
                ds_norms = ["log1p", "z_score"],
            ),
        ],
        "AORC": [
            Feature(
                name = "rel_humidity",
                group = "MET",
                key = "rel_humidity",
                clip = (0.0, 100.0),
                resampling = Resampling.bilinear,
                time_interp = ("existing", "linear"),
                ds_norms = ["z_score"],
            ),
            Feature(
                # consumed by the fuel-moisture derivation, then dropped
                name = "rh_max",
                key = "rh_max",
                clip = (0.0, 100.0),
                resampling = Resampling.bilinear,
                time_interp = ("existing", "linear"),
            ),
            Feature(
                name = "wind_mph",
                group = "MET",
                key = "wind_mph",
                resampling = Resampling.bilinear,
                time_interp = ("existing", "linear"),
                ds_clip = (0.0, 100.0),
                ds_norms = ["log1p", "z_score"],
            ),
            Feature(
                name = "wind_dir",
                key = "wind_dir",
                clip = (0.0, 360.0),
                # nearest: a raw angle blends through 180 degrees on 359->1
                # wraparounds under bilinear resampling
                resampling = Resampling.nearest,
                time_interp = ("existing", "nearest"),
                # dropped
            ),
        ],
        "IGNITIONS": [
            Feature(
                # int32 fire id per day where a USFS perimeter polygon is active,
                # painted at its final extent; consumed by the label gate, never
                # a model input
                name = "perimeter_id",
                key = "Fire_Perimeter",
                # NO TIME INTERPOLATION
            ),
            Feature(
                # ign_occ: 1 where an FPA-FOD fire was discovered that day, every
                # cause; ign_cause: one plane per cause class, only mapped causes
                name = "ignitions",
                key = "Fire_Occurence",
                expand_names=["ign_occ", "ign_cause"]
                # NO TIME INTERPOLATION
            ),
            Feature(
                name = "ign_KDE",
                group = "STATE",
                # KDE names are "kde_[burn cause]"
                expand_names = ["kde_natural_lightning", "kde_human", "kde_industrial", "kde_debris"],
                key = "Fire_KDE",
                kde_smooth_radius_km = 20,
                # annual half-life keeps a multi-year spatial ignition prior while
                # holding the accumulator stationary; a train-fit z_score stays
                # calibrated on the later eval/test years
                kde_half_life_days = 365,
                # per_area first: the accumulator is fires per cell; per-km2
                # density holds the same meaning across resolutions
                ds_norms = ["per_area", "z_score"]
                # NO TIME INTERPOLATION
            ),
        ],
        "USDA_WUI": [
            Feature(
                name = "usda_hs_density_km2",
                group = "QUASI_STATIC",
                key = "hs_density",
                # 99 percentile clip in processor
                time_interp = ("existing", "linear"),
                ds_norms = ["log1p", "z_score"]
            ),
            Feature(
                name = "usda_wui_index",
                group = "QUASI_STATIC",
                key = "wui_index",
                # nearest: the index is ordinal; linear blends across classes
                time_interp = ("existing", "nearest"),
                ds_norms = ["z_score"]
            ),
            Feature(
                name = "usda_dist_to_wui_km",
                group = "QUASI_STATIC",
                key = "dist_to_interface",
                time_interp = ("existing", "linear"),
                ds_norms = ["z_score"]
            )
        ],
        "GPW": [
            Feature(
                name = "pop_density",
                group = "QUASI_STATIC",
                resampling = Resampling.nearest,
                time_interp = ("existing", "linear"),
                ds_clip = (0.0, np.inf),
                ds_norms=["log1p", "z_score"]
            )
        ],
        "LANDFIRE": [
            Feature(
                name = "lf_elevation",
                group = "STATIC",
                key = "_Elev",
                resampling = Resampling.bilinear,
                clip = (0.0, 5000.0),
                time_interp = ("broadcast", "linear"),
                ds_norms = ["z_score"]
            ),
            Feature(
                name = "lf_slope",
                group = "STATIC",
                key = "_SlpD",
                resampling = Resampling.bilinear,
                time_interp = ("broadcast", "linear"),
                ds_norms = ["z_score"]
            ),
            Feature(
                # dropped
                name = "lf_aspect",
                key = "_Asp",
                resampling = Resampling.bilinear,
                time_interp = ("broadcast", "linear"),
                
            ),
        ],
        "MODIS": [
            Feature(
                name = "modis_lai",
                group = "STATE",
                key = "MCD15A2H",
                # simple reprojection
                resampling = Resampling.nearest, 
                # Ensure sharp dropoffs are captured
                time_interp = ("existing", "nearest"),
                ds_clip=(0.0, 10.0),
                ds_norms = ["z_score"],
            ),
            Feature(
                # dropped
                name = "mod13q1", # step function holds values for dropoffs (fires)
                expand_names = ["modis_ndvi", "modis_water_mask"],
                key = "MOD13Q1",
                resampling = Resampling.nearest,
                # forward fill in proc_modis
                # time_interp = ("existing", "nearest"),
            ),
            Feature(
                # only days_since_last_burn is a model input; modis_burn and
                # modis_burn_unc are label-side and dropped downstream. The group
                # and norms below reach the age layer alone
                name = "mcd64a1",
                group = "STATE",
                key = "MCD64A1",
                expand_names = ["modis_burn", "modis_burn_unc", "days_since_last_burn"],
                # min over the source pixels keeps the earliest burn day in a cell
                resampling = Resampling.min,
                # NO TIME INTERPOLATION, already on the master days in proc_modis
                # The ceiling is a 'never burned in the record' sentinel, not a
                # duration. z-scoring it throws recent burns past -10 sigma. minmax
                # stays bounded; log1p keeps the recent days legible.
                ds_norms = ["log1p", "minmax"],
            ),
        ],
        "FIRMS": [
            Feature(
                # 1 where a MODIS active-fire detection footprint covers the cell
                # centre that day; label-side and dropped downstream
                name = "firms_detect",
                key = "MODIS_AF",
                # NO TIME INTERPOLATION
            )
        ],
        "NLCD": [
            # LndCov class not used
            # using other features in place
            Feature(
                name = "frac_imp_surface",
                group = "QUASI_STATIC",
                key = "FctImp",
                resampling = Resampling.bilinear,
                time_interp = ("existing", "linear"),
                ds_clip = (0.0, 1.0),
            ),
            Feature(
                name = "canopy_cover_pct",
                group = "QUASI_STATIC",
                key = "tccconus",
                resampling = Resampling.bilinear,
                time_interp = ("existing", "linear"),
                ds_clip = (0.0, 1.0),
            ),
            Feature(
                # fraction of the cell in pasture/hay or cultivated crops, averaged
                # from the 30 m class raster
                name = "cropland_frac",
                group = "QUASI_STATIC",
                key = "LndCov",
                resampling = Resampling.average,
                time_interp = ("existing", "linear"),
                ds_clip = (0.0, 1.0),
            )
        ],
        "LIGHTNING": [
            Feature(
                name = "lightning_strikes",
                group = "STATE",
                # daily CG strike count per 0.1 deg tile
                resampling = Resampling.nearest,
                # the product carries a value (>=1 or a true zero) for every
                # day; no temporal interpolation
                time_interp = None,
                ds_norms = ["log1p", "z_score"],
            )
        ],
        "CENSUSROADS": [
            Feature(
                name = "d_to_road",
                group = "STATIC",
                resampling = Resampling.nearest,
                time_interp = ("broadcast", "linear"),
                ds_clip=(0, 10000), # 10km
                ds_norms = ["log1p", "z_score"]
            )
        ],
    }


def drv_feat_config() -> List[Feature]:
    """ Returns the derived feature definitions, in dependency order: later
        derivations consume the output of earlier ones.
    """
    return [
        # -- fire state: fused burn events and what a forecaster can see. The
        #    satellite and perimeter layers are label-side only and leave here.
        Feature(expand_names=["burns", "burns_early", "burns_late", "burn_src", "burned", "active"],
            func="build_fire_state",
            inputs=["ign_occ", "modis_burn", "modis_burn_unc", "firms_detect", "perimeter_id", "cropland_frac"],
            drop_inputs=["modis_burn", "modis_burn_unc", "firms_detect", "perimeter_id"],
        ),
        Feature(name="no_act_fire_mask", is_mask=True,
            func="build_no_act_fire_mask",
            inputs=["active", "burned"],
        ),
        Feature(name="burn_next", is_label=True,
            func="build_burn_next",
            inputs=["burns", "no_act_fire_mask"],
        ),
        # -- the same label with unpinned MCD64 days at the ends of their
        #    uncertainty windows; scored beside the headline, never trained on
        Feature(name="burn_next_early", is_label=True,
            func="build_burn_next_early",
            inputs=["burns_early", "no_act_fire_mask"],
        ),
        Feature(name="burn_next_late", is_label=True,
            func="build_burn_next_late",
            inputs=["burns_late", "no_act_fire_mask"],
        ),
        Feature(name = "fire_spatial_roll",
            group = "STATE",
            func = "build_fire_spatial_rolling",
            inputs=["active"],
            ds_norms = ["log1p", "z_score"],
        ),
        Feature(name="burn_cause_day",
            func="build_burn_cause_day",
            inputs=["burns", "ign_occ", "ign_cause"],
            drop_inputs=["ign_occ", "ign_cause"],
        ),
        Feature(name="burn_next_cause", is_label=True,
            func="build_burn_next_cause",
            inputs=["burn_cause_day", "burns", "burn_next"],
            drop_inputs=["burn_cause_day", "burns_early", "burns_late"],
        ),
        Feature(name="valid_cause_mask", is_mask=True,
            func="build_valid_cause_mask",
            inputs=["burn_next_cause"],
        ),
        # -- carried into the splits as masks, read by scoring and the viewer
        #    for the day's fire state; the model never consumes them
        Feature(name="burns", is_mask=True),
        Feature(name="burn_src", is_mask=True),
        Feature(name="active", is_mask=True),
        Feature(name="burned", is_mask=True),
        Feature(name="land_mask", is_mask=True,
            func="build_land_mask",
            inputs=["modis_water_mask"],
            drop_inputs=["modis_water_mask"],
        ),
        Feature(
            name = "ndvi_anomaly",
            group = "STATE",
            func = "build_ndvi_anomaly",
            inputs=["modis_ndvi"],
            train_dependent=True,
            drop_inputs=["modis_ndvi"],
            # symmetric clip: negative anomalies (drier than climatology) carry fire-relevant signal 
            ds_clip=(-1.0, 1.0),
            ds_norms = ["z_score"],
        ),
        Feature(expand_names = ["precip_2d", "precip_5d"],
            group = "MET",
            func = "build_precip_cum",
            inputs=["precip_mm"], drop_inputs = None,
            ds_norms = ["log1p", "z_score"],
        ),
        Feature(expand_names = ["dead_fmo_100hr", "dead_fmo_1000hr"],
            group = "MET",
            func = "build_dead_fuel_derived",
            inputs=["temp_min", "temp_max", "rel_humidity", "rh_max", "precip_mm"],
            # EMCbar uses the diurnal extremes; rh_max is consumed and dropped.
            drop_inputs=["rh_max"],
            ds_norms = ["z_score"],
        ),
        Feature(name = "lightning_load",
            group = "STATE",
            func = "build_lightning_load",
            inputs=["lightning_strikes"], drop_inputs = None,
            ds_norms = ["log1p", "z_score"],
        ),
        Feature(expand_names = ["wind_dir_ew", "wind_dir_ns"],
            group = "MET",
            func = "build_wind_ew_ns",
            inputs=["wind_dir"],
            drop_inputs=["wind_dir"],
        ),
        Feature(expand_names = ["lf_aspect_ew", "lf_aspect_ns"],
            group = "STATIC",
            func = "build_aspect_ew_ns",
            inputs=["lf_aspect"],
            drop_inputs=["lf_aspect"],
        ),
        Feature(
            name = "fosberg_fwi",
            group = "MET",
            func = 'build_ffwi',
            inputs=["temp_avg", "rel_humidity", "wind_mph"],
            ds_norms = ["z_score"],
        ),
        Feature(
            name = "doy_sin",
            group = "SCALAR",
            func="build_doy_sin",
            inputs=["time"]
        ),
    ]

