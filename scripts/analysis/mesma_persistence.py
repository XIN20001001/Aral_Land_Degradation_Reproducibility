from __future__ import annotations

"""
Figure 5 persistent MESMA degradation analysis.

Core idea
---------
This script translates the temporal-persistence logic used in Figure 3 to the
continuous annual MESMA fractions BS, PV and NPV.

For each pixel, a persistent MESMA degradation event is accepted only when:

1. The two years immediately before the candidate onset are stable.
2. A degradation signal appears at the candidate onset and persists for
   three consecutive years.
3. The same dominant degradation pathway remains present in the recent
   post-onset record, using the 2020 to 2024 confirmation window.
4. Only the first qualifying event is retained for each pixel.

The accepted event retains THREE continuous components for ternary plotting:

    Bare soil exposure
        positive BS increase relative to the pre-event baseline

    Green vegetation loss
        the part of PV loss that is not matched by an NPV increase

    Vegetation senescence
        the concurrent part of PV loss and NPV increase

The three component magnitudes are normalized only for ternary colour mixing.
Colour darkness represents total event strength.

The archived 2001-2024 water-union mask is excluded before event detection.
Quantitative statistics are calculated at the native MESMA grid.
"""

from contextlib import ExitStack
import json
import math
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.patches import Patch, Polygon
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.warp import reproject
from rasterio.windows import Window
from shapely.geometry import mapping


ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mesma_map_helpers as mesma_maps


MESMA_DIR = ROOT / "data/raw/annual_mesma"

BASIN_SHP = ROOT / "data/boundaries/AralSeaBasinS.shp"
COUNTRIES_SHP = ROOT / "data/boundaries/countries.shp"


ARAL_MASK_30M = ROOT / "data/figure_source/aral_water_union_30m.tif"

OUTDIR = ROOT / "outputs/full/figure5"
FIGDIR = ROOT / "outputs/full/figure5"


ALL_YEARS = list(range(2001, 2025))

PERIODS = [
    (2001, 2005, "2001-2005"),
    (2006, 2010, "2006-2010"),
    (2011, 2015, "2011-2015"),
    (2016, 2020, "2016-2020"),
    (2021, 2024, "2021-2024"),
    (2001, 2024, "2001-2024"),
]


SOURCE_STABLE_YEARS = 2
TARGET_PERSIST_YEARS = 3

RECENT_STATE_START = 2020
RECENT_STATE_END = 2024
RECENT_STATE_FRACTION = 0.80
RECENT_STATE_MIN_OBS = 3


SOURCE_STABILITY_PP = 5.0
MIN_ONSET_COMPONENT_PP = 5.0
RETAIN_FRACTION = 0.50
RETAIN_FLOOR_PP = 2.5


EXCLUSION_MODE = "water_union_2001_2024"


BLOCK_SIZE = 512


DISPLAY_MAX_DIM = 2400


FORCE_RECOMPUTE = False


BAND_INDEX_OVERRIDE = {
    "BS": None,
    "PV": None,
    "NPV": None,
}


COLORS = {
    "BS": np.array([0xD5, 0x5E, 0x00], dtype=np.float32) / 255.0,
    "PV": np.array([0x00, 0x72, 0xB2], dtype=np.float32) / 255.0,
    "NPV": np.array([0xCC, 0x79, 0xA7], dtype=np.float32) / 255.0,
}

PATHWAY_NAMES = {
    0: "Bare soil exposure",
    1: "Green vegetation loss",
    2: "Vegetation senescence",
}

PATHWAY_KEYS = {
    0: "BS",
    1: "PV",
    2: "NPV",
}

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": [
            "Arial",
            "Helvetica",
            "DejaVu Sans",
            "sans-serif",
        ],
        "font.size": 9.5,
        "axes.titlesize": 10.5,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.linewidth": 0.8,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "savefig.facecolor": "white",
    }
)


def annual_raster(year: int) -> Path:
    return MESMA_DIR / f"Aral_MESMA_Degradation_{year}.tif"


def load_vector(path: Path, target_crs) -> gpd.GeoDataFrame:
    frame = gpd.read_file(path)
    if frame.crs is None:
        frame = frame.set_crs("EPSG:4326")
    return frame.to_crs(target_crs)


def normalize_band_name(value: object) -> str:
    text = "" if value is None else str(value)
    return (
        text.strip()
        .upper()
        .replace(" ", "")
        .replace("_", "")
        .replace("-", "")
    )


def resolve_band_indices(src: rasterio.io.DatasetReader) -> dict[str, int]:
    """
    Resolve BS, PV and NPV bands.

    Priority:
    1. Explicit BAND_INDEX_OVERRIDE.
    2. Raster band descriptions and common per-band tags.
    3. Known MESMA order used by the GEE workflow:
       DA, NPV, IS, BS, PV, ...
    4. Three-band fallback:
       BS, PV, NPV.
    """
    result: dict[str, int] = {}

    for key, value in BAND_INDEX_OVERRIDE.items():
        if value is not None:
            result[key] = int(value)

    names: dict[int, str] = {}
    for idx in range(1, src.count + 1):
        candidates = []

        if src.descriptions and len(src.descriptions) >= idx:
            candidates.append(src.descriptions[idx - 1])

        tags = src.tags(idx)
        for tag_name in (
            "DESCRIPTION",
            "description",
            "long_name",
            "NAME",
            "name",
            "band_name",
        ):
            if tag_name in tags:
                candidates.append(tags[tag_name])

        normalized = [
            normalize_band_name(value)
            for value in candidates
            if value is not None
        ]
        names[idx] = "|".join(normalized)

    for idx, name in names.items():
        tokens = set(name.split("|"))

        if "BS" not in result and any(
            token in {"BS", "BARESOIL", "BARESOILFRACTION"}
            for token in tokens
        ):
            result["BS"] = idx

        if "NPV" not in result and any(
            token in {
                "NPV",
                "NONPHOTOSYNTHETICVEGETATION",
                "NONPHOTOSYNTHETICVEGETATIONFRACTION",
            }
            for token in tokens
        ):
            result["NPV"] = idx

        if "PV" not in result and any(
            token in {
                "PV",
                "PHOTOSYNTHETICVEGETATION",
                "PHOTOSYNTHETICVEGETATIONFRACTION",
            }
            for token in tokens
        ):
            result["PV"] = idx

    missing = {"BS", "PV", "NPV"} - set(result)

    if missing:
        if src.count >= 5:


            fallback = {"NPV": 2, "BS": 4, "PV": 5}
            for key in list(missing):
                result[key] = fallback[key]
            print(
                "Band descriptions did not fully identify BS/PV/NPV. "
                "Using GEE MESMA fallback DA,NPV,IS,BS,PV.",
                flush=True,
            )
        elif src.count == 3:
            fallback = {"BS": 1, "PV": 2, "NPV": 3}
            for key in list(missing):
                result[key] = fallback[key]
            print(
                "Using three-band fallback BS=1, PV=2, NPV=3.",
                flush=True,
            )
        else:
            raise RuntimeError(
                "Could not identify BS, PV and NPV bands. "
                f"Raster has {src.count} bands. "
                "Set BAND_INDEX_OVERRIDE at the top of the script."
            )

    print(
        "Resolved MESMA bands:",
        f"BS={result['BS']},",
        f"PV={result['PV']},",
        f"NPV={result['NPV']}",
        flush=True,
    )
    return result


def infer_to_percentage_point_factor(
    src: rasterio.io.DatasetReader,
    bands: dict[str, int],
) -> float:
    """
    Infer how stored MESMA fractions should be converted to percentage points.

    Supported storage:
    0 to 1      -> multiply by 100
    0 to 100    -> multiply by 1
    0 to 10000  -> multiply by 0.01
    """
    samples = []

    sample_h = min(src.height, 160)
    sample_w = min(src.width, 160)

    for key in ("BS", "PV", "NPV"):
        arr = src.read(
            bands[key],
            out_shape=(sample_h, sample_w),
            resampling=Resampling.nearest,
        ).astype(np.float32)

        if src.nodata is not None:
            arr[arr == src.nodata] = np.nan

        arr = arr[np.isfinite(arr)]
        if arr.size:
            samples.append(arr)

    if not samples:
        raise RuntimeError("Could not infer MESMA scale because sample is empty.")

    values = np.concatenate(samples)
    q99 = float(np.nanpercentile(np.abs(values), 99))

    if q99 <= 1.5:
        factor = 100.0
        storage = "0 to 1 fractions"
    elif q99 <= 110.0:
        factor = 1.0
        storage = "0 to 100 percentage points"
    elif q99 <= 11000.0:
        factor = 0.01
        storage = "0 to 10000 scaled fractions"
    else:
        raise RuntimeError(
            f"Unexpected MESMA value scale. Sample 99th percentile = {q99:.3f}"
        )

    print(
        f"MESMA storage detected as {storage}; "
        f"conversion factor to percentage points = {factor}.",
        flush=True,
    )
    return factor


def validate_annual_rasters() -> None:
    missing = [str(annual_raster(year)) for year in ALL_YEARS if not annual_raster(year).exists()]
    if missing:
        raise FileNotFoundError(
            "Missing annual MESMA rasters:\n" + "\n".join(missing)
        )

    with rasterio.open(annual_raster(ALL_YEARS[0])) as ref:
        ref_shape = (ref.height, ref.width)
        ref_transform = ref.transform
        ref_crs = ref.crs

    for year in ALL_YEARS[1:]:
        with rasterio.open(annual_raster(year)) as src:
            if (src.height, src.width) != ref_shape:
                raise ValueError(f"Grid shape differs in {year}.")
            if src.transform != ref_transform:
                raise ValueError(f"Grid transform differs in {year}.")
            if src.crs != ref_crs:
                raise ValueError(f"CRS differs in {year}.")


def period_mask(
    event_year: np.ndarray,
    start: int,
    end: int,
) -> np.ndarray:
    if start == 2001 and end == 2024:
        return event_year > 0
    return (event_year >= start) & (event_year <= end)


def exclusion_mask_for_window(
    ref: rasterio.io.DatasetReader,
    window: Window,
    union_mask_src: rasterio.io.DatasetReader | None,
) -> np.ndarray:
    h = int(window.height)
    w = int(window.width)
    transform = ref.window_transform(window)

    if EXCLUSION_MODE == "none":
        return np.zeros((h, w), dtype=bool)

    if EXCLUSION_MODE == "water_union_2001_2024":
        if union_mask_src is None:
            raise RuntimeError("ARAL_MASK_30M is not open.")

        destination = np.zeros((h, w), dtype=np.uint8)

        reproject(
            source=rasterio.band(union_mask_src, 1),
            destination=destination,
            src_transform=union_mask_src.transform,
            src_crs=union_mask_src.crs,
            dst_transform=transform,
            dst_crs=ref.crs,
            src_nodata=0,
            dst_nodata=0,
            resampling=Resampling.max,
            num_threads=4,
        )
        return destination > 0

    raise RuntimeError("Unexpected exclusion mode.")


def degradation_components_from_baseline(
    baseline_bs: np.ndarray,
    baseline_pv: np.ndarray,
    baseline_npv: np.ndarray,
    target_bs: np.ndarray,
    target_pv: np.ndarray,
    target_npv: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return the three degradation dimensions in percentage points.

    Bare soil exposure
        positive BS increase.

    Vegetation senescence
        the part of PV loss that is concurrently matched by NPV increase.

    Green vegetation loss
        residual PV loss after removing the PV-to-NPV senescence component.

    The PV loss is partitioned into:
        green vegetation loss + vegetation senescence
    so these two vegetation dimensions do not double-count the same PV decline.
    """
    bs_exposure = np.maximum(target_bs - baseline_bs, 0.0)

    raw_pv_loss = np.maximum(baseline_pv - target_pv, 0.0)
    raw_npv_gain = np.maximum(target_npv - baseline_npv, 0.0)

    senescence = np.minimum(raw_pv_loss, raw_npv_gain)
    green_loss = np.maximum(raw_pv_loss - senescence, 0.0)

    return (
        bs_exposure.astype(np.float32),
        green_loss.astype(np.float32),
        senescence.astype(np.float32),
    )


def detect_persistent_mesma_event(
    bs_stack: np.ndarray,
    pv_stack: np.ndarray,
    npv_stack: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Detect the first persistent degradation event per pixel.

    Input arrays have shape:
        year, row, column

    Output:
        event_year
        event_bs
        event_pv
        event_senescence

    event_bs, event_pv and event_senescence store the mean three-year
    post-onset signal relative to the two-year pre-onset baseline.
    """
    n_years, h, w = bs_stack.shape

    event_year = np.zeros((h, w), dtype=np.uint16)
    event_bs = np.full((h, w), np.nan, dtype=np.float32)
    event_pv = np.full((h, w), np.nan, dtype=np.float32)
    event_sen = np.full((h, w), np.nan, dtype=np.float32)

    first_idx = SOURCE_STABLE_YEARS
    last_idx_exclusive = n_years - TARGET_PERSIST_YEARS + 1

    for idx in range(first_idx, last_idx_exclusive):
        transition_year = ALL_YEARS[idx]

        pre_slice = slice(idx - SOURCE_STABLE_YEARS, idx)
        target_slice = slice(idx, idx + TARGET_PERSIST_YEARS)

        pre_bs = bs_stack[pre_slice]
        pre_pv = pv_stack[pre_slice]
        pre_npv = npv_stack[pre_slice]

        target_bs = bs_stack[target_slice]
        target_pv = pv_stack[target_slice]
        target_npv = npv_stack[target_slice]

        pre_valid = (
            np.all(np.isfinite(pre_bs), axis=0)
            & np.all(np.isfinite(pre_pv), axis=0)
            & np.all(np.isfinite(pre_npv), axis=0)
        )

        target_valid = (
            np.all(np.isfinite(target_bs), axis=0)
            & np.all(np.isfinite(target_pv), axis=0)
            & np.all(np.isfinite(target_npv), axis=0)
        )

        if SOURCE_STABLE_YEARS == 2:
            source_stable = (
                np.abs(pre_bs[1] - pre_bs[0]) <= SOURCE_STABILITY_PP
            ) & (
                np.abs(pre_pv[1] - pre_pv[0]) <= SOURCE_STABILITY_PP
            ) & (
                np.abs(pre_npv[1] - pre_npv[0]) <= SOURCE_STABILITY_PP
            )
        else:
            source_stable = (
                (np.nanmax(pre_bs, axis=0) - np.nanmin(pre_bs, axis=0) <= SOURCE_STABILITY_PP)
                & (np.nanmax(pre_pv, axis=0) - np.nanmin(pre_pv, axis=0) <= SOURCE_STABILITY_PP)
                & (np.nanmax(pre_npv, axis=0) - np.nanmin(pre_npv, axis=0) <= SOURCE_STABILITY_PP)
            )

        baseline_bs = np.nanmean(pre_bs, axis=0)
        baseline_pv = np.nanmean(pre_pv, axis=0)
        baseline_npv = np.nanmean(pre_npv, axis=0)

        target_components = []

        for offset in range(TARGET_PERSIST_YEARS):
            comps = degradation_components_from_baseline(
                baseline_bs,
                baseline_pv,
                baseline_npv,
                target_bs[offset],
                target_pv[offset],
                target_npv[offset],
            )
            target_components.append(np.stack(comps, axis=0))


        target_components = np.stack(target_components, axis=1)

        mean_components = np.nanmean(target_components, axis=1)
        dominant_idx = np.argmax(mean_components, axis=0)
        dominant_strength = np.max(mean_components, axis=0)


        selected_target = np.take_along_axis(
            target_components,
            dominant_idx[None, None, :, :],
            axis=0,
        )[0]

        retain_threshold = np.maximum(
            RETAIN_FLOOR_PP,
            RETAIN_FRACTION * dominant_strength,
        )

        target_persistent = np.all(
            selected_target >= retain_threshold[None, :, :],
            axis=0,
        )

        candidate = (
            (event_year == 0)
            & pre_valid
            & target_valid
            & source_stable
            & (dominant_strength >= MIN_ONSET_COMPONENT_PP)
            & target_persistent
        )

        if not candidate.any():
            continue


        recent_start_year = max(RECENT_STATE_START, transition_year)
        recent_years = [
            year
            for year in ALL_YEARS
            if recent_start_year <= year <= RECENT_STATE_END
        ]

        if len(recent_years) < RECENT_STATE_MIN_OBS:
            continue

        recent_indices = [ALL_YEARS.index(year) for year in recent_years]
        required = max(
            RECENT_STATE_MIN_OBS,
            int(math.ceil(RECENT_STATE_FRACTION * len(recent_years))),
        )

        recent_confirm_count = np.zeros((h, w), dtype=np.uint8)
        recent_valid_count = np.zeros((h, w), dtype=np.uint8)

        for recent_idx in recent_indices:
            recent_valid = (
                np.isfinite(bs_stack[recent_idx])
                & np.isfinite(pv_stack[recent_idx])
                & np.isfinite(npv_stack[recent_idx])
            )

            recent_components = np.stack(
                degradation_components_from_baseline(
                    baseline_bs,
                    baseline_pv,
                    baseline_npv,
                    bs_stack[recent_idx],
                    pv_stack[recent_idx],
                    npv_stack[recent_idx],
                ),
                axis=0,
            )

            selected_recent = np.take_along_axis(
                recent_components,
                dominant_idx[None, :, :],
                axis=0,
            )[0]

            recent_valid_count += recent_valid.astype(np.uint8)

            recent_confirm_count += (
                recent_valid
                & (selected_recent >= retain_threshold)
            ).astype(np.uint8)

        recent_ok = (
            (recent_valid_count >= required)
            & (recent_confirm_count >= required)
        )

        final_candidate = candidate & recent_ok

        if not final_candidate.any():
            continue

        event_year[final_candidate] = transition_year

        event_bs[final_candidate] = mean_components[0][final_candidate]
        event_pv[final_candidate] = mean_components[1][final_candidate]
        event_sen[final_candidate] = mean_components[2][final_candidate]

    return event_year, event_bs, event_pv, event_sen


def geographic_row_areas_km2(
    transform: rasterio.Affine,
    height: int,
) -> np.ndarray:
    radius = 6371008.8

    dx = abs(transform.a)
    dy = abs(transform.e)

    ys = transform.f + (np.arange(height) + 0.5) * transform.e

    top = np.radians(ys + dy / 2.0)
    bottom = np.radians(ys - dy / 2.0)

    return (
        radius
        * radius
        * math.radians(dx)
        * (np.sin(top) - np.sin(bottom))
        / 1_000_000.0
    )


def pixel_area_weights_km2(
    ref: rasterio.io.DatasetReader,
    window: Window,
) -> np.ndarray:
    h = int(window.height)
    w = int(window.width)
    transform = ref.window_transform(window)

    if ref.crs is not None and ref.crs.is_geographic:
        row_areas = geographic_row_areas_km2(transform, h)
        return np.broadcast_to(row_areas[:, None], (h, w))

    pixel_area = abs(
        transform.a * transform.e - transform.b * transform.d
    ) / 1_000_000.0

    return np.full((h, w), pixel_area, dtype=np.float64)


def analysis_signature() -> dict:
    return {
        "years": [ALL_YEARS[0], ALL_YEARS[-1]],
        "source_stable_years": SOURCE_STABLE_YEARS,
        "target_persist_years": TARGET_PERSIST_YEARS,
        "recent_start": RECENT_STATE_START,
        "recent_end": RECENT_STATE_END,
        "recent_fraction": RECENT_STATE_FRACTION,
        "recent_min_obs": RECENT_STATE_MIN_OBS,
        "source_stability_pp": SOURCE_STABILITY_PP,
        "min_onset_component_pp": MIN_ONSET_COMPONENT_PP,
        "retain_fraction": RETAIN_FRACTION,
        "retain_floor_pp": RETAIN_FLOOR_PP,
        "exclusion_mode": EXCLUSION_MODE,
        "component_definition": "BS_gain; residual_PV_loss; min(PV_loss,NPV_gain)",
    }


def cache_matches(marker: Path) -> bool:
    if not marker.exists():
        return False

    try:
        saved = json.loads(marker.read_text(encoding="utf-8"))
    except Exception:
        return False

    return saved == analysis_signature()


def output_paths() -> tuple[Path, Path, Path]:
    event_year_path = OUTDIR / "Figure5_persistent_event_year.tif"
    component_path = OUTDIR / "Figure5_persistent_event_components.tif"
    marker = OUTDIR / "Figure5_persistent_analysis_signature.json"
    return event_year_path, component_path, marker


def process_native_grid() -> tuple[Path, Path, pd.DataFrame]:
    OUTDIR.mkdir(parents=True, exist_ok=True)

    event_year_path, component_path, marker = output_paths()

    summary_csv = OUTDIR / "Figure5_persistent_summary_by_period.csv"

    if (
        not FORCE_RECOMPUTE
        and event_year_path.exists()
        and component_path.exists()
        and summary_csv.exists()
        and cache_matches(marker)
    ):
        print("Using cached persistent MESMA outputs.", flush=True)
        return event_year_path, component_path, pd.read_csv(summary_csv)

    validate_annual_rasters()

    basin = gpd.read_file(BASIN_SHP)

    with rasterio.open(annual_raster(ALL_YEARS[0])) as ref0:
        if basin.crs is None:
            basin = basin.set_crs("EPSG:4326")
        basin = basin.to_crs(ref0.crs)

    basin_geometries = [
        geom
        for geom in basin.geometry
        if geom is not None and not geom.is_empty
    ]

    region_name_col = (
        "SUBREGNAME"
        if "SUBREGNAME" in basin.columns
        else basin.columns[0]
    )

    region_records = []
    for region_id, (_, row) in enumerate(basin.iterrows(), start=1):
        region_records.append(
            (
                region_id,
                str(row[region_name_col]),
                row.geometry,
            )
        )

    with ExitStack() as stack:
        datasets = {
            year: stack.enter_context(rasterio.open(annual_raster(year)))
            for year in ALL_YEARS
        }

        ref = datasets[ALL_YEARS[0]]

        bands = resolve_band_indices(ref)
        factor = infer_to_percentage_point_factor(ref, bands)

        union_mask_src = None
        if EXCLUSION_MODE == "water_union_2001_2024":
            if not ARAL_MASK_30M.exists():
                raise FileNotFoundError(ARAL_MASK_30M)
            union_mask_src = stack.enter_context(
                rasterio.open(ARAL_MASK_30M)
            )

        year_profile = ref.profile.copy()
        year_profile.update(
            driver="GTiff",
            count=1,
            dtype="uint16",
            nodata=0,
            compress="DEFLATE",
            tiled=True,
            BIGTIFF="YES",
        )

        component_profile = ref.profile.copy()
        component_profile.update(
            driver="GTiff",
            count=3,
            dtype="float32",
            nodata=-9999.0,
            compress="DEFLATE",
            tiled=True,
            BIGTIFF="YES",
        )

        event_year_dst = stack.enter_context(
            rasterio.open(event_year_path, "w", **year_profile)
        )
        component_dst = stack.enter_context(
            rasterio.open(component_path, "w", **component_profile)
        )

        component_dst.set_band_description(1, "Bare soil exposure")
        component_dst.set_band_description(2, "Green vegetation loss")
        component_dst.set_band_description(3, "Vegetation senescence")


        period_stats = {
            label: {
                "total_area": 0.0,
                "dominant_area": np.zeros(3, dtype=np.float64),
                "component_signal_area": np.zeros(3, dtype=np.float64),
            }
            for _, _, label in PERIODS
        }

        annual_stats = {
            year: {
                "total_area": 0.0,
                "dominant_area": np.zeros(3, dtype=np.float64),
            }
            for year in ALL_YEARS
        }

        region_stats = {
            (label, region_id): {
                "total_area": 0.0,
                "dominant_area": np.zeros(3, dtype=np.float64),
            }
            for _, _, label in PERIODS
            for region_id, _, _ in region_records
        }

        total_blocks = (
            math.ceil(ref.height / BLOCK_SIZE)
            * math.ceil(ref.width / BLOCK_SIZE)
        )
        block_index = 0

        for row_off in range(0, ref.height, BLOCK_SIZE):
            h = min(BLOCK_SIZE, ref.height - row_off)

            for col_off in range(0, ref.width, BLOCK_SIZE):
                w = min(BLOCK_SIZE, ref.width - col_off)
                window = Window(col_off, row_off, w, h)
                block_index += 1

                transform = ref.window_transform(window)

                basin_mask = rasterize(
                    [(mapping(geom), 1) for geom in basin_geometries],
                    out_shape=(h, w),
                    transform=transform,
                    fill=0,
                    dtype="uint8",
                    all_touched=False,
                ).astype(bool)

                if not basin_mask.any():
                    event_year_dst.write(
                        np.zeros((h, w), dtype=np.uint16),
                        1,
                        window=window,
                    )
                    component_dst.write(
                        np.full((3, h, w), -9999.0, dtype=np.float32),
                        window=window,
                    )
                    continue

                exclusion_mask = exclusion_mask_for_window(
                    ref,
                    window,
                    union_mask_src,
                )

                analysis_mask = basin_mask & (~exclusion_mask)

                region_ids = rasterize(
                    [
                        (mapping(geom), region_id)
                        for region_id, _, geom in region_records
                    ],
                    out_shape=(h, w),
                    transform=transform,
                    fill=0,
                    dtype="uint16",
                    all_touched=False,
                )

                bs_stack = np.full(
                    (len(ALL_YEARS), h, w),
                    np.nan,
                    dtype=np.float32,
                )
                pv_stack = np.full_like(bs_stack, np.nan)
                npv_stack = np.full_like(bs_stack, np.nan)

                for yi, year in enumerate(ALL_YEARS):
                    src = datasets[year]

                    bs = src.read(
                        bands["BS"],
                        window=window,
                    ).astype(np.float32)
                    pv = src.read(
                        bands["PV"],
                        window=window,
                    ).astype(np.float32)
                    npv = src.read(
                        bands["NPV"],
                        window=window,
                    ).astype(np.float32)

                    nodata = src.nodata

                    valid = (
                        np.isfinite(bs)
                        & np.isfinite(pv)
                        & np.isfinite(npv)
                    )

                    if nodata is not None:
                        valid &= (
                            (bs != nodata)
                            & (pv != nodata)
                            & (npv != nodata)
                        )

                    bs *= factor
                    pv *= factor
                    npv *= factor


                    valid &= (
                        (bs >= -1.0)
                        & (bs <= 101.0)
                        & (pv >= -1.0)
                        & (pv <= 101.0)
                        & (npv >= -1.0)
                        & (npv <= 101.0)
                    )

                    valid &= analysis_mask

                    bs_stack[yi] = np.where(valid, bs, np.nan)
                    pv_stack[yi] = np.where(valid, pv, np.nan)
                    npv_stack[yi] = np.where(valid, npv, np.nan)

                (
                    event_year,
                    event_bs,
                    event_pv,
                    event_sen,
                ) = detect_persistent_mesma_event(
                    bs_stack,
                    pv_stack,
                    npv_stack,
                )

                event_year_dst.write(
                    event_year,
                    1,
                    window=window,
                )

                out_components = np.stack(
                    [event_bs, event_pv, event_sen],
                    axis=0,
                ).astype(np.float32)

                write_components = np.where(
                    np.isfinite(out_components),
                    out_components,
                    -9999.0,
                )

                component_dst.write(
                    write_components,
                    window=window,
                )

                weights = pixel_area_weights_km2(ref, window)

                component_stack = np.stack(
                    [event_bs, event_pv, event_sen],
                    axis=0,
                )

                event_valid = (
                    (event_year > 0)
                    & np.all(np.isfinite(component_stack), axis=0)
                )

                dominant = np.argmax(
                    np.where(
                        np.isfinite(component_stack),
                        component_stack,
                        -np.inf,
                    ),
                    axis=0,
                )


                for year in ALL_YEARS:
                    mask = event_valid & (event_year == year)

                    if not mask.any():
                        continue

                    area = float(weights[mask].sum())
                    annual_stats[year]["total_area"] += area

                    for pathway_id in range(3):
                        pathway_mask = mask & (dominant == pathway_id)
                        if pathway_mask.any():
                            annual_stats[year]["dominant_area"][pathway_id] += float(
                                weights[pathway_mask].sum()
                            )


                for start, end, label in PERIODS:
                    mask = event_valid & period_mask(
                        event_year,
                        start,
                        end,
                    )

                    if not mask.any():
                        continue

                    area = float(weights[mask].sum())
                    period_stats[label]["total_area"] += area

                    for pathway_id in range(3):
                        pathway_mask = mask & (dominant == pathway_id)

                        if pathway_mask.any():
                            period_stats[label]["dominant_area"][pathway_id] += float(
                                weights[pathway_mask].sum()
                            )

                        values = component_stack[pathway_id]
                        period_stats[label]["component_signal_area"][pathway_id] += float(
                            np.sum(values[mask] * weights[mask])
                        )


                    for region_id, _, _ in region_records:
                        region_mask = mask & (region_ids == region_id)

                        if not region_mask.any():
                            continue

                        rstats = region_stats[(label, region_id)]
                        rstats["total_area"] += float(
                            weights[region_mask].sum()
                        )

                        for pathway_id in range(3):
                            rm = region_mask & (dominant == pathway_id)
                            if rm.any():
                                rstats["dominant_area"][pathway_id] += float(
                                    weights[rm].sum()
                                )

                if block_index % 10 == 0 or block_index == total_blocks:
                    print(
                        f"Persistent MESMA analysis: "
                        f"block {block_index}/{total_blocks}",
                        flush=True,
                    )


    period_rows = []

    for _, _, label in PERIODS:
        stats = period_stats[label]
        total_area = stats["total_area"]

        row = {
            "period": label,
            "persistent_event_area_km2": total_area,
        }

        for pathway_id in range(3):
            key = PATHWAY_KEYS[pathway_id]
            area = stats["dominant_area"][pathway_id]

            row[f"{key}_dominant_area_km2"] = area
            row[f"{key}_dominant_share_pct"] = (
                100.0 * area / total_area
                if total_area > 0
                else np.nan
            )

            signal_area = stats["component_signal_area"][pathway_id]
            row[f"{key}_area_weighted_mean_signal_pp"] = (
                signal_area / total_area
                if total_area > 0
                else np.nan
            )

        period_rows.append(row)

    period_frame = pd.DataFrame(period_rows)

    period_frame.to_csv(
        summary_csv,
        index=False,
        encoding="utf-8-sig",
    )


    annual_rows = []

    for year in ALL_YEARS:
        stats = annual_stats[year]
        total_area = stats["total_area"]

        row = {
            "year": year,
            "persistent_event_area_km2": total_area,
        }

        for pathway_id in range(3):
            key = PATHWAY_KEYS[pathway_id]
            area = stats["dominant_area"][pathway_id]

            row[f"{key}_dominant_area_km2"] = area
            row[f"{key}_dominant_share_pct"] = (
                100.0 * area / total_area
                if total_area > 0
                else np.nan
            )

        annual_rows.append(row)

    pd.DataFrame(annual_rows).to_csv(
        OUTDIR / "Figure5_persistent_summary_by_year.csv",
        index=False,
        encoding="utf-8-sig",
    )


    region_rows_out = []

    region_name_lookup = {
        region_id: name
        for region_id, name, _ in region_records
    }

    for _, _, label in PERIODS:
        for region_id, _, _ in region_records:
            stats = region_stats[(label, region_id)]
            total_area = stats["total_area"]

            row = {
                "period": label,
                "region": region_name_lookup[region_id],
                "persistent_event_area_km2": total_area,
            }

            for pathway_id in range(3):
                key = PATHWAY_KEYS[pathway_id]
                area = stats["dominant_area"][pathway_id]

                row[f"{key}_dominant_area_km2"] = area
                row[f"{key}_dominant_share_pct"] = (
                    100.0 * area / total_area
                    if total_area > 0
                    else np.nan
                )

            region_rows_out.append(row)

    pd.DataFrame(region_rows_out).to_csv(
        OUTDIR / "Figure5_persistent_summary_by_subbasin.csv",
        index=False,
        encoding="utf-8-sig",
    )

    marker.write_text(
        json.dumps(
            analysis_signature(),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return event_year_path, component_path, period_frame


def full_resolution_strength_p95(
    component_path: Path,
) -> float:
    """
    Approximate the full-resolution 95th percentile of total event strength.

    A fine histogram avoids loading the complete raster into memory.
    """
    bin_edges = np.linspace(0.0, 300.0, 6001)
    histogram = np.zeros(len(bin_edges) - 1, dtype=np.int64)

    with rasterio.open(component_path) as src:
        nodata = src.nodata

        for _, window in src.block_windows(1):
            arr = src.read(
                [1, 2, 3],
                window=window,
            ).astype(np.float32)

            valid = np.all(np.isfinite(arr), axis=0)

            if nodata is not None:
                valid &= np.all(arr != nodata, axis=0)

            total = np.sum(arr, axis=0)

            values = total[
                valid
                & np.isfinite(total)
                & (total > 0)
            ]

            if values.size:
                hist, _ = np.histogram(
                    np.clip(values, 0, 300),
                    bins=bin_edges,
                )
                histogram += hist

    count = histogram.sum()

    if count == 0:
        raise RuntimeError("No persistent MESMA events were detected.")

    threshold_count = 0.95 * count
    cumulative = np.cumsum(histogram)
    idx = int(np.searchsorted(cumulative, threshold_count))

    idx = min(max(idx, 0), len(bin_edges) - 2)

    return float((bin_edges[idx] + bin_edges[idx + 1]) / 2.0)


def display_shape(src: rasterio.io.DatasetReader) -> tuple[int, int]:
    scale = min(
        1.0,
        DISPLAY_MAX_DIM / max(src.height, src.width),
    )

    h = max(1, int(round(src.height * scale)))
    w = max(1, int(round(src.width * scale)))

    return h, w


def load_display_event_arrays(
    event_year_path: Path,
    component_path: Path,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    tuple[float, float, float, float],
]:
    with rasterio.open(event_year_path) as ysrc, rasterio.open(component_path) as csrc:
        h, w = display_shape(ysrc)

        event_year = ysrc.read(
            1,
            out_shape=(h, w),
            resampling=Resampling.nearest,
        )

        components = csrc.read(
            [1, 2, 3],
            out_shape=(3, h, w),
            resampling=Resampling.nearest,
        ).astype(np.float32)

        nodata = csrc.nodata
        if nodata is not None:
            components[components == nodata] = np.nan

        bounds = ysrc.bounds
        extent = (
            bounds.left,
            bounds.right,
            bounds.bottom,
            bounds.top,
        )

    return (
        event_year,
        components[0],
        components[1],
        components[2],
        extent,
    )


def rgb_from_components(
    bs: np.ndarray,
    pv: np.ndarray,
    sen: np.ndarray,
    strength_p95: float,
    event_mask: np.ndarray,
) -> np.ndarray:
    total = bs + pv + sen

    valid = (
        event_mask
        & np.isfinite(bs)
        & np.isfinite(pv)
        & np.isfinite(sen)
        & np.isfinite(total)
        & (total > 0)
    )

    weights = np.zeros(
        (3, *total.shape),
        dtype=np.float32,
    )

    weights[0, valid] = bs[valid] / total[valid]
    weights[1, valid] = pv[valid] / total[valid]
    weights[2, valid] = sen[valid] / total[valid]

    hue = (
        weights[0, :, :, None] * COLORS["BS"]
        + weights[1, :, :, None] * COLORS["PV"]
        + weights[2, :, :, None] * COLORS["NPV"]
    )

    strength = np.zeros_like(total, dtype=np.float32)

    strength[valid] = np.clip(
        total[valid] / strength_p95,
        0.0,
        1.0,
    )


    strength = np.sqrt(strength)

    base = np.array(
        [0.96, 0.96, 0.94],
        dtype=np.float32,
    )

    rgb = (
        base * (1.0 - strength[:, :, None])
        + hue * strength[:, :, None]
    )

    rgb[~valid] = 1.0

    return np.clip(rgb, 0.0, 1.0)


def ternary_key(
    ax: plt.Axes,
    strength_p95: float,
) -> None:
    n = 240

    image = np.ones((n, n, 4), dtype=np.float32)

    yy, xx = np.mgrid[0:n, 0:n]
    x = xx / (n - 1)
    h = np.sqrt(3.0) / 2.0
    y = yy / (n - 1) * h

    top = np.array([0.5, h])
    left = np.array([0.0, 0.0])
    right = np.array([1.0, 0.0])

    denom = (
        (left[1] - right[1]) * (top[0] - right[0])
        + (right[0] - left[0]) * (top[1] - right[1])
    )

    w_bs = (
        (left[1] - right[1]) * (x - right[0])
        + (right[0] - left[0]) * (y - right[1])
    ) / denom

    w_pv = (
        (right[1] - top[1]) * (x - right[0])
        + (top[0] - right[0]) * (y - right[1])
    ) / denom

    w_sen = 1.0 - w_bs - w_pv

    inside = (
        (w_bs >= 0)
        & (w_pv >= 0)
        & (w_sen >= 0)
    )

    hue = (
        w_bs[:, :, None] * COLORS["BS"]
        + w_pv[:, :, None] * COLORS["PV"]
        + w_sen[:, :, None] * COLORS["NPV"]
    )

    image[:, :, :3] = np.clip(hue, 0.0, 1.0)
    image[:, :, 3] = inside.astype(np.float32)

    ax.imshow(
        image,
        extent=(0, 1, 0, h),
        origin="lower",
    )

    ax.add_patch(
        Polygon(
            [[0, 0], [1, 0], [0.5, h]],
            fill=False,
            edgecolor="#333333",
            linewidth=0.9,
        )
    )

    ax.text(
        0.5,
        h + 0.04,
        "Bare soil\nexposure",
        ha="center",
        va="bottom",
        fontsize=8.8,
    )
    ax.text(
        0.0,
        -0.04,
        "Green vegetation\nloss",
        ha="center",
        va="top",
        fontsize=8.8,
    )
    ax.text(
        1.0,
        -0.04,
        "Vegetation\nsenescence",
        ha="center",
        va="top",
        fontsize=8.4,
    )

    ax.set_xlim(-0.10, 1.10)
    ax.set_ylim(-0.10, h + 0.16)
    ax.axis("off")


def stacked_summary(
    ax: plt.Axes,
    summary: pd.DataFrame,
) -> None:
    labels = [
        "2001-05",
        "2006-10",
        "2011-15",
        "2016-20",
        "2021-24",
        "2001-24",
    ]

    y = np.arange(len(summary))
    left = np.zeros(len(summary), dtype=float)

    columns = [
        ("BS_dominant_share_pct", "BS", "Bare soil exposure"),
        ("PV_dominant_share_pct", "PV", "Green vegetation loss"),
        ("NPV_dominant_share_pct", "NPV", "Vegetation senescence"),
    ]

    bar_height = 0.66

    for column, key, label in columns:
        values = summary[column].to_numpy(dtype=float)

        ax.barh(
            y,
            values,
            left=left,
            color=COLORS[key],
            edgecolor="white",
            linewidth=0.5,
            height=bar_height,
            label=label,
        )

        for yi, value, start_value in zip(y, values, left):
            if not np.isfinite(value) or value <= 0:
                continue

            x_center = start_value + value / 2.0
            fontsize = 7.1 if value >= 8 else 6.2

            ax.text(
                x_center,
                yi,
                f"{value:.0f}",
                ha="center",
                va="center",
                color="white",
                fontsize=fontsize,
                fontweight="bold",
            )

        left += np.nan_to_num(values, nan=0.0)

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_ylim(len(summary) - 0.45, -1.05)

    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Area share of persistent events (%)", fontsize=10.5)

    ax.grid(
        axis="x",
        color="#D9D9D9",
        linewidth=0.45,
    )
    ax.set_axisbelow(True)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    handles = [
        Patch(
            facecolor=COLORS[key],
            edgecolor="none",
            label=label,
        )
        for _, key, label in columns
    ]

    ax.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.0, 0.985),
        ncol=3,
        frameon=False,
        borderaxespad=0.0,
        handlelength=1.35,
        columnspacing=1.05,
        fontsize=8.4,
    )


def make_figure(
    event_year_path: Path,
    component_path: Path,
    summary: pd.DataFrame,
) -> Path:
    FIGDIR.mkdir(parents=True, exist_ok=True)

    strength_p95 = full_resolution_strength_p95(
        component_path
    )

    (
        event_year,
        event_bs,
        event_pv,
        event_sen,
        map_extent,
    ) = load_display_event_arrays(
        event_year_path,
        component_path,
    )

    bx0, bx1, by0, by1 = mesma_maps.shape_bbox(
        BASIN_SHP
    )

    pad_x = max((bx1 - bx0) * 0.035, 0.25)
    pad_y = max((by1 - by0) * 0.035, 0.25)

    view_extent = (
        bx0 - pad_x,
        bx1 + pad_x,
        by0 - pad_y,
        by1 + pad_y,
    )

    panel_rgbs = []

    for start, end, label in PERIODS:
        mask = period_mask(
            event_year,
            start,
            end,
        )

        rgb = rgb_from_components(
            event_bs,
            event_pv,
            event_sen,
            strength_p95,
            mask,
        )

        panel_rgbs.append(
            (start, end, label, rgb)
        )

    width_in = 183.0 / 25.4
    height_in = 210.0 / 25.4

    fig = plt.figure(
        figsize=(width_in, height_in),
        dpi=180,
    )

    gs = fig.add_gridspec(
        3,
        3,
        height_ratios=[1.0, 1.0, 0.76],
        left=0.065,
        right=0.985,
        top=0.975,
        bottom=0.07,
        hspace=0.06,
        wspace=0.05,
    )

    letters = list("abcdef")

    for index, (
        letter,
        (start, end, label, rgb),
    ) in enumerate(zip(letters, panel_rgbs)):
        row = index // 3
        col = index % 3

        ax = fig.add_subplot(gs[row, col])

        ax.imshow(
            rgb,
            extent=map_extent,
            origin="upper",
            interpolation="nearest",
            zorder=1,
        )

        mesma_maps.style_map_axis(
            ax,
            map_extent,
            view_extent,
            BASIN_SHP,
            COUNTRIES_SHP,
        )
        ax.tick_params(axis="both", labelsize=9.5)

        display_title = f"{start}-{end}"

        ax.text(
            0.11,
            1.01,
            display_title,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=10,
        )

        ax.text(
            0.0,
            1.01,
            f"({letter})",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

        if col != 0:
            ax.tick_params(
                axis="y",
                labelleft=False,
            )

        if row == 0:
            ax.tick_params(
                axis="x",
                labelbottom=False,
            )

        if row == 1:
            position = ax.get_position()

            ax.set_position(
                [
                    position.x0,
                    position.y0 + 0.04,
                    position.width,
                    position.height,
                ]
            )

    bar_ax = fig.add_subplot(gs[2, :2])

    bar_pos = bar_ax.get_position()

    bar_ax.set_position(
        [
            bar_pos.x0 + 0.040,
            bar_pos.y0,
            bar_pos.width - 0.030,
            bar_pos.height,
        ]
    )

    stacked_summary(
        bar_ax,
        summary,
    )

    bar_ax.text(
        0.0,
        1.02,
        "(g)",
        transform=bar_ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10,
        fontweight="bold",
    )

    bar_ax.text(
        0.08,
        1.02,
        "Dominant persistent degradation pathway by onset period",
        transform=bar_ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9.4,
    )

    key_ax = fig.add_subplot(gs[2, 2])
    key_pos = key_ax.get_position()
    key_ax.set_position(
        [
            key_pos.x0 + 0.045,
            key_pos.y0 + 0.035,
            key_pos.width - 0.090,
            key_pos.height - 0.085,
        ]
    )

    ternary_key(
        key_ax,
        strength_p95,
    )

    stem = (
        FIGDIR
        / "Figure5_MESMA_persistent_degradation_ternary_excluding_AralUnion"
    )

    fig.savefig(
        stem.with_suffix(".png"),
        dpi=300,
    )

    fig.savefig(
        stem.with_suffix(".tiff"),
        dpi=600,
        pil_kwargs={"compression": "tiff_lzw"},
    )

    fig.savefig(
        stem.with_suffix(".pdf"),
    )

    fig.savefig(
        stem.with_suffix(".svg"),
    )

    plt.close(fig)

    return stem


def write_method_report(
    summary: pd.DataFrame,
    figure_stem: Path,
) -> None:
    report = f"""
# Figure 5 persistent MESMA degradation analysis

## Persistent-event rule

Source stability:
The two annual observations immediately before candidate onset must differ by
no more than {SOURCE_STABILITY_PP:.1f} percentage points for each of BS, PV and NPV.

Initial persistence:
The candidate onset and the next {TARGET_PERSIST_YEARS - 1} years are compared
with the mean of the two pre-onset years. The mean dominant degradation
component must be at least {MIN_ONSET_COMPONENT_PP:.1f} percentage points.
In every target year, that pathway must retain at least
max({RETAIN_FLOOR_PP:.1f} percentage points,
{RETAIN_FRACTION:.0%} of its initial mean magnitude).

Recent-state confirmation:
Using post-onset observations within {RECENT_STATE_START} to {RECENT_STATE_END},
at least {RECENT_STATE_FRACTION:.0%} of available years and at least
{RECENT_STATE_MIN_OBS} observations must retain the same dominant pathway.

First event:
Only the first qualifying persistent event is retained per pixel.

## Three ternary dimensions

Bare soil exposure:
positive BS increase relative to the two-year pre-onset baseline.

Vegetation senescence:
the concurrent part of PV loss and NPV increase,
min(PV loss, NPV gain).

Green vegetation loss:
the residual PV loss after the senescence component has been removed.

## Temporal censoring

With a two-year source period and a three-year target period,
the directly detectable onset window is 2003 to 2022.

## Spatial exclusion

EXCLUSION_MODE = {EXCLUSION_MODE}
The current script setting uses ARAL_MASK_30M, that is the 2001-2024 water-union mask generated in the Figure 3 workflow.

## Figure

{figure_stem}

## Native-grid quantitative summary

{summary.to_csv(index=False)}
""".strip()

    (OUTDIR / "Figure5_persistent_method_report.md").write_text(
        report + "\n",
        encoding="utf-8",
    )


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    FIGDIR.mkdir(parents=True, exist_ok=True)

    event_year_path, component_path, summary = process_native_grid()

    figure_stem = make_figure(
        event_year_path,
        component_path,
        summary,
    )

    write_method_report(
        summary,
        figure_stem,
    )

    print("", flush=True)
    print("Persistent MESMA analysis completed.", flush=True)
    print(f"Figure: {figure_stem}", flush=True)
    print("", flush=True)
    print("Period statistics:", flush=True)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
