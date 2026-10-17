from __future__ import annotations

"""Reclassify Figure 3 degradation-related transitions with two temporal filters.

The script uses the complete annual FROMGLC30 record from 2001 to 2024. A
transition is accepted only when (1) the source class is stable immediately
before the change, (2) the target class persists for several consecutive years
after the change, and (3) the recent 2020-2024 state confirms that the change
did not subsequently bounce back.

For terrestrial transitions, recent-state confirmation is deliberately
conservative: the original target class must remain dominant in the recent
post-transition observations. For water-loss transitions, persistence is defined
by continued absence of open water because an exposed lakebed may later be
colonized by terrestrial vegetation while remaining hydrologically exposed.

Area summaries are calculated from every valid 30 m FROMGLC30 pixel in streamed
blocks. Map panels are display-sampled at about 2.7 km and are never used for
area statistics.
"""

import math
import re
from collections import defaultdict
from pathlib import Path

import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import rasterio.warp
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter
from rasterio.features import rasterize
from rasterio.merge import merge
from rasterio.windows import Window
from shapely.geometry import box, mapping


# -----------------------------------------------------------------------------
# Standalone map helpers
# -----------------------------------------------------------------------------
# The earlier Figure 3 script imported helper functions from
# ``aral_fromglc30_transition_composite.py``.  This persistence version is
# intentionally self-contained so it can run directly from the tools folder.

COUNTRY_BOUNDARY_PATH = None  # Optional: set to a country-boundary shapefile if desired.

def pretty_region(value: object) -> str:
    """Return a clean display name for a subbasin label."""
    if value is None:
        return "Unknown"
    raw = str(value).strip()
    # Remove suffixes such as ``(ex 2161)`` that are present in some
    # SUBREGNAME values, then normalize spacing/case for plotting and tables.
    name = re.sub(r"\s*\(ex\s*\d+\)\s*$", "", raw, flags=re.IGNORECASE).strip()
    key = re.sub(r"[^A-Za-z]+", " ", name).strip().lower()
    replacements = {
        "amu darya": "Amu Darya",
        "amudarya": "Amu Darya",
        "syr darya": "Syr Darya",
        "syrdarya": "Syr Darya",
        "aral sea south": "Aral Sea South",
        "south aral sea": "Aral Sea South",
        "aral sea north": "Aral Sea North",
        "north aral sea": "Aral Sea North",
    }
    return replacements.get(key, name.title())


def read_shapes(path: Path) -> gpd.GeoDataFrame:
    """Read vector boundaries and return them in EPSG:4326."""
    frame = gpd.read_file(path)
    if frame.crs is None:
        frame = frame.set_crs("EPSG:4326")
    return frame.to_crs("EPSG:4326")


def map_grid_geometry(
    reference_path: Path,
    shapes: gpd.GeoDataFrame,
    step: int,
) -> tuple[tuple[float, float, float, float], np.ndarray]:
    """Build the common display-grid extent and basin mask.

    The quantitative statistics remain at the native 30 m resolution.  This
    helper is used only for the display-sampled map panels.
    """
    xmin, ymin, xmax, ymax = shapes.total_bounds
    with rasterio.open(reference_path) as ref:
        dx = abs(ref.transform.a) * step
        dy = abs(ref.transform.e) * step

    # Snap the plotting extent to the display grid so tile placement and the
    # basin mask share exactly the same dimensions.
    xmin = math.floor(xmin / dx) * dx
    ymin = math.floor(ymin / dy) * dy
    xmax = math.ceil(xmax / dx) * dx
    ymax = math.ceil(ymax / dy) * dy
    width = int(np.ceil((xmax - xmin) / dx))
    height = int(np.ceil((ymax - ymin) / dy))
    transform = rasterio.transform.from_origin(xmin, ymax, dx, dy)

    geoms = [
        (mapping(geom), 1)
        for geom in shapes.geometry
        if geom is not None and not geom.is_empty
    ]
    basin_mask = rasterize(
        geoms,
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype="uint8",
        all_touched=False,
    ).astype(bool)
    return (xmin, ymin, xmax, ymax), basin_mask


def _degree_e(value: float, _pos: int | None = None) -> str:
    return f"{value:.0f}°E"


def _degree_n(value: float, _pos: int | None = None) -> str:
    return f"{value:.0f}°N"


def draw_boundaries_and_grid(
    ax: plt.Axes,
    shapes: gpd.GeoDataFrame,
    extent: tuple[float, float, float, float],
    country_shapes: gpd.GeoDataFrame | None = None,
) -> None:
    """Draw basin/subbasin boundaries, optional country boundaries and labels."""
    xmin, ymin, xmax, ymax = extent

    if country_shapes is not None and not country_shapes.empty:
        country_shapes.boundary.plot(
            ax=ax, color="#c9d4df", linewidth=0.35, zorder=1
        )

    shapes.boundary.plot(ax=ax, color="#222222", linewidth=0.65, zorder=5)

    name_col = "SUBREGNAME" if "SUBREGNAME" in shapes.columns else None
    if name_col is not None:
        for _, row in shapes.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue
            p = geom.representative_point()
            label = pretty_region(row[name_col])
            # Match the compact line wrapping used in the existing figure.
            if label in {"Aral Sea North", "Aral Sea South"}:
                label = label.replace(" Sea ", " Sea\n")
            ax.text(
                p.x, p.y, label,
                ha="center", va="center",
                fontsize=6.5, color="#222222", zorder=6,
            )

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(5))
    ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(5))
    ax.xaxis.set_major_formatter(FuncFormatter(_degree_e))
    ax.yaxis.set_major_formatter(FuncFormatter(_degree_n))
    ax.grid(True, linestyle="--", linewidth=0.35, color="#cfcfcf", zorder=0)


ROOT = Path(__file__).resolve().parents[2]
FROMGLC_DIR = ROOT / "data/raw/FROMGLC30_V002"
SUBBASINS = ROOT / "data/boundaries/AralSeaBasinS.shp"
# The archived final water-union mask is the input to downstream land analyses.
# Its upstream delineation is outside this compact reproducibility package.
OUTDIR = ROOT / "outputs/full/figure3"
FIGDIR = ROOT / "outputs/full/figure3"
ARAL_MASK_30M = ROOT / "data/figure_source/aral_water_union_30m.tif"

ALL_YEARS = list(range(2001, 2025))
PERIODS = [
    (2001, 2005, "2001-2005"),
    (2006, 2010, "2006-2010"),
    (2011, 2015, "2011-2015"),
    (2016, 2020, "2016-2020"),
    (2021, 2024, "2021-2024"),
    (2001, 2024, "2001-2024"),
]

# Initial temporal-persistence rule. A candidate change in year t is retained
# only if the source class occurred in the two immediately preceding years and
# the target class remains unchanged in t, t+1, and t+2. Therefore the earliest
# detectable transition year is 2003 and the latest is 2022.
SOURCE_STABLE_YEARS = 2
TARGET_PERSIST_YEARS = 3

# Recent-state confirmation. For transitions established on or before 2020, at
# least 4 of the five years 2020-2024 must confirm the persistent state. For
# transitions established in 2021 or 2022, only post-transition years are used;
# at least 80% of the available years must confirm persistence, with a minimum
# of three observations. This avoids automatically rejecting a 2022 transition
# simply because 2025-2026 observations do not yet exist.
RECENT_STATE_START = 2020
RECENT_STATE_END = 2024
RECENT_STATE_FRACTION = 0.80
RECENT_STATE_MIN_OBS = 3

CATEGORY_ORDER = [
    "Natural-cover degradation",
    "Land-use intensification",
    "Hydrological degradation",
    "Aral Sea lakebed exposure",
]
CATEGORY_COLORS = {
    "Natural-cover degradation": "#b2182b",
    "Land-use intensification": "#fdb863",
    "Hydrological degradation": "#ef8a62",
    "Aral Sea lakebed exposure": "#2166ac",
}
CATEGORY_ID = {name: idx + 1 for idx, name in enumerate(CATEGORY_ORDER)}

CLASS_CODES = {
    "Cropland": 10,
    "Forest": 20,
    "Grassland": 30,
    "Shrubland": 40,
    "Wetland": 50,
    "Water": 60,
    "Bare land": 90,
}

TRANSITION_RULES = {
    "Natural-cover degradation": {
        (20, 30),  # Forest -> Grassland
        (40, 30),  # Shrubland -> Grassland
        (20, 90),  # Vegetation -> Bare land
        (30, 90),
        (40, 90),
        (50, 90),  # Wetland -> Bare land
    },
    "Land-use intensification": {
        (30, 10),  # Grassland -> Cropland
        (20, 10),  # Forest -> Cropland
        (50, 10),  # Wetland -> Cropland
    },
}
HYDRO_TARGETS = {90, 30, 10}

BLOCK = 1024
DISPLAY_STEP = 90
WATER_OCCURRENCE_THRESHOLD = 0.0


def tile_key(path: Path) -> str:
    match = re.search(r"(N\d{2}E\d{3})", path.name)
    if match is None:
        raise ValueError(f"Cannot parse FROMGLC tile from {path.name}")
    return match.group(1)


def occurrence_tile_key(path: Path) -> str:
    match = re.search(r"(h\d{2}v\d{2})", path.name)
    if match is None:
        raise ValueError(f"Cannot parse occurrence tile from {path.name}")
    return match.group(1)


def find_fromglc_files() -> dict[int, dict[str, Path]]:
    """Find FROMGLC30 tiles for every year required by the persistence filter."""
    result: dict[int, dict[str, Path]] = {}
    for year in ALL_YEARS:
        paths = sorted((FROMGLC_DIR / str(year)).glob("*.tif"))
        result[year] = {tile_key(path): path for path in paths}
        if not result[year]:
            raise FileNotFoundError(
                f"No FROMGLC30 files found for {year}. "
                "The persistence-filtered analysis requires annual maps for every year 2001-2024."
            )

    common = set.intersection(*(set(result[year]) for year in ALL_YEARS))
    if not common:
        raise RuntimeError("No common FROMGLC30 tiles across all years 2001-2024")

    missing_summary = []
    for year in ALL_YEARS:
        missing = sorted(set(common) - set(result[year]))
        if missing:
            missing_summary.append(f"{year}: {len(missing)} missing common tiles")
    if missing_summary:
        raise RuntimeError("\n".join(missing_summary))

    return {year: {key: result[year][key] for key in sorted(common)} for year in ALL_YEARS}

def row_pixel_areas(transform: rasterio.Affine, height: int) -> np.ndarray:
    radius = 6371008.8
    dx = abs(transform.a)
    dy = abs(transform.e)
    ys = transform.f + (np.arange(height) + 0.5) * transform.e
    top = np.radians(ys + dy / 2)
    bottom = np.radians(ys - dy / 2)
    return radius * radius * math.radians(dx) * (np.sin(top) - np.sin(bottom)) / 1_000_000


def classify_transition_pair(
    arr0: np.ndarray,
    arr1: np.ndarray,
    aral_mask: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    """Classify a source-to-target transition after temporal persistence is confirmed."""
    out = np.zeros(arr0.shape, dtype="uint8")

    for category, pairs in TRANSITION_RULES.items():
        category_mask = np.zeros(arr0.shape, dtype=bool)
        for from_code, to_code in pairs:
            category_mask |= (arr0 == from_code) & (arr1 == to_code)
        out[valid & category_mask] = CATEGORY_ID[category]

    water_loss = valid & (arr0 == CLASS_CODES["Water"]) & np.isin(arr1, list(HYDRO_TARGETS))
    out[water_loss & ~aral_mask] = CATEGORY_ID["Hydrological degradation"]
    out[
        valid
        & aral_mask
        & (arr0 == CLASS_CODES["Water"])
        & (arr1 == CLASS_CODES["Bare land"])
    ] = CATEGORY_ID["Aral Sea lakebed exposure"]
    return out


def recent_state_confirmed(
    stack: np.ndarray,
    transition_idx: int,
    target_arr: np.ndarray,
    categories: np.ndarray,
    nodata: int,
) -> np.ndarray:
    """Confirm that a qualifying transition did not subsequently bounce back.

    Terrestrial categories (natural-cover degradation and land-use
    intensification) use a strict target-state rule: the original target class
    must still occur in the required share of recent post-transition years.

    Water-loss categories use a hydrological rule: recent observations must
    remain non-water. This allows a formerly exposed lakebed to become
    vegetated without being treated as hydrological recovery.

    The confirmation window begins at max(2020, transition year) and ends in
    2024. At least 80% of the available observations must confirm persistence,
    with at least three observations. For older transitions this is equivalent
    to at least 4 of 5 years during 2020-2024.
    """
    shape = target_arr.shape
    confirmed = np.zeros(shape, dtype=bool)

    transition_year = ALL_YEARS[transition_idx]
    start_year = max(RECENT_STATE_START, transition_year)
    recent_years = [y for y in ALL_YEARS if start_year <= y <= RECENT_STATE_END]
    if len(recent_years) < RECENT_STATE_MIN_OBS:
        return confirmed

    required = max(
        RECENT_STATE_MIN_OBS,
        int(math.ceil(RECENT_STATE_FRACTION * len(recent_years))),
    )
    indices = [ALL_YEARS.index(y) for y in recent_years]
    recent = stack[indices, ...]
    valid_count = np.sum(recent != nodata, axis=0)

    terrestrial = (categories == CATEGORY_ID["Natural-cover degradation"]) | (
        categories == CATEGORY_ID["Land-use intensification"]
    )
    if terrestrial.any():
        exact_target_count = np.sum(recent == target_arr[None, ...], axis=0)
        confirmed |= terrestrial & (valid_count >= required) & (exact_target_count >= required)

    water_related = (categories == CATEGORY_ID["Hydrological degradation"]) | (
        categories == CATEGORY_ID["Aral Sea lakebed exposure"]
    )
    if water_related.any():
        nonwater_count = np.sum(
            (recent != nodata) & (recent != CLASS_CODES["Water"]), axis=0
        )
        confirmed |= water_related & (valid_count >= required) & (nonwater_count >= required)

    return confirmed


def detect_persistent_transitions(
    arrays: dict[int, np.ndarray],
    aral_mask: np.ndarray,
    nodata: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Detect the first persistent degradation-related transition per pixel.

    Persistence rule
    ----------------
    For a candidate transition beginning in year t:
      1. the same source class must occur in each of the SOURCE_STABLE_YEARS
         immediately preceding t;
      2. the same target class must occur in t and the following
         TARGET_PERSIST_YEARS - 1 years;
      3. source and target classes must differ;
      4. the source-target pair must match one of the degradation-related rules;
      5. the recent 2020-2024 state must confirm that the transition has not
         subsequently bounced back.

    For terrestrial transitions, recent confirmation requires persistence of the
    same target class. For water-loss transitions, it requires continued absence
    of open water. The first event satisfying both the initial-persistence and
    recent-state criteria is retained. Pixels therefore contribute at most once.
    With the default 2-year source + 3-year target rule, detectable transition
    years are 2003-2022.
    """
    stack = np.stack([arrays[year] for year in ALL_YEARS], axis=0)
    shape = stack.shape[1:]

    event_category = np.zeros(shape, dtype="uint8")
    event_year = np.zeros(shape, dtype="uint16")
    event_from = np.zeros(shape, dtype="int16")
    event_to = np.zeros(shape, dtype="int16")

    first_idx = SOURCE_STABLE_YEARS
    last_idx_exclusive = len(ALL_YEARS) - TARGET_PERSIST_YEARS + 1

    for idx in range(first_idx, last_idx_exclusive):
        source_arr = stack[idx - 1]
        target_arr = stack[idx]

        source_stable = np.ones(shape, dtype=bool)
        for offset in range(1, SOURCE_STABLE_YEARS + 1):
            source_stable &= stack[idx - offset] == source_arr
            source_stable &= stack[idx - offset] != nodata

        target_persistent = np.ones(shape, dtype=bool)
        for offset in range(TARGET_PERSIST_YEARS):
            target_persistent &= stack[idx + offset] == target_arr
            target_persistent &= stack[idx + offset] != nodata

        valid = (source_arr != nodata) & (target_arr != nodata) & (source_arr != target_arr)
        stable_change = valid & source_stable & target_persistent
        if not stable_change.any():
            continue

        categories = classify_transition_pair(source_arr, target_arr, aral_mask, stable_change)
        recent_ok = recent_state_confirmed(stack, idx, target_arr, categories, nodata)
        candidate = (event_category == 0) & (categories > 0) & recent_ok
        if not candidate.any():
            continue

        event_category[candidate] = categories[candidate]
        event_year[candidate] = ALL_YEARS[idx]
        event_from[candidate] = source_arr[candidate].astype("int16")
        event_to[candidate] = target_arr[candidate].astype("int16")

    return event_category, event_year, event_from, event_to


def categories_for_period(
    event_category: np.ndarray, event_year: np.ndarray, year0: int, year1: int
) -> np.ndarray:
    """Return persistent-transition categories whose establishment year falls in a period."""
    if year0 == 2001 and year1 == 2024:
        return event_category
    out = np.zeros(event_category.shape, dtype="uint8")
    mask = (event_year >= year0) & (event_year <= year1)
    out[mask] = event_category[mask]
    return out

def read_aral_mask(mask_src: rasterio.DatasetReader, shape: tuple[int, int], transform: rasterio.Affine) -> np.ndarray:
    height, width = shape
    bounds = rasterio.transform.array_bounds(height, width, transform)
    window = rasterio.windows.from_bounds(*bounds, transform=mask_src.transform)
    destination = mask_src.read(
        1,
        window=window,
        out_shape=shape,
        boundless=True,
        fill_value=0,
        resampling=rasterio.enums.Resampling.max,
    )
    return destination > 0


def region_name(value: object) -> str:
    return pretty_region(value)


def full_resolution_statistics(files: dict[int, dict[str, Path]]) -> pd.DataFrame:
    cache = OUTDIR / "Figure3_persistent_recentstate_transition_areas_full30m.csv"
    cache_marker = OUTDIR / "Figure3_persistent_recentstate_transition_areas_full30m.complete"
    cache_version = (
        f"persistent-v2-recentstate-source{SOURCE_STABLE_YEARS}-target{TARGET_PERSIST_YEARS}"
        f"-recent{RECENT_STATE_START}_{RECENT_STATE_END}-frac{RECENT_STATE_FRACTION:.2f}"
    )
    if (
        cache.exists()
        and cache_marker.exists()
        and cache_marker.read_text(encoding="utf-8").strip() == cache_version
    ):
        print(f"Using cached persistence-filtered statistics: {cache}", flush=True)
        return pd.read_csv(cache)

    basin = gpd.read_file(SUBBASINS)
    if basin.crs is None:
        basin = basin.set_crs("EPSG:4326")
    basin = basin.to_crs("EPSG:4326")
    name_col = "SUBREGNAME" if "SUBREGNAME" in basin.columns else basin.columns[0]
    region_rows = [(idx + 1, region_name(row[name_col]), row.geometry) for idx, (_, row) in enumerate(basin.iterrows())]
    basin_union = basin.geometry.union_all()
    totals: dict[tuple[str, str], float] = defaultdict(float)
    details: dict[tuple[str, str, int, int], float] = defaultdict(float)
    annual_totals: dict[tuple[int, str], float] = defaultdict(float)

    tile_keys = sorted(files[ALL_YEARS[0]])
    with rasterio.open(ARAL_MASK_30M) as aral_src:
        for tile_index, key in enumerate(tile_keys, start=1):
            datasets = {year: rasterio.open(files[year][key]) for year in ALL_YEARS}
            try:
                ref = datasets[ALL_YEARS[0]]
                tile_geom = box(*ref.bounds)
                if not basin_union.intersects(tile_geom):
                    continue

                nodata = int(ref.nodata if ref.nodata is not None else 255)

                for row_off in range(0, ref.height, BLOCK):
                    height = min(BLOCK, ref.height - row_off)
                    for col_off in range(0, ref.width, BLOCK):
                        width = min(BLOCK, ref.width - col_off)
                        window = Window(col_off, row_off, width, height)
                        transform = ref.window_transform(window)
                        bounds = rasterio.windows.bounds(window, ref.transform)
                        block_geom = box(*bounds)
                        if not basin_union.intersects(block_geom):
                            continue

                        region_shapes = [
                            (mapping(geom), region_id)
                            for region_id, _name, geom in region_rows
                            if geom.intersects(block_geom)
                        ]
                        if not region_shapes:
                            continue

                        region_ids = rasterize(
                            region_shapes,
                            out_shape=(height, width),
                            transform=transform,
                            fill=0,
                            dtype="uint8",
                            all_touched=False,
                        )
                        basin_mask = region_ids > 0
                        if not basin_mask.any():
                            continue

                        arrays = {year: datasets[year].read(1, window=window) for year in ALL_YEARS}
                        if block_geom.intersects(box(*aral_src.bounds)):
                            aral_mask = read_aral_mask(aral_src, (height, width), transform)
                        else:
                            aral_mask = np.zeros((height, width), dtype=bool)

                        event_category, event_year, event_from, event_to = detect_persistent_transitions(
                            arrays, aral_mask, nodata
                        )

                        area_rows = row_pixel_areas(transform, height)
                        weights = np.broadcast_to(area_rows[:, None], (height, width))

                        # Annual establishment areas are useful for diagnostics and optional trend analysis.
                        for year in range(ALL_YEARS[0], ALL_YEARS[-1] + 1):
                            year_mask = basin_mask & (event_year == year)
                            if not year_mask.any():
                                continue
                            for category in CATEGORY_ORDER:
                                category_id = CATEGORY_ID[category]
                                mask = year_mask & (event_category == category_id)
                                if mask.any():
                                    annual_totals[(year, category)] += float(weights[mask].sum())

                        for year0, year1, period in PERIODS:
                            classes = categories_for_period(event_category, event_year, year0, year1)
                            for category in CATEGORY_ORDER:
                                category_id = CATEGORY_ID[category]
                                mask = basin_mask & (classes == category_id)
                                if not mask.any():
                                    continue

                                totals[(period, category)] += float(weights[mask].sum())

                                pair_codes = event_from.astype("int32") * 1000 + event_to.astype("int32")
                                pair_values = pair_codes[mask]
                                pair_weights = weights[mask]
                                for encoded in np.unique(pair_values):
                                    from_code, to_code = divmod(int(encoded), 1000)
                                    details[(period, category, from_code, to_code)] += float(
                                        pair_weights[pair_values == encoded].sum()
                                    )

                                for region_id, name, _geom in region_rows:
                                    region_mask = mask & (region_ids == region_id)
                                    if region_mask.any():
                                        totals[(f"{period}|{name}", category)] += float(weights[region_mask].sum())
            finally:
                for dataset in datasets.values():
                    dataset.close()
            print(f"Persistent full 30 m statistics: tile {tile_index}/{len(tile_keys)} ({key})", flush=True)

    rows: list[dict[str, object]] = []
    for period, category in [(p[2], c) for p in PERIODS for c in CATEGORY_ORDER]:
        rows.append({
            "period": period,
            "region": "Aral Sea Basin",
            "category": category,
            "area_km2": totals[(period, category)],
        })
        for _region_id, name, _geom in region_rows:
            rows.append({
                "period": period,
                "region": name,
                "category": category,
                "area_km2": totals[(f"{period}|{name}", category)],
            })

    frame = pd.DataFrame(rows)
    frame.to_csv(cache, index=False, encoding="utf-8-sig")

    detail_rows = [
        {
            "period": period,
            "category": category,
            "from_code": from_code,
            "to_code": to_code,
            "area_km2": area,
        }
        for (period, category, from_code, to_code), area in sorted(details.items())
    ]
    pd.DataFrame(detail_rows).to_csv(
        OUTDIR / "Figure3_persistent_recentstate_transition_pathways_full30m.csv", index=False, encoding="utf-8-sig"
    )

    annual_rows = []
    for year in ALL_YEARS:
        for category in CATEGORY_ORDER:
            annual_rows.append({
                "year": year,
                "category": category,
                "new_persistent_transition_area_km2": annual_totals[(year, category)],
            })
    pd.DataFrame(annual_rows).to_csv(
        OUTDIR / "Figure3_persistent_recentstate_transition_area_by_year.csv", index=False, encoding="utf-8-sig"
    )

    cache_marker.write_text(cache_version + "\n", encoding="utf-8")
    return frame

def display_maps(files: dict[int, dict[str, Path]]) -> dict[str, tuple[np.ndarray, tuple[float, float, float, float]]]:
    cache = OUTDIR / "Figure3_persistent_recentstate_display_maps_step90.npz"
    cache_marker = OUTDIR / "Figure3_persistent_recentstate_display_maps_step90.complete"
    cache_version = (
        f"persistent-v2-recentstate-source{SOURCE_STABLE_YEARS}-target{TARGET_PERSIST_YEARS}"
        f"-recent{RECENT_STATE_START}_{RECENT_STATE_END}-frac{RECENT_STATE_FRACTION:.2f}"
    )
    if (
        cache.exists()
        and cache_marker.exists()
        and cache_marker.read_text(encoding="utf-8").strip() == cache_version
    ):
        saved = np.load(cache)
        return {
            label: (saved[f"canvas_{label}"], tuple(saved[f"extent_{label}"].tolist()))
            for _year0, _year1, label in PERIODS
        }

    shapes = read_shapes(SUBBASINS)
    first_key = next(iter(files[ALL_YEARS[0]]))
    extent, basin_mask = map_grid_geometry(files[ALL_YEARS[0]][first_key], shapes, DISPLAY_STEP)
    xmin, ymin, xmax, ymax = extent
    with rasterio.open(files[ALL_YEARS[0]][first_key]) as ref:
        dx = abs(ref.transform.a) * DISPLAY_STEP
        dy = abs(ref.transform.e) * DISPLAY_STEP
    width = int(np.ceil((xmax - xmin) / dx))
    height = int(np.ceil((ymax - ymin) / dy))
    canvases = {label: np.zeros((height, width), dtype="uint8") for _a, _b, label in PERIODS}

    with rasterio.open(ARAL_MASK_30M) as aral_src:
        for key in sorted(files[ALL_YEARS[0]]):
            datasets = {year: rasterio.open(files[year][key]) for year in ALL_YEARS}
            try:
                ref = datasets[ALL_YEARS[0]]
                ixmin, iymin = max(xmin, ref.bounds.left), max(ymin, ref.bounds.bottom)
                ixmax, iymax = min(xmax, ref.bounds.right), min(ymax, ref.bounds.top)
                if ixmin >= ixmax or iymin >= iymax:
                    continue

                window = rasterio.windows.from_bounds(ixmin, iymin, ixmax, iymax, ref.transform)
                out_h = max(1, int(np.ceil(window.height / DISPLAY_STEP)))
                out_w = max(1, int(np.ceil(window.width / DISPLAY_STEP)))
                transform = ref.window_transform(window) * rasterio.Affine.scale(
                    window.width / out_w, window.height / out_h
                )
                arrays = {
                    year: datasets[year].read(
                        1,
                        window=window,
                        out_shape=(out_h, out_w),
                        resampling=rasterio.enums.Resampling.nearest,
                    )
                    for year in ALL_YEARS
                }
                aral_mask = read_aral_mask(aral_src, (out_h, out_w), transform)
                nodata = int(ref.nodata if ref.nodata is not None else 255)
                event_category, event_year, _event_from, _event_to = detect_persistent_transitions(
                    arrays, aral_mask, nodata
                )

                x0 = transform.c + transform.a * 0.5
                y0 = transform.f + transform.e * 0.5
                c0 = int(np.floor((x0 - xmin) / dx))
                r0 = int(np.floor((ymax - y0) / dy))

                for year0, year1, label in PERIODS:
                    classified = categories_for_period(event_category, event_year, year0, year1)
                    dr0, dc0 = max(0, r0), max(0, c0)
                    sr0, sc0 = max(0, -r0), max(0, -c0)
                    rows = min(height - dr0, out_h - sr0)
                    cols = min(width - dc0, out_w - sc0)
                    if rows > 0 and cols > 0:
                        sub = classified[sr0 : sr0 + rows, sc0 : sc0 + cols]
                        target = canvases[label][dr0 : dr0 + rows, dc0 : dc0 + cols]
                        target[sub > 0] = sub[sub > 0]
            finally:
                for dataset in datasets.values():
                    dataset.close()

    for canvas in canvases.values():
        canvas[~basin_mask] = 0

    np.savez_compressed(
        cache,
        **{f"canvas_{label}": canvases[label] for _a, _b, label in PERIODS},
        **{f"extent_{label}": np.asarray(extent) for _a, _b, label in PERIODS},
    )
    cache_marker.write_text(cache_version + "\n", encoding="utf-8")
    return {label: (canvases[label], extent) for _a, _b, label in PERIODS}

def polish_map_axis(ax: plt.Axes, letter: str, period: str, show_x: bool, show_y: bool) -> None:
    ax.set_aspect("auto")
    ax.set_title(f"({letter})  {period}", loc="left", fontsize=9.2, fontweight="bold", pad=3.5)
    for text in ax.texts:
        text.set_fontfamily("sans-serif")
        text.set_fontsize(6.5)
        text.set_fontweight("normal")
    ax.tick_params(axis="both", labelsize=6.5, width=0.7, length=2.7, pad=1.8)
    if not show_x:
        ax.tick_params(axis="x", labelbottom=False)
    if not show_y:
        ax.tick_params(axis="y", labelleft=False)
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)
        spine.set_color("#333333")
    for line in ax.lines:
        if line.get_linestyle() == "--":
            line.set_linewidth(0.35)


def style_bar_axis(ax: plt.Axes) -> None:
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.75)
        spine.set_color("#333333")
    ax.grid(axis="y", color="#D8D8D8", linewidth=0.4)
    ax.set_axisbelow(True)
    ax.tick_params(width=0.7, length=2.7, labelsize=6.6)


def draw_map_panel(ax: plt.Axes, canvas: np.ndarray, extent, shapes, countries) -> None:
    colors = ["#ffffff00"] + [CATEGORY_COLORS[name] for name in CATEGORY_ORDER]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(-0.5, len(colors) + 0.5, 1), cmap.N)
    xmin, ymin, xmax, ymax = extent
    ax.imshow(canvas, extent=(xmin, xmax, ymin, ymax), origin="upper", cmap=cmap, norm=norm, interpolation="nearest", zorder=3)
    draw_boundaries_and_grid(ax, shapes, extent, country_shapes=countries)


def make_figure(stats: pd.DataFrame, maps) -> Path:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    shapes = read_shapes(SUBBASINS)
    countries = read_shapes(COUNTRY_BOUNDARY_PATH) if COUNTRY_BOUNDARY_PATH is not None and Path(COUNTRY_BOUNDARY_PATH).exists() else None
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7,
            "axes.labelsize": 7.5,
            "xtick.labelsize": 6.6,
            "ytick.labelsize": 6.6,
            "legend.fontsize": 7,
            "axes.linewidth": 0.7,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )
    fig = plt.figure(figsize=(183 / 25.4, 171 / 25.4), facecolor="white")
    left, right = 0.06, 0.95
    map_width, map_height = 0.27, 0.26
    map_lefts = [left, 0.37, 0.68]
    map_bottoms = [0.705, 0.405]
    for idx, (year0, year1, label) in enumerate(PERIODS):
        row, col = divmod(idx, 3)
        ax = fig.add_axes([map_lefts[col], map_bottoms[row], map_width, map_height])
        canvas, extent = maps[label]
        draw_map_panel(ax, canvas, extent, shapes, countries)
        ax.set_position([map_lefts[col], map_bottoms[row], map_width, map_height])
        polish_map_axis(ax, chr(ord("a") + idx), label, row == 1, col == 0)

    panel_gap = 0.065
    panel_width = (right - left - panel_gap) / 2
    ax_g = fig.add_axes([left, 0.115, panel_width, 0.215])
    ax_h = fig.add_axes([left + panel_width + panel_gap, 0.115, panel_width, 0.215])

    basin = stats[stats["region"] == "Aral Sea Basin"].pivot_table(
        index="period", columns="category", values="area_km2", aggfunc="sum", fill_value=0
    )
    basin = basin.reindex([p[2] for p in PERIODS], fill_value=0).reindex(columns=CATEGORY_ORDER, fill_value=0)
    x = np.arange(len(basin))
    bottom = np.zeros(len(basin))
    for category in CATEGORY_ORDER:
        values = basin[category].to_numpy() / 1000
        ax_g.bar(x, values, bottom=bottom, width=0.74, color=CATEGORY_COLORS[category], edgecolor="white", linewidth=0.35)
        bottom += values
    ax_g.set_xticks(x)
    ax_g.set_xticklabels(basin.index, rotation=32, ha="right")
    ax_g.set_ylabel(r"Area ($10^3$ km$^2$)")
    ax_g.yaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{value:,.0f}"))
    ax_g.set_ylim(0, bottom.max() * 1.07)
    ax_g.set_title(r"(g)  Long-term persistent transition areas ($10^3$ km$^2$)", loc="left", fontsize=9.2, fontweight="bold", pad=4.5)
    style_bar_axis(ax_g)

    regions = stats[(stats["period"] == "2001-2024") & (stats["region"] != "Aral Sea Basin")].pivot_table(
        index="region", columns="category", values="area_km2", aggfunc="sum", fill_value=0
    )
    regions = regions.reindex(columns=CATEGORY_ORDER, fill_value=0)
    regions = regions.loc[regions.sum(axis=1).sort_values(ascending=False).index]
    x = np.arange(len(regions))
    bottom = np.zeros(len(regions))
    for category in CATEGORY_ORDER:
        values = regions[category].to_numpy() / 1000
        ax_h.bar(x, values, bottom=bottom, width=0.74, color=CATEGORY_COLORS[category], edgecolor="white", linewidth=0.35)
        bottom += values
    ax_h.set_xticks(x)
    ax_h.set_xticklabels(regions.index, rotation=28, ha="right")
    ax_h.set_ylabel(r"Area ($10^3$ km$^2$)")
    ax_h.yaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{value:,.0f}"))
    ax_h.set_ylim(0, bottom.max() * 1.07)
    ax_h.set_title(r"(h)  Subbasin transition areas ($10^3$ km$^2$)", loc="left", fontsize=9.2, fontweight="bold", pad=4.5)
    style_bar_axis(ax_h)

    handles = [Patch(facecolor=CATEGORY_COLORS[name], edgecolor="none", label=name) for name in CATEGORY_ORDER]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.025), columnspacing=1.25, handlelength=1.3)
    stem = FIGDIR / "Figure3_AralUnion_reclassified_persistent_recentstate_full30m"
    fig.savefig(stem.with_suffix(".png"), dpi=600)
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".svg"))
    plt.close(fig)
    return stem


def write_reports(stats: pd.DataFrame, stem: Path) -> None:
    basin = stats[stats["region"] == "Aral Sea Basin"].copy()
    full = basin[basin["period"] == "2001-2024"].set_index("category")["area_km2"].reindex(CATEGORY_ORDER)
    table = basin.pivot_table(index="period", columns="category", values="area_km2", fill_value=0).reindex(
        [p[2] for p in PERIODS]
    )
    total = float(full.sum())
    degradation_excluding_exposure = float(
        full["Natural-cover degradation"]
        + full["Land-use intensification"]
        + full["Hydrological degradation"]
    )

    pathways = pd.read_csv(OUTDIR / "Figure3_persistent_recentstate_transition_pathways_full30m.csv")
    water_to_bare = pathways[
        (pathways["period"] == "2001-2024")
        & (pathways["from_code"] == 60)
        & (pathways["to_code"] == 90)
    ]
    water_to_bare_total = float(water_to_bare["area_km2"].sum())

    aral_mask_area = 0.0
    with rasterio.open(ARAL_MASK_30M) as mask_src:
        for _index, window in mask_src.block_windows(1):
            mask = mask_src.read(1, window=window) > 0
            transform = mask_src.window_transform(window)
            row_areas = row_pixel_areas(transform, mask.shape[0])
            aral_mask_area += float((mask * row_areas[:, None]).sum())

    rounded = table.round(2)
    markdown_rows = [
        "| Period | " + " | ".join(str(column) for column in rounded.columns) + " |",
        "| --- | " + " | ".join("---:" for _ in rounded.columns) + " |",
    ]
    for period, row in rounded.iterrows():
        markdown_rows.append(
            "| " + str(period) + " | " + " | ".join(f"{float(value):,.2f}" for value in row) + " |"
        )

    detectable_start = ALL_YEARS[0] + SOURCE_STABLE_YEARS
    detectable_end = ALL_YEARS[-1] - TARGET_PERSIST_YEARS + 1

    lines = [
        "# Figure 3 long-term persistent land-transition reclassification",
        "",
        "## Temporal persistence rule",
        "",
        f"- Source stability: the source class must persist for {SOURCE_STABLE_YEARS} consecutive years immediately before the transition.",
        f"- Initial target persistence: the target class must persist for {TARGET_PERSIST_YEARS} consecutive years beginning in the transition year.",
        f"- Recent-state confirmation: for transitions established by {RECENT_STATE_START}, at least 4 of 5 years during {RECENT_STATE_START}-{RECENT_STATE_END} must confirm persistence; later transitions use at least {RECENT_STATE_FRACTION:.0%} of available post-transition years with a minimum of {RECENT_STATE_MIN_OBS} observations.",
        "- For natural-cover degradation and land-use intensification, recent confirmation requires the same target class; for hydrological degradation and Aral Sea lakebed exposure, it requires continued absence of open water.",
        "- Only the first transition satisfying both filters is retained for each pixel.",
        f"- With this rule and a 2001-2024 record, persistent transition years are directly detectable from {detectable_start} through {detectable_end}.",
        "- The beginning and end of the record are therefore temporally censored: transitions beginning before the detectable window or during the final two years cannot be confirmed with the same rule.",
        "",
        "## Operational transition classes",
        "",
        "- Natural-cover degradation: Forest→Grassland, Shrubland→Grassland, and Forest/Grassland/Shrubland/Wetland→Bare land.",
        "- Land-use intensification: Grassland/Forest/Wetland→Cropland.",
        "- Hydrological degradation: Water→Bare land/Grassland/Cropland outside the Aral mask.",
        "- Aral Sea lakebed exposure: Water→Bare land inside the Aral mask.",
        "- Aral mask: archived 2001-2024 observed-water union projected to the 30 m FROMGLC30 grid; its upstream delineation is not reproduced in this package.",
        "- Transition areas use every valid 30 m pixel; display sampling is used only in map panels.",
        "- The 2021-2024 period is right-censored: with a three-year target-persistence requirement, new transitions can be confirmed only through 2022.",
        "",
        "## Basin-wide long-term persistent-transition areas by establishment period (km²)",
        "",
        "\n".join(markdown_rows),
        "",
        "## Full-period composition (first long-term persistent event per pixel)",
        "",
    ]

    if total > 0:
        for category, area in full.items():
            lines.append(f"- {category}: {area:,.2f} km² ({100 * area / total:.2f}% of the four-category total).")
    else:
        for category, area in full.items():
            lines.append(f"- {category}: {area:,.2f} km².")

    lines.extend(
        [
            "",
            "## Derived checks and interpretation",
            "",
            f"- The 2001-2024 Aral water-union mask covers {aral_mask_area:,.2f} km².",
            f"- Long-term persistent degradation-related transitions excluding Aral Sea lakebed exposure total {degradation_excluding_exposure:,.2f} km².",
            f"- Long-term persistent Water→Bare totals {water_to_bare_total:,.2f} km².",
            "- Period panels show the establishment year of the first transition that satisfies both the initial persistence and recent-state confirmation criteria.",
            "- In the 2021-2024 panel, the 3-year initial persistence rule can directly confirm establishment only through 2022.",
            "- The five period totals are mutually exclusive and, subject to record-edge censoring, sum to the full-period first-event total.",
        ]
    )

    (OUTDIR / "analysis-report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUTDIR / "stats-appendix.md").write_text(
        "# Statistical appendix\n\n"
        "This figure reports deterministic full-pixel area aggregation after a two-stage temporal persistence filter. "
        f"A source state must be stable for {SOURCE_STABLE_YEARS} consecutive years immediately before a transition, "
        f"and the target state must persist for {TARGET_PERSIST_YEARS} consecutive years beginning with the transition year. "
        f"A recent-state confirmation is then applied using {RECENT_STATE_START}-{RECENT_STATE_END}: older terrestrial transitions require the same target class in at least 4 of 5 recent years, while water-loss transitions require continued absence of open water. "
        "Transitions established after 2020 use at least 80% of available post-transition recent years, with a minimum of three observations. "
        "Only the first transition satisfying both stages per pixel is retained. "
        "No sampling-based confidence intervals or sensitivity analysis are used. Pixel areas are calculated row-wise "
        "on the geographic grid using a spherical Earth area formula.\n",
        encoding="utf-8",
    )
    (OUTDIR / "figure-catalog.md").write_text(
        "# Figure catalog\n\n"
        f"- Figure 3: `{stem.name}.png/.tiff/.pdf/.svg`\n"
        "- Quantitative source data: `Figure3_persistent_recentstate_transition_areas_full30m.csv`\n"
        "- Pathway source data: `Figure3_persistent_recentstate_transition_pathways_full30m.csv`\n"
        "- Annual establishment areas: `Figure3_persistent_recentstate_transition_area_by_year.csv`\n"
        f"- Aral mask: `{ARAL_MASK_30M}`\n"
        "- Map display cache: `Figure3_persistent_recentstate_display_maps_step90.npz`\n",
        encoding="utf-8",
    )

def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    files = find_fromglc_files()
    if not ARAL_MASK_30M.exists():
        raise FileNotFoundError(ARAL_MASK_30M)
    stats = full_resolution_statistics(files)
    maps = display_maps(files)
    stem = make_figure(stats, maps)
    write_reports(stats, stem)
    print(f"Figure: {stem}", flush=True)


if __name__ == "__main__":
    main()
