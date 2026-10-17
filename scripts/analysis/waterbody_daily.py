from __future__ import annotations

import argparse
import calendar
import math
import re
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.colors import LinearSegmentedColormap
from rasterio.features import geometry_mask
from rasterio.merge import merge
from rasterio.plot import plotting_extent
from rasterio.transform import Affine
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from rasterio.windows import Window, from_bounds
from shapely.geometry import box


WATER_VALUES = {240, 250}
INLAND_WATER_VALUE = 250
ICE_VALUE = 100
LAND_VALUE = 10
NODATA_VALUE = 0

SEASON_ORDER = ["DJF", "MAM", "JJA", "SON"]
SEASON_MONTHS = {
    "DJF": {12, 1, 2},
    "MAM": {3, 4, 5},
    "JJA": {6, 7, 8},
    "SON": {9, 10, 11},
}

COUNTRY_LABELS = [
    ("Kazakhstan", 66.0, 47.5),
    ("Uzbekistan", 63.8, 41.3),
    ("Turkmenistan", 59.6, 38.1),
    ("Tajikistan", 70.3, 38.4),
    ("Kyrgyzstan", 74.0, 41.4),
    ("Afghanistan", 66.4, 35.8),
]


@dataclass
class TileContext:
    tile: str
    sample_path: Path
    window: Window
    transform: Affine
    crs: object
    basin_mask: np.ndarray
    subbasin_masks: dict[str, np.ndarray]
    pixel_area_km2: float


def clean_subbasin_name(value: object) -> str:
    text = str(value)
    replacements = {
        "SYR DARYA": "Syr Darya",
        "AMU DARYA": "Amu Darya",
        "ARAL SEA (NORTH) (ex 2161)": "Aral Sea (North)",
        "ARAL SEA (SOUTH) (ex 2172)": "Aral Sea (South)",
    }
    return replacements.get(text, re.sub(r"\s*\(ex\s*\d+\)\s*", "", text, flags=re.IGNORECASE))


def parse_years(value: str) -> list[int]:
    years: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = [int(x) for x in part.split("-", 1)]
            years.update(range(start, end + 1))
        else:
            years.add(int(part))
    return sorted(years)


def doy_to_date(year: int, doy: int) -> datetime:
    return datetime.strptime(f"{year}{doy:03d}", "%Y%j")


def season_for_month(month: int) -> str:
    for season, months in SEASON_MONTHS.items():
        if month in months:
            return season
    raise ValueError(month)


def find_water_files(data_dir: Path, years: list[int]) -> pd.DataFrame:
    pattern = re.compile(r"(?P<year>\d{4})h(?P<h>\d{2})v(?P<v>\d{2})(?P<doy>\d{3})_water\.tif$", re.IGNORECASE)
    rows = []
    for path in data_dir.glob("*/*/*_water.tif"):
        match = pattern.match(path.name)
        if not match:
            continue
        year = int(match.group("year"))
        if year not in years:
            continue
        doy = int(match.group("doy"))
        date = doy_to_date(year, doy)
        tile = f"h{int(match.group('h')):02d}v{int(match.group('v')):02d}"
        rows.append(
            {
                "year": year,
                "doy": doy,
                "date": date.date().isoformat(),
                "month": date.month,
                "season": season_for_month(date.month),
                "tile": tile,
                "path": path,
            }
        )
    if not rows:
        raise SystemExit(f"No *_water.tif files found under {data_dir}")
    df = pd.DataFrame(rows).sort_values(["year", "doy", "tile"]).reset_index(drop=True)
    return df


def read_basin(shp: Path) -> gpd.GeoDataFrame:
    basin = gpd.read_file(shp)
    if basin.crs is None:
        basin = basin.set_crs("EPSG:4326")
    basin = basin.to_crs("EPSG:4326")
    basin["subbasin"] = basin["SUBREGNAME"].map(clean_subbasin_name)
    return basin[["subbasin", "geometry"]]


def country_path_default() -> Path:
    return Path(__file__).resolve().parents[2] / "data/boundaries/countries.shp"


def load_countries(path: Path | None, extent: tuple[float, float, float, float]) -> gpd.GeoDataFrame | None:
    if path is None or not path.exists():
        return None
    countries = gpd.read_file(path)
    if countries.crs is None:
        countries = countries.set_crs("EPSG:4326")
    countries = countries.to_crs("EPSG:4326")
    clip = gpd.GeoDataFrame(geometry=[box(*extent)], crs="EPSG:4326")
    try:
        return gpd.clip(countries, clip)
    except Exception:
        return countries.cx[extent[0] : extent[2], extent[1] : extent[3]]


def build_tile_contexts(files: pd.DataFrame, basin: gpd.GeoDataFrame) -> dict[str, TileContext]:
    contexts: dict[str, TileContext] = {}
    for tile, tile_rows in files.groupby("tile"):
        sample_path = Path(tile_rows.iloc[0]["path"])
        with rasterio.open(sample_path) as src:
            basin_src = basin.to_crs(src.crs)
            union = basin_src.geometry.union_all() if hasattr(basin_src.geometry, "union_all") else basin_src.unary_union
            raster_box = box(*src.bounds)
            if not union.intersects(raster_box):
                continue
            bounds = union.intersection(raster_box).bounds
            window = from_bounds(*bounds, transform=src.transform)
            window = window.round_offsets().round_lengths()
            window = Window(
                max(0, int(window.col_off)),
                max(0, int(window.row_off)),
                min(src.width - max(0, int(window.col_off)), int(window.width) + 2),
                min(src.height - max(0, int(window.row_off)), int(window.height) + 2),
            )
            if window.width <= 0 or window.height <= 0:
                continue
            transform = src.window_transform(window)
            out_shape = (int(window.height), int(window.width))
            basin_mask = geometry_mask(
                [union],
                out_shape=out_shape,
                transform=transform,
                invert=True,
                all_touched=False,
            )
            subbasin_masks = {}
            for _, row in basin_src.iterrows():
                geom = row.geometry
                if geom is None or geom.is_empty or not geom.intersects(raster_box):
                    continue
                mask = geometry_mask(
                    [geom],
                    out_shape=out_shape,
                    transform=transform,
                    invert=True,
                    all_touched=False,
                )
                if mask.any():
                    subbasin_masks[str(row["subbasin"])] = mask
            pixel_area_km2 = abs(src.transform.a * src.transform.e) / 1_000_000
            contexts[tile] = TileContext(
                tile=tile,
                sample_path=sample_path,
                window=window,
                transform=transform,
                crs=src.crs,
                basin_mask=basin_mask,
                subbasin_masks=subbasin_masks,
                pixel_area_km2=pixel_area_km2,
            )
            print(f"context {tile}: window={int(window.width)}x{int(window.height)}, subbasins={len(subbasin_masks)}")
    return contexts


def summarize_array(arr: np.ndarray, mask: np.ndarray, pixel_area_km2: float) -> dict[str, float]:
    inside = mask
    valid = inside & (arr != NODATA_VALUE)
    water = inside & np.isin(arr, list(WATER_VALUES))
    inland = inside & (arr == INLAND_WATER_VALUE)
    ice = inside & (arr == ICE_VALUE)
    land = inside & (arr == LAND_VALUE)
    valid_pixels = int(valid.sum())
    water_pixels = int(water.sum())
    return {
        "valid_area_km2": valid_pixels * pixel_area_km2,
        "water_area_km2": water_pixels * pixel_area_km2,
        "inland_water_area_km2": int(inland.sum()) * pixel_area_km2,
        "ice_area_km2": int(ice.sum()) * pixel_area_km2,
        "land_area_km2": int(land.sum()) * pixel_area_km2,
        "water_fraction_of_valid": water_pixels / valid_pixels if valid_pixels else math.nan,
    }


def process_daily_stats(
    files: pd.DataFrame,
    contexts: dict[str, TileContext],
    out_dir: Path,
    chunk_size: int,
    workers: int,
    resume: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    intermediate_dir = out_dir / "intermediate"
    intermediate_dir.mkdir(parents=True, exist_ok=True)
    tile_cache = intermediate_dir / "tile_daily_stats.csv"
    sub_cache = intermediate_dir / "subbasin_tile_daily_stats.csv"
    done_keys: set[str] = set()
    if resume and tile_cache.exists():
        usecols = ["date", "tile"]
        done_df = pd.read_csv(tile_cache, usecols=usecols)
        done_keys = set(done_df["date"].astype(str) + "|" + done_df["tile"].astype(str))
        print(f"resume: {len(done_keys):,} tile-day records already cached")

    work = files.copy()
    work["key"] = work["date"].astype(str) + "|" + work["tile"].astype(str)
    if done_keys:
        work = work[~work["key"].isin(done_keys)].drop(columns=["key"]).reset_index(drop=True)
    else:
        work = work.drop(columns=["key"])

    def process_one(row_dict: dict[str, object]) -> tuple[dict[str, object], list[dict[str, object]]]:
        ctx = contexts.get(str(row_dict["tile"]))
        if ctx is None:
            raise RuntimeError(f"tile does not intersect basin: {row_dict['tile']}")
        with rasterio.open(row_dict["path"]) as src:
            arr = src.read(1, window=ctx.window)
        stats = summarize_array(arr, ctx.basin_mask, ctx.pixel_area_km2)
        tile_row = {
            "date": row_dict["date"],
            "year": int(row_dict["year"]),
            "doy": int(row_dict["doy"]),
            "month": int(row_dict["month"]),
            "season": row_dict["season"],
            "tile": row_dict["tile"],
            **stats,
        }
        sub_rows_local = []
        for subbasin, mask in ctx.subbasin_masks.items():
            sub_stats = summarize_array(arr, mask, ctx.pixel_area_km2)
            sub_rows_local.append(
                {
                    "date": row_dict["date"],
                    "year": int(row_dict["year"]),
                    "doy": int(row_dict["doy"]),
                    "month": int(row_dict["month"]),
                    "season": row_dict["season"],
                    "tile": row_dict["tile"],
                    "subbasin": subbasin,
                    **sub_stats,
                }
            )
        return tile_row, sub_rows_local

    t0 = time.time()
    total = len(work)
    print(f"daily stats work queue: {total:,} tile-day files")
    tile_buffer: list[dict[str, object]] = []
    sub_buffer: list[dict[str, object]] = []

    def flush() -> None:
        nonlocal tile_buffer, sub_buffer
        if tile_buffer:
            pd.DataFrame(tile_buffer).to_csv(
                tile_cache,
                mode="a",
                header=not tile_cache.exists(),
                index=False,
                encoding="utf-8-sig",
            )
            tile_buffer = []
        if sub_buffer:
            pd.DataFrame(sub_buffer).to_csv(
                sub_cache,
                mode="a",
                header=not sub_cache.exists(),
                index=False,
                encoding="utf-8-sig",
            )
            sub_buffer = []

    records = work.to_dict("records")
    submitted = 0
    completed = 0
    max_pending = max(workers * 3, workers)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = set()
        iterator = iter(records)
        while len(pending) < max_pending:
            try:
                pending.add(pool.submit(process_one, next(iterator)))
                submitted += 1
            except StopIteration:
                break
        while pending:
            for future in as_completed(pending):
                pending.remove(future)
                tile_row, sub_rows_local = future.result()
                tile_buffer.append(tile_row)
                sub_buffer.extend(sub_rows_local)
                completed += 1
                if completed % chunk_size == 0 or completed == total:
                    flush()
                    elapsed = (time.time() - t0) / 60
                    print(f"processed {completed:,}/{total:,} queued files ({elapsed:.1f} min)")
                try:
                    pending.add(pool.submit(process_one, next(iterator)))
                    submitted += 1
                except StopIteration:
                    pass
                break
    flush()

    tile_df = pd.read_csv(tile_cache)
    sub_df = pd.read_csv(sub_cache)
    daily = (
        tile_df.groupby(["date", "year", "doy", "month", "season"], as_index=False)[
            ["valid_area_km2", "water_area_km2", "inland_water_area_km2", "ice_area_km2", "land_area_km2"]
        ]
        .sum()
        .sort_values("date")
    )
    daily["water_fraction_of_valid"] = daily["water_area_km2"] / daily["valid_area_km2"]
    sub_daily = (
        sub_df.groupby(["date", "year", "doy", "month", "season", "subbasin"], as_index=False)[
            ["valid_area_km2", "water_area_km2", "inland_water_area_km2", "ice_area_km2", "land_area_km2"]
        ]
        .sum()
        .sort_values(["date", "subbasin"])
    )
    sub_daily["water_fraction_of_valid"] = sub_daily["water_area_km2"] / sub_daily["valid_area_km2"]
    daily.to_csv(out_dir / "daily_water_area_basin.csv", index=False, encoding="utf-8-sig")
    sub_daily.to_csv(out_dir / "daily_water_area_subbasins.csv", index=False, encoding="utf-8-sig")
    return daily, sub_daily


def add_summary_tables(daily: pd.DataFrame, sub_daily: pd.DataFrame, out_dir: Path) -> None:
    annual = (
        daily.groupby("year")["water_area_km2"]
        .agg(
            mean_water_area_km2="mean",
            median_water_area_km2="median",
            min_water_area_km2="min",
            max_water_area_km2="max",
            p10_water_area_km2=lambda x: x.quantile(0.10),
            p90_water_area_km2=lambda x: x.quantile(0.90),
            std_water_area_km2="std",
            observed_days="count",
        )
        .reset_index()
    )
    annual["mean_valid_area_km2"] = daily.groupby("year")["valid_area_km2"].mean().values
    annual["mean_water_fraction_of_valid"] = annual["mean_water_area_km2"] / annual["mean_valid_area_km2"]
    annual.to_csv(out_dir / "annual_water_summary.csv", index=False, encoding="utf-8-sig")

    seasonal = (
        daily.groupby(["year", "season"])["water_area_km2"]
        .agg(
            mean_water_area_km2="mean",
            median_water_area_km2="median",
            min_water_area_km2="min",
            max_water_area_km2="max",
            observed_days="count",
        )
        .reset_index()
    )
    seasonal["season"] = pd.Categorical(seasonal["season"], SEASON_ORDER, ordered=True)
    seasonal = seasonal.sort_values(["year", "season"])
    seasonal.to_csv(out_dir / "seasonal_water_summary.csv", index=False, encoding="utf-8-sig")

    sub_annual = (
        sub_daily.groupby(["year", "subbasin"])["water_area_km2"]
        .agg(
            mean_water_area_km2="mean",
            median_water_area_km2="median",
            min_water_area_km2="min",
            max_water_area_km2="max",
            observed_days="count",
        )
        .reset_index()
    )
    sub_annual.to_csv(out_dir / "annual_water_summary_subbasins.csv", index=False, encoding="utf-8-sig")

    trend_rows = []
    x = annual["year"].to_numpy(dtype=float)
    for column in ["mean_water_area_km2", "p90_water_area_km2", "p10_water_area_km2"]:
        y = annual[column].to_numpy(dtype=float)
        slope, intercept = np.polyfit(x, y, 1)
        pred = slope * x + intercept
        ss_res = float(np.sum((y - pred) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        trend_rows.append(
            {
                "series": column,
                "slope_km2_per_year": slope,
                "change_2001_2024_km2": slope * (2024 - 2001),
                "r2": 1 - ss_res / ss_tot if ss_tot else math.nan,
            }
        )
    pd.DataFrame(trend_rows).to_csv(out_dir / "linear_trends.csv", index=False, encoding="utf-8-sig")


def write_annual_occurrence_tiles(
    files: pd.DataFrame,
    contexts: dict[str, TileContext],
    out_dir: Path,
    chunk_size: int,
) -> None:
    raster_dir = out_dir / "rasters" / "annual_occurrence_tiles"
    raster_dir.mkdir(parents=True, exist_ok=True)
    grouped = files.groupby(["year", "tile"])
    total_groups = len(grouped)
    for group_idx, ((year, tile), group) in enumerate(grouped, start=1):
        ctx = contexts.get(tile)
        if ctx is None:
            continue
        out_path = raster_dir / f"water_occurrence_{year}_{tile}.tif"
        if out_path.exists():
            continue
        water_count = np.zeros(ctx.basin_mask.shape, dtype=np.uint16)
        valid_count = np.zeros(ctx.basin_mask.shape, dtype=np.uint16)
        for idx, row in enumerate(group.itertuples(index=False), start=1):
            with rasterio.open(row.path) as src:
                arr = src.read(1, window=ctx.window)
            inside = ctx.basin_mask
            valid = inside & (arr != NODATA_VALUE)
            water = inside & np.isin(arr, list(WATER_VALUES))
            valid_count += valid.astype(np.uint16)
            water_count += water.astype(np.uint16)
            if idx % chunk_size == 0:
                print(f"occurrence {year} {tile}: {idx}/{len(group)}")
        occurrence = np.full(ctx.basin_mask.shape, np.nan, dtype=np.float32)
        np.divide(water_count, valid_count, out=occurrence, where=valid_count > 0)
        profile = {
            "driver": "GTiff",
            "height": occurrence.shape[0],
            "width": occurrence.shape[1],
            "count": 1,
            "dtype": "float32",
            "crs": ctx.crs,
            "transform": ctx.transform,
            "nodata": np.nan,
            "compress": "deflate",
            "predictor": 2,
            "tiled": True,
            "blockxsize": 256,
            "blockysize": 256,
        }
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(occurrence, 1)
        print(f"annual occurrence done {group_idx}/{total_groups}: {out_path}")


def save_line_figures(out_dir: Path) -> None:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    annual = pd.read_csv(out_dir / "annual_water_summary.csv")
    seasonal = pd.read_csv(out_dir / "seasonal_water_summary.csv")
    daily = pd.read_csv(out_dir / "daily_water_area_basin.csv")
    sub_annual = pd.read_csv(out_dir / "annual_water_summary_subbasins.csv")

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=300)
    ax.plot(annual["year"], annual["mean_water_area_km2"], color="#1f78b4", lw=2.2, label="Annual mean")
    ax.fill_between(
        annual["year"],
        annual["p10_water_area_km2"],
        annual["p90_water_area_km2"],
        color="#9ecae1",
        alpha=0.35,
        label="Daily P10-P90",
    )
    ax.set_xlabel("Year")
    ax.set_ylabel("Water area (km$^2$)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(fig_dir / "annual_water_area_trend.png")
    plt.close(fig)

    pivot = seasonal.pivot(index="year", columns="season", values="mean_water_area_km2").reindex(columns=SEASON_ORDER)
    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=300)
    for season in SEASON_ORDER:
        ax.plot(pivot.index, pivot[season], marker="o", ms=3, lw=1.6, label=season)
    ax.set_xlabel("Year")
    ax.set_ylabel("Seasonal mean water area (km$^2$)")
    ax.legend(ncol=4, frameon=False)
    fig.tight_layout()
    fig.savefig(fig_dir / "seasonal_water_area_timeseries.png")
    plt.close(fig)

    clim = daily.groupby(["year", "doy"])["water_area_km2"].mean().reset_index()
    heat = clim.pivot(index="year", columns="doy", values="water_area_km2")
    fig, ax = plt.subplots(figsize=(10.5, 5.2), dpi=300)
    im = ax.imshow(heat.values, aspect="auto", cmap="YlGnBu", interpolation="nearest")
    ax.set_xlabel("Day of year")
    ax.set_ylabel("Year")
    ax.set_yticks(np.arange(len(heat.index))[::2])
    ax.set_yticklabels(heat.index[::2])
    cbar = fig.colorbar(im, ax=ax, pad=0.012)
    cbar.set_label("Water area (km$^2$)")
    fig.tight_layout()
    fig.savefig(fig_dir / "daily_water_area_year_doy_heatmap.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5.2), dpi=300)
    for subbasin, group in sub_annual.groupby("subbasin"):
        ax.plot(group["year"], group["mean_water_area_km2"], lw=1.6, label=subbasin)
    ax.set_xlabel("Year")
    ax.set_ylabel("Annual mean water area (km$^2$)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "subbasin_annual_water_area.png")
    plt.close(fig)


def plot_occurrence_maps(
    out_dir: Path,
    basin: gpd.GeoDataFrame,
    countries: gpd.GeoDataFrame | None,
    years: list[int],
) -> None:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tile_dir = out_dir / "rasters" / "annual_occurrence_tiles"
    cmap = LinearSegmentedColormap.from_list("water_occurrence", ["#f7fbff", "#c6dbef", "#6baed6", "#08519c"])
    basin4326 = basin.to_crs("EPSG:4326")
    minx, miny, maxx, maxy = basin4326.total_bounds
    pad_x = (maxx - minx) * 0.04
    pad_y = (maxy - miny) * 0.04
    extent4326 = (minx - pad_x, miny - pad_y, maxx + pad_x, maxy + pad_y)

    for year in years:
        paths = sorted(tile_dir.glob(f"water_occurrence_{year}_*.tif"))
        if not paths:
            continue
        datasets = [rasterio.open(path) for path in paths]
        vrts = [
            WarpedVRT(ds, crs="EPSG:4326", resolution=0.01, resampling=Resampling.nearest, nodata=np.nan)
            for ds in datasets
        ]
        mosaic, transform = merge(vrts, nodata=np.nan)
        for vrt in vrts:
            vrt.close()
        for ds in datasets:
            ds.close()
        arr = mosaic[0]
        fig, ax = plt.subplots(figsize=(8.4, 6.2), dpi=300)
        ax.imshow(arr, extent=plotting_extent(arr, transform), origin="upper", cmap=cmap, vmin=0, vmax=1)
        basin4326.boundary.plot(ax=ax, color="black", linewidth=0.8, zorder=3)
        if countries is not None:
            countries.boundary.plot(ax=ax, color="0.35", linewidth=0.45, zorder=2)
        for name, lon, lat in COUNTRY_LABELS:
            if extent4326[0] <= lon <= extent4326[2] and extent4326[1] <= lat <= extent4326[3]:
                ax.text(lon, lat, name, ha="center", va="center", fontsize=7.5, color="0.25", zorder=4)
        ax.set_xlim(extent4326[0], extent4326[2])
        ax.set_ylim(extent4326[1], extent4326[3])
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        cbar = fig.colorbar(ax.images[0], ax=ax, pad=0.012)
        cbar.set_label("Water occurrence frequency")
        fig.tight_layout()
        fig.savefig(fig_dir / f"water_occurrence_{year}.png")
        plt.close(fig)

    change_years = [years[0], years[-1]] if len(years) >= 2 else []
    if len(change_years) == 2:
        arrays = []
        transforms = []
        for year in change_years:
            paths = sorted(tile_dir.glob(f"water_occurrence_{year}_*.tif"))
            datasets = [rasterio.open(path) for path in paths]
            vrts = [
                WarpedVRT(ds, crs="EPSG:4326", resolution=0.01, resampling=Resampling.nearest, nodata=np.nan)
                for ds in datasets
            ]
            mosaic, transform = merge(vrts, nodata=np.nan)
            for vrt in vrts:
                vrt.close()
            for ds in datasets:
                ds.close()
            arrays.append(mosaic[0])
            transforms.append(transform)
        diff = arrays[1] - arrays[0]
        fig, ax = plt.subplots(figsize=(8.4, 6.2), dpi=300)
        im = ax.imshow(
            diff,
            extent=plotting_extent(diff, transforms[0]),
            origin="upper",
            cmap="RdBu",
            vmin=-1,
            vmax=1,
        )
        basin4326.boundary.plot(ax=ax, color="black", linewidth=0.8, zorder=3)
        if countries is not None:
            countries.boundary.plot(ax=ax, color="0.35", linewidth=0.45, zorder=2)
        for name, lon, lat in COUNTRY_LABELS:
            if extent4326[0] <= lon <= extent4326[2] and extent4326[1] <= lat <= extent4326[3]:
                ax.text(lon, lat, name, ha="center", va="center", fontsize=7.5, color="0.25", zorder=4)
        ax.set_xlim(extent4326[0], extent4326[2])
        ax.set_ylim(extent4326[1], extent4326[3])
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        cbar = fig.colorbar(im, ax=ax, pad=0.012)
        cbar.set_label(f"Occurrence change ({change_years[1]} - {change_years[0]})")
        fig.tight_layout()
        fig.savefig(fig_dir / f"water_occurrence_change_{change_years[0]}_{change_years[1]}.png")
        plt.close(fig)


def write_report(out_dir: Path) -> None:
    annual = pd.read_csv(out_dir / "annual_water_summary.csv")
    seasonal = pd.read_csv(out_dir / "seasonal_water_summary.csv")
    trends = pd.read_csv(out_dir / "linear_trends.csv")
    first = annual.iloc[0]
    last = annual.iloc[-1]
    mean_change = last["mean_water_area_km2"] - first["mean_water_area_km2"]
    pct = mean_change / first["mean_water_area_km2"] * 100 if first["mean_water_area_km2"] else math.nan
    slope = trends.loc[trends["series"] == "mean_water_area_km2", "slope_km2_per_year"].iloc[0]
    seasonal_mean = seasonal.groupby("season")["mean_water_area_km2"].mean().reindex(SEASON_ORDER)
    report = f"""# Daily MODIS Waterbody Analysis, Aral Sea Basin, 2001-2024

## Analysis Scope

This analysis uses the daily 500 m iEarth Global Water Body Mapping Dataset to quantify open-water dynamics within `AralSeaBasinS.shp`. Water pixels are defined as values 240 and 250. Valid observation area excludes value 0. Statistics are reported for the whole basin and for the four sub-basins in the basin shapefile.

## Main Descriptive Findings

The annual mean mapped water area changed from {first['mean_water_area_km2']:.1f} km2 in {int(first['year'])} to {last['mean_water_area_km2']:.1f} km2 in {int(last['year'])}, a net change of {mean_change:.1f} km2 ({pct:.1f}%). A linear fit to annual mean daily water area gives a slope of {slope:.2f} km2 yr-1 over 2001-2024.

Across all years, the seasonal mean water area ranks as follows: {', '.join(f'{s}: {v:.1f} km2' for s, v in seasonal_mean.items())}. The daily heatmap and seasonal time series should be used together: the heatmap shows within-year timing and intermittent observation gaps, whereas seasonal averages summarize the recurrent hydrological signal.

## Figure Guide

- `annual_water_area_trend.png`: annual mean daily water area with daily P10-P90 envelope.
- `seasonal_water_area_timeseries.png`: interannual evolution of DJF, MAM, JJA, and SON water area.
- `daily_water_area_year_doy_heatmap.png`: day-of-year by year view of water-area seasonality.
- `subbasin_annual_water_area.png`: annual mean water area by sub-basin.
- `water_occurrence_*.png`: spatial frequency of daily water detection for selected years.
- `water_occurrence_change_2001_2024.png`: spatial shift in annual water occurrence frequency between the first and last analysis years.

## Caveats

This version treats water values 240 and 250 as water and excludes only value 0 from valid observations. It does not yet impose a separate QC threshold. A QC-filtered sensitivity analysis can be added if the study requires conservative cloud-confidence screening.
"""
    (out_dir / "analysis-report.md").write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze daily iEarth 500 m waterbody dynamics in the Aral Sea basin.")
    root = Path(__file__).resolve().parents[2]
    parser.add_argument("--data-dir", default=str(root / "data/raw/daily_waterbody"))
    parser.add_argument("--basin-shp", default=str(root / "data/boundaries/AralSeaBasinS.shp"))
    parser.add_argument("--countries-shp", default=str(country_path_default()))
    parser.add_argument("--years", default="2001-2024")
    parser.add_argument("--output-dir", default=str(root / "outputs/full/waterbody"))
    parser.add_argument("--selected-map-years", default="2001,2005,2010,2015,2020,2024")
    parser.add_argument("--progress-every", type=int, default=2500)
    parser.add_argument("--workers", type=int, default=4, help="Parallel file readers for daily statistics.")
    parser.add_argument("--no-resume", action="store_true", help="Ignore cached tile-day statistics and rescan all files.")
    parser.add_argument("--skip-daily-stats", action="store_true")
    parser.add_argument("--skip-occurrence", action="store_true")
    parser.add_argument("--skip-figures", action="store_true")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    years = parse_years(args.years)
    selected_years = parse_years(args.selected_map_years)
    basin = read_basin(Path(args.basin_shp))
    extent = tuple(basin.total_bounds)
    countries = load_countries(Path(args.countries_shp), extent)
    files = find_water_files(data_dir, years)
    files.to_csv(out_dir / "water_files_inventory.csv", index=False, encoding="utf-8-sig")
    print(f"water files: {len(files):,}; years: {min(years)}-{max(years)}")
    contexts = build_tile_contexts(files, basin)
    print(f"active tiles intersecting basin: {len(contexts)}")
    files = files[files["tile"].isin(contexts)].reset_index(drop=True)
    print(f"water files inside intersecting tiles: {len(files):,}")

    daily_path = out_dir / "daily_water_area_basin.csv"
    sub_daily_path = out_dir / "daily_water_area_subbasins.csv"
    if args.skip_daily_stats and daily_path.exists() and sub_daily_path.exists():
        daily = pd.read_csv(daily_path)
        sub_daily = pd.read_csv(sub_daily_path)
    else:
        daily, sub_daily = process_daily_stats(
            files,
            contexts,
            out_dir,
            args.progress_every,
            args.workers,
            resume=not args.no_resume,
        )
        add_summary_tables(daily, sub_daily, out_dir)

    if not args.skip_occurrence:
        write_annual_occurrence_tiles(files, contexts, out_dir, args.progress_every)

    if not args.skip_figures:
        save_line_figures(out_dir)
        plot_occurrence_maps(out_dir, basin, countries, selected_years)
        write_report(out_dir)
    print(f"outputs: {out_dir}")


if __name__ == "__main__":
    main()
