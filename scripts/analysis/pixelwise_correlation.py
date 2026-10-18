from __future__ import annotations

"""
Figure 7: pixel-wise coupling between annual Aral Sea shrinkage and annual
MESMA fractions, restricted to pixels already identified as persistent
degradation events by the Figure 5 workflow.

Key design choice
-----------------
This script does NOT redefine persistence. It reuses Figure 5's
Figure5_persistent_event_year.tif directly, so Figure 5 and Figure 7 share the
same temporal-persistence criterion:

1. two stable pre-onset years;
2. three persistent target years;
3. recent-state confirmation during 2020 to 2024;
4. only the first qualifying event retained;
5. the 2001 to 2024 Aral water-union mask was already excluded in Figure 5.

For every retained persistent-degradation pixel, Pearson r is then calculated
from the full annual 2001 to 2024 series between basin-wide annual lake
shrinkage and the local annual BS, PV, or NPV fraction.

Because subtracting a pixel-specific constant baseline does not change Pearson
r, the panels are labelled BS, PV and NPV rather than delta BS, delta PV and
delta NPV.
"""

import math
from contextlib import ExitStack
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window


ROOT = Path(__file__).resolve().parents[2]

MESMA_DIR = ROOT / "data/raw/annual_mesma"
WATER_TABLE = ROOT / "data/figure_source/water_annual_shrinkage.csv"


FIG5_OUTDIR = ROOT / "outputs/full/figure5"
FIG5_EVENT_YEAR = FIG5_OUTDIR / "Figure5_persistent_event_year.tif"
if not FIG5_EVENT_YEAR.exists():
    FIG5_EVENT_YEAR = ROOT / "data/figure_source/figure5_event_year.tif"

OUTDIR = ROOT / "outputs/full/figure7"
RASTER_DIR = OUTDIR / "rasters"
TABLE_DIR = OUTDIR / "tables"
FIGDIR = ROOT / "outputs/full/figure7"

BASIN_SHP = ROOT / "data/boundaries/AralSeaBasinS.shp"
COUNTRY_SHP = ROOT / "data/boundaries/countries.shp"

ALL_YEARS = list(range(2001, 2025))
BANDS = {"NPV": 2, "BS": 4, "PV": 5}
COMPONENTS = [
    ("BS", "BS"),
    ("PV", "PV"),
    ("NPV", "NPV"),
]
VIEW_EXTENT = [55.0, 80.0, 33.0, 50.0]
BLOCK_HEIGHT = 128
FORCE_RECOMPUTE = False


def ensure_dirs() -> None:
    for path in [OUTDIR, RASTER_DIR, TABLE_DIR, FIGDIR]:
        path.mkdir(parents=True, exist_ok=True)


def annual_raster(year: int) -> Path:
    return MESMA_DIR / f"Aral_MESMA_Degradation_{year}.tif"


def stat_raster(component: str, stat: str) -> Path:
    return RASTER_DIR / f"{component}_{stat}_vs_shrinkage_persistent_Fig5_2001_2024.tif"


def same_grid(a: rasterio.io.DatasetReader, b: rasterio.io.DatasetReader) -> bool:
    return (
        a.width == b.width
        and a.height == b.height
        and a.crs == b.crs
        and a.transform.almost_equals(b.transform)
    )


def validate_inputs() -> None:
    missing = []
    if not FIG5_EVENT_YEAR.exists():
        missing.append(str(FIG5_EVENT_YEAR))
    if not WATER_TABLE.exists():
        missing.append(str(WATER_TABLE))
    for year in ALL_YEARS:
        path = annual_raster(year)
        if not path.exists():
            missing.append(str(path))
    if missing:
        raise FileNotFoundError("Missing required input files:\n" + "\n".join(missing))

    with rasterio.open(annual_raster(2001)) as ref, rasterio.open(FIG5_EVENT_YEAR) as evt:
        if not same_grid(ref, evt):
            raise RuntimeError(
                "Figure 5 event-year raster and MESMA rasters do not share the same grid.\n"
                f"MESMA: {annual_raster(2001)}\n"
                f"Figure 5 event raster: {FIG5_EVENT_YEAR}"
            )

        for year in ALL_YEARS[1:]:
            with rasterio.open(annual_raster(year)) as src:
                if not same_grid(ref, src):
                    raise RuntimeError(f"MESMA grid mismatch in {year}: {annual_raster(year)}")


def load_shrinkage() -> np.ndarray:
    table = pd.read_csv(WATER_TABLE, encoding="utf-8-sig")
    required = {"year", "shrinkage_pct_from_baseline"}
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(
            f"Water table lacks columns: {sorted(missing)}\n"
            f"Available columns: {list(table.columns)}"
        )

    series = (
        table.set_index("year")
        .reindex(ALL_YEARS)["shrinkage_pct_from_baseline"]
    )
    if series.isna().any():
        bad = series[series.isna()].index.tolist()
        raise ValueError(f"Missing shrinkage values for years: {bad}")
    return series.to_numpy(dtype="float32")


def infer_scale_factor(src: rasterio.io.DatasetReader) -> float:
    """Convert stored MESMA values to percentage points."""
    sample_h = min(src.height, 160)
    sample_w = min(src.width, 160)
    vals = []
    for band in [BANDS["BS"], BANDS["PV"], BANDS["NPV"]]:
        arr = src.read(
            band,
            out_shape=(sample_h, sample_w),
            resampling=rasterio.enums.Resampling.nearest,
        ).astype("float32")
        if src.nodata is not None:
            arr[arr == src.nodata] = np.nan
        arr = arr[np.isfinite(arr)]
        if arr.size:
            vals.append(arr)
    if not vals:
        raise RuntimeError("Could not infer MESMA value scale.")
    q99 = float(np.nanpercentile(np.abs(np.concatenate(vals)), 99))
    if q99 <= 1.5:
        return 100.0
    if q99 <= 110.0:
        return 1.0
    if q99 <= 11000.0:
        return 0.01
    raise RuntimeError(f"Unexpected MESMA scale, 99th percentile = {q99:.3f}")


def betacf_array(a: float, b: float, x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x
    tiny = 1e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = np.ones_like(x, dtype="float64")
    d = 1.0 / np.maximum(tiny, 1.0 - qab * x / qap)
    h = d.copy()
    for m in range(1, 201):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 / np.maximum(tiny, 1.0 + aa * d)
        c = np.maximum(tiny, 1.0 + aa / c)
        h *= d * c

        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 / np.maximum(tiny, 1.0 + aa * d)
        c = np.maximum(tiny, 1.0 + aa / c)
        delta = d * c
        h *= delta
        if np.max(np.abs(delta - 1.0)) < 3e-12:
            break
    return h


def regularized_beta_array(x: np.ndarray, a: float, b: float) -> np.ndarray:
    x = np.asarray(x, dtype="float64")
    result = np.empty_like(x)
    threshold = (a + 1.0) / (a + b + 2.0)
    bt = np.exp(
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * np.log(x)
        + b * np.log1p(-x)
    )
    lower = x < threshold
    result[lower] = bt[lower] * betacf_array(a, b, x[lower]) / a
    result[~lower] = 1.0 - (
        bt[~lower] * betacf_array(b, a, 1.0 - x[~lower]) / b
    )
    return np.clip(result, 0.0, 1.0).astype("float32")


def two_sided_pearson_p(r: np.ndarray, n: np.ndarray) -> np.ndarray:
    out = np.full(r.shape, np.nan, dtype="float32")
    valid = np.isfinite(r) & np.isfinite(n) & (n >= 4) & (np.abs(r) < 1)
    if np.any(valid):
        n_int = np.where(np.isfinite(n), n, -9999).astype("int32")
        for n_value in np.unique(n_int[valid]):
            use = valid & (n_int == n_value)
            x = np.clip(1.0 - r[use] * r[use], 1e-12, 1.0)
            out[use] = regularized_beta_array(
                x,
                (n_value - 2) / 2.0,
                0.5,
            )
    perfect = np.isfinite(r) & np.isfinite(n) & (n >= 4) & (np.abs(r) >= 1)
    out[perfect] = 0.0
    return out


def pixel_stats(y: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Calculate slope, Pearson r, two-sided p and n for y[time,row,col]."""
    valid = np.isfinite(y)
    n = valid.sum(axis=0).astype("float32")

    x3 = x[:, None, None].astype("float32")
    y0 = np.where(valid, y, 0.0).astype("float32")
    x0 = np.where(valid, x3, 0.0).astype("float32")

    sx = x0.sum(axis=0)
    sy = y0.sum(axis=0)
    sxx = (x0 * x0).sum(axis=0)
    syy = (y0 * y0).sum(axis=0)
    sxy = (x0 * y0).sum(axis=0)

    den_x = n * sxx - sx * sx
    den_y = n * syy - sy * sy
    numerator = n * sxy - sx * sy

    slope = np.full(n.shape, np.nan, dtype="float32")
    r = np.full(n.shape, np.nan, dtype="float32")
    ok = (n >= 4) & (den_x > 0) & (den_y > 0)
    slope[ok] = (numerator[ok] / den_x[ok]).astype("float32")
    r[ok] = (
        numerator[ok] / np.sqrt(den_x[ok] * den_y[ok])
    ).astype("float32")
    r = np.clip(r, -1.0, 1.0)
    p = two_sided_pearson_p(r, n)
    return slope, r, p, n


def expected_outputs() -> list[Path]:
    return [
        stat_raster(component, stat)
        for component, _ in COMPONENTS
        for stat in ["slope", "r", "p", "n"]
    ]


def compute_correlations() -> None:
    if not FORCE_RECOMPUTE and all(path.exists() for path in expected_outputs()):
        print("Existing Figure 7 persistent-coupling rasters found; skipping recomputation.")
        return

    shrinkage = load_shrinkage()

    with rasterio.open(annual_raster(2001)) as ref:
        width, height = ref.width, ref.height
        profile = ref.profile.copy()
        factor = infer_scale_factor(ref)

    profile.update(
        count=1,
        dtype="float32",
        nodata=np.nan,
        compress="deflate",
        predictor=2,
        tiled=True,
    )

    with ExitStack() as stack:
        sources = {
            year: stack.enter_context(rasterio.open(annual_raster(year)))
            for year in ALL_YEARS
        }
        event_src = stack.enter_context(rasterio.open(FIG5_EVENT_YEAR))

        writers = {
            f"{component}_{stat}": stack.enter_context(
                rasterio.open(stat_raster(component, stat), "w", **profile)
            )
            for component, _ in COMPONENTS
            for stat in ["slope", "r", "p", "n"]
        }

        for row0 in range(0, height, BLOCK_HEIGHT):
            h = min(BLOCK_HEIGHT, height - row0)
            window = Window(0, row0, width, h)

            event_year = event_src.read(1, window=window)
            persistent_mask = event_year > 0

            stacks = {"BS": [], "PV": [], "NPV": []}

            for year in ALL_YEARS:
                src = sources[year]
                for component, _ in COMPONENTS:
                    arr = src.read(BANDS[component], window=window).astype("float32")
                    if src.nodata is not None:
                        arr[arr == src.nodata] = np.nan
                    arr *= factor
                    arr[(arr < -1.0) | (arr > 101.0)] = np.nan
                    arr[~persistent_mask] = np.nan
                    stacks[component].append(arr)

            for component, _ in COMPONENTS:
                data = np.stack(stacks[component], axis=0)
                slope, r, p, n = pixel_stats(data, shrinkage)


                slope[~persistent_mask] = np.nan
                r[~persistent_mask] = np.nan
                p[~persistent_mask] = np.nan
                n[~persistent_mask] = np.nan

                writers[f"{component}_slope"].write(slope, 1, window=window)
                writers[f"{component}_r"].write(r, 1, window=window)
                writers[f"{component}_p"].write(p, 1, window=window)
                writers[f"{component}_n"].write(n, 1, window=window)

            print(f"Processed rows {row0:,} to {row0 + h - 1:,} of {height:,}", flush=True)


def summarize_outputs() -> pd.DataFrame:
    rows = []
    with rasterio.open(FIG5_EVENT_YEAR) as evt:
        event = evt.read(1)
        n_persistent_total = int((event > 0).sum())

    for component, label in COMPONENTS:
        with rasterio.open(stat_raster(component, "r")) as src:
            r = src.read(1).astype("float32")
        with rasterio.open(stat_raster(component, "p")) as src:
            p = src.read(1).astype("float32")
        with rasterio.open(stat_raster(component, "slope")) as src:
            slope = src.read(1).astype("float32")
        with rasterio.open(stat_raster(component, "n")) as src:
            n = src.read(1).astype("float32")

        valid = np.isfinite(r)
        sig_pos = valid & np.isfinite(p) & (p < 0.05) & (r > 0)
        sig_neg = valid & np.isfinite(p) & (p < 0.05) & (r < 0)
        n_valid = int(valid.sum())

        rows.append(
            {
                "period": "2001-2024",
                "component": component,
                "label": label,
                "persistence_source": "Figure5_persistent_event_year.tif",
                "n_figure5_persistent_pixels": n_persistent_total,
                "n_valid_correlation_pixels": n_valid,
                "median_n_years": float(np.nanmedian(n)) if n_valid else np.nan,
                "mean_r": float(np.nanmean(r)) if n_valid else np.nan,
                "median_r": float(np.nanmedian(r)) if n_valid else np.nan,
                "mean_slope_pp_per_shrinkage_pct": float(np.nanmean(slope)) if n_valid else np.nan,
                "median_slope_pp_per_shrinkage_pct": float(np.nanmedian(slope)) if n_valid else np.nan,
                "n_sig_positive": int(sig_pos.sum()),
                "n_sig_negative": int(sig_neg.sum()),
                "pct_sig_positive": float(sig_pos.sum() / n_valid * 100.0) if n_valid else np.nan,
                "pct_sig_negative": float(sig_neg.sum() / n_valid * 100.0) if n_valid else np.nan,
            }
        )

    frame = pd.DataFrame(rows)
    path = TABLE_DIR / "Figure7_persistent_coupling_summary_2001_2024.csv"
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    return frame


def load_boundaries():
    import geopandas as gpd
    countries = gpd.read_file(COUNTRY_SHP) if COUNTRY_SHP.exists() else None
    subbasins = gpd.read_file(BASIN_SHP)
    if countries is not None and countries.crs is None:
        countries = countries.set_crs("EPSG:4326")
    if subbasins.crs is None:
        subbasins = subbasins.set_crs("EPSG:4326")
    if countries is not None:
        countries = countries.to_crs("EPSG:4326")
    subbasins = subbasins.to_crs("EPSG:4326")
    return countries, subbasins


def add_boundaries(ax, countries, subbasins) -> None:
    if countries is not None:
        countries.boundary.plot(ax=ax, color="0.70", linewidth=0.55, zorder=4)
    subbasins.boundary.plot(ax=ax, color="#083c9b", linewidth=1.15, zorder=5)


def degree_e(x: float, pos: int) -> str:
    return f"{int(x)}°E"


def degree_n(y: float, pos: int) -> str:
    return f"{int(y)}°N"


def format_map_axis(ax: plt.Axes, row: int, col: int) -> None:
    mean_lat = (VIEW_EXTENT[2] + VIEW_EXTENT[3]) / 2.0
    geographic_aspect = 1.0 / np.cos(np.deg2rad(mean_lat))
    ax.set_aspect(geographic_aspect, adjustable="box")
    ax.set_xlim(VIEW_EXTENT[0], VIEW_EXTENT[1])
    ax.set_ylim(VIEW_EXTENT[2], VIEW_EXTENT[3])
    ax.set_xticks([55, 65, 75])
    ax.set_yticks([35, 40, 45, 50])
    ax.xaxis.set_major_formatter(FuncFormatter(degree_e))
    ax.yaxis.set_major_formatter(FuncFormatter(degree_n))
    if row == 0:
        ax.tick_params(axis="x", labelbottom=False)
    if col > 0:
        ax.tick_params(axis="y", labelleft=False)
    else:
        ax.tick_params(axis="y", labelrotation=90)
    ax.grid(True, color="0.86", linestyle="--", linewidth=0.65, zorder=0)
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_color("black")


def add_panel_letter(ax: plt.Axes, letter: str) -> None:
    ax.text(
        0.015,
        0.98,
        f"({letter})",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=16,
        fontweight="bold",
        zorder=10,
    )


def read_raster(path: Path) -> tuple[np.ndarray, list[float]]:
    with rasterio.open(path) as src:
        arr = src.read(1).astype("float32")
        if src.nodata is not None and np.isfinite(src.nodata):
            arr[arr == src.nodata] = np.nan
        extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]
    return arr, extent


def make_figure() -> Path:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 12,
            "axes.linewidth": 1.0,
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
        }
    )

    countries, subbasins = load_boundaries()

    fig, axes = plt.subplots(2, 3, figsize=(13.0, 7.8), dpi=180, constrained_layout=False)
    fig.subplots_adjust(
        left=0.055,
        right=0.885,
        bottom=0.095,
        top=0.94,
        wspace=0.08,
        hspace=0.09,
    )

    letters = iter("abcdef")
    r_image = None
    p_image = None

    for col, (component, title) in enumerate(COMPONENTS):
        r, extent = read_raster(stat_raster(component, "r"))
        p, _ = read_raster(stat_raster(component, "p"))

        ax = axes[0, col]
        r_image = ax.imshow(
            np.ma.masked_invalid(r),
            extent=extent,
            origin="upper",
            cmap="RdBu_r",
            vmin=-1.0,
            vmax=1.0,
            interpolation="nearest",
            zorder=1,
        )
        add_boundaries(ax, countries, subbasins)
        format_map_axis(ax, 0, col)
        add_panel_letter(ax, next(letters))
        ax.set_title(title, fontsize=18, pad=8)

        ax = axes[1, col]
        p_display = p.copy()
        finite = np.isfinite(p_display)
        p_display[finite] = np.clip(p_display[finite], 0.001, 1.0)
        p_image = ax.imshow(
            np.ma.masked_invalid(p_display),
            extent=extent,
            origin="upper",
            cmap="YlGnBu_r",
            norm=LogNorm(vmin=0.001, vmax=1.0),
            interpolation="nearest",
            zorder=1,
        )
        add_boundaries(ax, countries, subbasins)
        format_map_axis(ax, 1, col)
        add_panel_letter(ax, next(letters))

    cax_r = fig.add_axes([0.897, 0.525, 0.014, 0.415])
    cbar_r = fig.colorbar(r_image, cax=cax_r)
    cbar_r.set_ticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    cbar_r.set_label(r"Pearson $r$", fontsize=14, rotation=90, labelpad=12)
    cbar_r.ax.tick_params(labelsize=12)

    cax_p = fig.add_axes([0.897, 0.095, 0.014, 0.415])
    cbar_p = fig.colorbar(p_image, cax=cax_p)
    cbar_p.set_ticks([0.001, 0.01, 0.05, 0.1, 1.0])
    cbar_p.set_ticklabels(["0.001", "0.01", "0.05", "0.1", "1"])
    cbar_p.set_label(r"$p$ value", fontsize=14, rotation=90, labelpad=12)
    cbar_p.ax.tick_params(labelsize=12)

    stem = FIGDIR / "Figure7_persistent_degradation_coupling_reusing_Figure5"
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(
        stem.with_suffix(".tiff"),
        dpi=600,
        bbox_inches="tight",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)
    return stem


def write_method_note(summary: pd.DataFrame, figure_stem: Path) -> None:
    text = f"""# Figure 7 persistent degradation coupling

Persistence definition
The spatial domain is inherited directly from Figure 5 by using:
{FIG5_EVENT_YEAR}

A pixel is retained when Figure 5 assigned event_year > 0. Therefore Figure 7
uses exactly the same persistence logic and exactly the same Aral water-union
exclusion as Figure 5.

Correlation definition
For each retained pixel, Pearson r is calculated across the annual 2001 to 2024
series between lake shrinkage percentage and the annual local BS, PV or NPV
fraction. The calculation uses all valid annual observations in the 24-year
record. Pixels outside the Figure 5 persistent-event mask are set to NoData.

Interpretation
Positive BS r means that BS tended to be higher in years with stronger lake
shrinkage. Negative PV r means that PV tended to be lower in years with stronger
lake shrinkage. NPV can have either sign and should be interpreted together with
Figure 5's vegetation-senescence pathway.

The p maps show exact two-sided Pearson p values. These are descriptive spatial
statistics and are not corrected for multiple pixel-wise comparisons.

Figure
{figure_stem}

Summary
{summary.to_csv(index=False)}
"""
    (OUTDIR / "Figure7_method_note.md").write_text(text, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    validate_inputs()
    compute_correlations()
    summary = summarize_outputs()
    figure_stem = make_figure()
    write_method_note(summary, figure_stem)

    print("\nFigure 7 completed.")
    print(f"Figure: {figure_stem}")
    print("\nSummary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
