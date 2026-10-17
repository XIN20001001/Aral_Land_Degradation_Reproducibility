"""Render Figures 1–7 from the archived figure-source data."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap, LogNorm
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
import geopandas as gpd


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "figure_source"
OUT = ROOT / "outputs" / "figures"
PERIODS = ["2001-2005", "2006-2010", "2011-2015", "2016-2020", "2021-2024", "2001-2024"]
METRICS = ["BS", "PV", "NPV"]


def table(name: str) -> pd.DataFrame:
    path = DATA / name
    if not path.is_file():
        raise FileNotFoundError(f"Required figure source is missing: {path}")
    return pd.read_csv(path)


def save(fig: plt.Figure, number: int) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"Figure_{number}.png", dpi=250, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def figure1() -> None:
    water = table("figure1_water_annual.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [1.35, 1]})
    boundaries = ROOT / "data" / "boundaries"
    basin = gpd.read_file(boundaries / "AralSeaBasinS.shp")
    water_2001 = gpd.read_file(boundaries / "Aral_2001_water_occurrence_extent.gpkg")
    basin.boundary.plot(ax=axes[0], color="#647681", linewidth=.7)
    water_2001.plot(ax=axes[0], color="#3b86a5", edgecolor="none", label="2001 water extent")
    axes[0].set(xlabel="Longitude", ylabel="Latitude", title="(a) Aral Sea Basin and water extent")
    axes[0].set_xlim(56, 80)
    axes[0].set_ylim(33, 50)
    axes[0].set_aspect("equal")
    ax = axes[1]
    ax.fill_between(water.year, water.p10_water_area_km2, water.p90_water_area_km2,
                    color="#88aabb", alpha=.35, label="Daily 10th–90th percentiles")
    ax.plot(water.year, water.mean_water_area_km2, "o-", color="#175c83", ms=3,
            label="Annual mean")
    ax.set(xlabel="Year", ylabel="Aral Sea water area (km²)", title="(b) Water surface area")
    ax.legend(frameon=False, fontsize=8)
    save(fig, 1)


def figure2() -> None:
    annual = table("figure2_landcover_annual.csv")
    matrix = table("figure2_transition_matrix.csv").set_index("from_class")
    classes = table("figure2_net_change.csv").class_name.tolist()
    years = [2001, 2005, 2010, 2015, 2020, 2024]
    piv = annual.query("region == 'Aral Sea Basin'").pivot(index="class_name", columns="year", values="area_km2")
    intervals = np.column_stack([(piv[b] - piv[a]).reindex(classes).to_numpy() for a, b in zip(years[:-1], years[1:])])
    total = (piv[2024] - piv[2001]).reindex(classes).to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.3), gridspec_kw={"width_ratios": [1.35, 1]})
    y = np.arange(len(classes))
    axes[0].barh(y, total, color="#bed4d5", label="2001–2024 net")
    for j in range(5):
        axes[0].barh(y + (j - 2) * .13, intervals[:, j], height=.11,
                     label=PERIODS[j])
    axes[0].set(yticks=y, yticklabels=classes, xlabel="Area change (km²)", title="(a) Land-cover change")
    axes[0].axvline(0, color="black", lw=.7)
    axes[0].legend(fontsize=7, frameon=False, ncol=2)
    ordered = matrix.reindex(index=classes, columns=classes).fillna(0)
    percent = ordered.div(ordered.sum(axis=1).replace(0, np.nan), axis=0) * 100
    im = axes[1].imshow(np.log1p(percent), cmap="YlOrRd", vmin=0, vmax=np.log1p(100))
    axes[1].set(xticks=y, yticks=y, xticklabels=classes, yticklabels=classes,
                title="(b) 2001–2024 transition matrix", xlabel="2024", ylabel="2001")
    axes[1].tick_params(axis="x", labelrotation=60)
    cb = fig.colorbar(im, ax=axes[1], fraction=.045)
    cb.set_ticks(np.log1p([0, 1, 5, 20, 50, 100]), labels=["0", "1", "5", "20", "50", "100"])
    cb.set_label("Share of 2001 class (%)")
    fig.tight_layout()
    save(fig, 2)


def figure3() -> None:
    maps = np.load(DATA / "figure3_maps.npz")
    areas = table("figure3_areas.csv")
    colors = ["#ffffff", "#b2182b", "#fdb863", "#ef8a62", "#2166ac"]
    labels = ["Natural-cover degradation", "Land-use intensification",
              "Hydrological degradation", "Aral Sea lakebed exposure"]
    fig, axes = plt.subplots(3, 3, figsize=(12, 11))
    for ax, period, letter in zip(axes.flat[:6], PERIODS, "abcdef"):
        canvas = maps[f"canvas_{period}"]
        ax.imshow(canvas, cmap=ListedColormap(colors), norm=BoundaryNorm(np.arange(-.5, 5.5), 5))
        ax.set_title(f"({letter}) {period}", loc="left")
        ax.axis("off")
    groups = areas.query("region == 'Aral Sea Basin'").pivot(index="period", columns="category", values="area_km2")
    groups = groups.reindex(PERIODS).fillna(0)
    for name, color in zip(labels, colors[1:]):
        axes[2, 0].plot(PERIODS[:5], groups.loc[PERIODS[:5], name], "o-", color=color, label=name)
    axes[2, 0].tick_params(axis="x", rotation=45)
    axes[2, 0].set(ylabel="Area (km²)", title="(g) Area by period")
    regions = areas.query("period == '2001-2024' and region != 'Aral Sea Basin'")
    region_piv = regions.pivot(index="region", columns="category", values="area_km2").fillna(0)
    bottom = np.zeros(len(region_piv))
    for name, color in zip(labels, colors[1:]):
        values = region_piv[name].to_numpy()
        axes[2, 1].bar(region_piv.index, values, bottom=bottom, color=color, label=name)
        bottom += values
    axes[2, 1].tick_params(axis="x", rotation=45)
    axes[2, 1].set(ylabel="Area (km²)", title="(h) Full-period area by subbasin")
    axes[2, 2].axis("off")
    axes[2, 2].legend(*axes[2, 1].get_legend_handles_labels(), loc="center", frameon=False)
    fig.tight_layout()
    save(fig, 3)


def figure4() -> None:
    monthly = table("figure4_monthly.csv")
    basin = table("figure4_annual_basin.csv")
    regions = table("figure4_annual_subbasins.csv")
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    for i, metric in enumerate(METRICS):
        ax = axes[i, 0]
        x = monthly.year + (monthly.month - .5) / 12
        ax.plot(x, monthly[metric], lw=.6, color="#9bafbc")
        ax.plot(basin.year, basin[metric], "o-", ms=2.5, color="#245b85")
        slope, intercept = np.polyfit(basin.year, basin[metric], 1)
        ax.plot(basin.year, slope * basin.year + intercept, "--", color="#9b3030",
                label=f"OLS {slope:.2f} pp yr⁻¹")
        ax.set(ylabel=f"{metric} (%)", title=f"({chr(97+2*i)}) Basin {metric}")
        ax.legend(frameon=False, fontsize=8)
        ax = axes[i, 1]
        for region, part in regions.groupby("subbasin"):
            ax.plot(part.year, part[metric], "o-", ms=2, lw=1, label=region)
        ax.set(ylabel=f"{metric} (%)", title=f"({chr(98+2*i)}) Subbasins {metric}")
        if i == 0:
            ax.legend(frameon=False, fontsize=7, ncol=2)
    axes[-1, 0].set_xlabel("Year")
    axes[-1, 1].set_xlabel("Year")
    fig.tight_layout()
    save(fig, 4)


def figure5() -> None:
    periods = table("figure5_periods.csv").set_index("period")
    with rasterio.open(DATA / "figure5_event_year.tif") as events, rasterio.open(DATA / "figure5_components.tif") as components:
        shape = (round(events.height * min(1, 2400 / events.width)),
                 round(events.width * min(1, 2400 / events.width)))
        year = events.read(1, out_shape=shape, resampling=Resampling.nearest)
        strengths = components.read(out_shape=(3, *shape), resampling=Resampling.nearest).astype(np.float32)
        strengths[strengths == components.nodata] = np.nan
        edges = np.linspace(0, 300, 6001)
        histogram = np.zeros(len(edges) - 1, dtype=np.int64)
        for _, window in components.block_windows(1):
            block = components.read([1, 2, 3], window=window).astype(np.float32)
            valid = np.all(np.isfinite(block) & (block != components.nodata), axis=0)
            total = block.sum(axis=0)
            values = total[valid & (total > 0)]
            if values.size:
                histogram += np.histogram(np.clip(values, 0, 300), bins=edges)[0]
    index = int(np.searchsorted(np.cumsum(histogram), .95 * histogram.sum()))
    strength_p95 = float((edges[index] + edges[index + 1]) / 2)
    fig, axes = plt.subplots(3, 3, figsize=(12, 11))
    bins = [(2001, 2005), (2006, 2010), (2011, 2015), (2016, 2020), (2021, 2024), (2001, 2024)]
    corner_colors = np.array([[0xD5, 0x5E, 0x00], [0x00, 0x72, 0xB2], [0xCC, 0x79, 0xA7]], dtype=np.float32) / 255
    total = strengths.sum(axis=0)
    for ax, (start, end), period, letter in zip(axes.flat[:6], bins, PERIODS, "abcdef"):
        valid = ((year >= start) & (year <= end) & np.all(np.isfinite(strengths), axis=0)
                 & np.isfinite(total) & (total > 0))
        weights = np.zeros_like(strengths)
        weights[:, valid] = strengths[:, valid] / total[valid]
        hue = np.einsum("chw,ck->hwk", weights, corner_colors)
        amount = np.zeros_like(total)
        amount[valid] = np.sqrt(np.clip(total[valid] / strength_p95, 0, 1))
        display = np.array([.96, .96, .94]) * (1 - amount[..., None]) + hue * amount[..., None]
        display[~valid] = 1
        ax.imshow(np.clip(display, 0, 1), interpolation="nearest")
        ax.set_title(f"({letter}) {period}", loc="left")
        ax.axis("off")
    shares = periods.loc[PERIODS[:5]]
    bottom = np.zeros(5)
    for component, color, label in [("BS", "#d55e00", "Bare-soil exposure"),
                                    ("PV", "#0072b2", "Green-vegetation loss"),
                                    ("NPV", "#cc79a7", "Vegetation senescence")]:
        value = shares[f"{component}_dominant_share_pct"].to_numpy()
        axes[2, 0].bar(PERIODS[:5], value, bottom=bottom, color=color, label=label)
        bottom += value
    axes[2, 0].tick_params(axis="x", rotation=45)
    axes[2, 0].set(ylabel="Share (%)", title="(g) Persistent response pathways")
    axes[2, 1].axis("off")
    axes[2, 1].legend(*axes[2, 0].get_legend_handles_labels(), loc="center", frameon=False)
    axes[2, 2].axis("off")
    fig.tight_layout()
    save(fig, 5)


def figure6() -> None:
    annual = table("figure6_annual.csv")
    pearson = table("figure6_pearson.csv")
    partial = table("figure6_partial.csv")
    fig, axes = plt.subplots(3, 1, figsize=(11, 10))
    water_ax = axes[0].twinx()
    water_ax.plot(annual.year, annual.water_area_km2, "k-", label="Water area")
    water_ax.set_ylabel("Water area (km²)")
    colors = {"delta_BS": "#b2182b", "delta_PV": "#38936d", "delta_NPV": "#8569a5"}
    for metric in ["delta_BS", "delta_PV", "delta_NPV"]:
        axes[0].plot(annual.year, annual[metric], label=metric.replace("delta_", "Δ"), color=colors[metric])
    axes[0].set(ylabel="MESMA change (pp)", title="(a) Annual water area and persistent-area MESMA change")
    axes[0].legend(frameon=False, ncol=3, fontsize=8)
    for ax, df, title in [(axes[1], pearson, "(b) Monthly Pearson r"),
                          (axes[2], partial, "(c) Climate-adjusted partial r")]:
        for metric, part in df.groupby("metric"):
            part = part.sort_values("month")
            ax.plot(part.month, part.r, "o-", label=metric.replace("delta_", "Δ"), color=colors[metric])
            ax.fill_between(part.month, part.ci95_low, part.ci95_high,
                            color=colors[metric], alpha=.12)
        ax.axhline(0, color="black", lw=.5)
        ax.set(xlim=(1, 12), xticks=range(1, 13), ylabel="Correlation", title=title)
    axes[2].set_xlabel("Calendar month")
    axes[1].legend(frameon=False, ncol=3, fontsize=8)
    fig.tight_layout()
    save(fig, 6)


def figure7() -> None:
    fig, axes = plt.subplots(3, 2, figsize=(12, 12))
    for i, metric in enumerate(METRICS):
        for j, stat in enumerate(["r", "p"]):
            path = DATA / "figure7" / f"{metric}_{stat}_vs_shrinkage_persistent_Fig5_2001_2024.tif"
            with rasterio.open(path) as raster:
                shape = (raster.height // 4, raster.width // 4)
                values = raster.read(1, out_shape=shape, resampling=Resampling.nearest, masked=True)
            ax = axes[i, j]
            if stat == "r":
                image = ax.imshow(values, cmap="RdBu_r", vmin=-1, vmax=1)
                label = "Pearson r"
            else:
                image = ax.imshow(values, cmap="viridis_r", vmin=0, vmax=.05)
                label = "Two-sided p"
            ax.set_title(f"({chr(97+2*i+j)}) {metric} {label}", loc="left")
            ax.axis("off")
            fig.colorbar(image, ax=ax, fraction=.035, label=label)
    fig.tight_layout()
    save(fig, 7)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure", type=int, choices=range(1, 8), help="Render one figure; default: all")
    args = parser.parse_args()
    figures = [figure1, figure2, figure3, figure4, figure5, figure6, figure7]
    selected = [args.figure] if args.figure else range(1, 8)
    for number in selected:
        figures[number - 1]()
        print(f"Wrote {OUT / f'Figure_{number}.png'}")


if __name__ == "__main__":
    main()
