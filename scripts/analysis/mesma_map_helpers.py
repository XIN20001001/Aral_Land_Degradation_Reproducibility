"""Map helpers used by the original Figure 5 persistence algorithm."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import shapefile


def shape_bbox(path: Path) -> tuple[float, float, float, float]:
    xmin, ymin, xmax, ymax = shapefile.Reader(str(path), encoding="utf-8").bbox
    return float(xmin), float(xmax), float(ymin), float(ymax)


def _draw_boundaries(ax: plt.Axes, path: Path, *, color: str, linewidth: float) -> None:
    if not path.exists():
        return
    for shape in shapefile.Reader(str(path), encoding="utf-8").iterShapes():
        points = np.asarray(shape.points, dtype=float)
        ends = list(shape.parts) + [len(points)]
        for start, end in zip(ends[:-1], ends[1:]):
            segment = points[start:end]
            if len(segment) > 1:
                ax.plot(segment[:, 0], segment[:, 1], color=color, linewidth=linewidth, zorder=5)


def style_map_axis(
    ax: plt.Axes,
    map_extent: tuple[float, float, float, float],
    view_extent: tuple[float, float, float, float],
    basin_shp: Path,
    countries_shp: Path,
) -> None:
    ax.set_xlim(view_extent[0], view_extent[1])
    ax.set_ylim(view_extent[2], view_extent[3])
    mean_lat = .5 * (view_extent[2] + view_extent[3])
    ax.set_aspect(1 / np.cos(np.radians(mean_lat)), adjustable="box")
    ax.set_xticks(np.arange(60, 81, 5), [f"{x}°E" for x in range(60, 81, 5)])
    ax.set_yticks(np.arange(35, 51, 5), [f"{y}°N" for y in range(35, 51, 5)])
    ax.grid(color=".78", linewidth=.35, linestyle="--", zorder=2)
    _draw_boundaries(ax, countries_shp, color=".38", linewidth=.35)
    _draw_boundaries(ax, basin_shp, color=".05", linewidth=.65)
