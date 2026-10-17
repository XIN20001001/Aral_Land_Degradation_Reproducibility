"""Recompute published table statistics from archived research inputs."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from scipy import stats


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "figure_source"
OUT = ROOT / "outputs" / "analysis"
METRICS = ["delta_BS", "delta_PV", "delta_NPV"]
CONTROLS = ["precip_anom_mm", "t2m_anom_c", "vpd_anom_kpa"]


def read(name: str) -> pd.DataFrame:
    path = DATA / name
    if not path.is_file():
        raise FileNotFoundError(f"Missing archived research input: {path}")
    return pd.read_csv(path)


def partial_pearson(x: np.ndarray, y: np.ndarray, controls: np.ndarray) -> tuple[float, float]:
    """Use the residualized Pearson estimator and n-k-2 degrees of freedom."""
    n, k = controls.shape
    if n <= k + 2:
        return math.nan, math.nan
    mean = controls.mean(axis=0)
    sd = controls.std(axis=0, ddof=1)
    if np.any(sd == 0):
        raise ValueError("A climate-control series has zero variance")
    design = np.column_stack([np.ones(n), (controls - mean) / sd])
    rx = x - design @ np.linalg.lstsq(design, x, rcond=None)[0]
    ry = y - design @ np.linalg.lstsq(design, y, rcond=None)[0]
    r = float(np.clip(np.corrcoef(rx, ry)[0, 1], -1, 1))
    t = r * math.sqrt((n - k - 2) / max(1e-15, 1 - r * r))
    return r, float(2 * stats.t.sf(abs(t), n - k - 2))


def block_ci(x: np.ndarray, y: np.ndarray, controls: np.ndarray | None,
             rng: np.random.Generator, reps: int = 3000) -> tuple[float, float]:
    """Original Figure 6 moving-block bootstrap: length 3, 3000 replicates."""
    n, length = len(x), 3
    starts = np.arange(0, n - length + 1)
    values = []
    for _ in range(reps):
        picked = rng.choice(starts, size=math.ceil(n / length), replace=True)
        index = np.concatenate([np.arange(start, start + length) for start in picked])[:n]
        xb, yb = x[index], y[index]
        if controls is None:
            if np.std(xb) == 0 or np.std(yb) == 0:
                continue
            r = float(np.corrcoef(xb, yb)[0, 1])
        else:
            r, _ = partial_pearson(xb, yb, controls[index])
        if np.isfinite(r):
            values.append(r)
    if len(values) < max(100, reps // 4):
        return math.nan, math.nan
    return tuple(float(v) for v in np.percentile(values, [2.5, 97.5]))


def correlations(bootstrap: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    merged = read("figure6_monthly_merged.csv")
    if len(merged) != 288 or merged.duplicated(["year", "month"]).any():
        raise ValueError("Figure 6 requires one row for each month of 2001–2024")
    zero, partial = [], []
    rng = np.random.default_rng(20260923)
    for month in range(1, 13):
        by_month = merged.loc[(merged.month == month) & (merged.year >= 2002)].sort_values("year")
        for metric in METRICS:
            use = by_month[["shrinkage_pct", metric, *CONTROLS]].apply(pd.to_numeric, errors="coerce").dropna()
            x, y = use.shrinkage_pct.to_numpy(float), use[metric].to_numpy(float)
            climate = use[CONTROLS].to_numpy(float)
            r, p = stats.pearsonr(x, y)
            pr, pp = partial_pearson(x, y, climate)
            zr = {"month": month, "metric": metric, "n_years": len(use), "r": r, "p": p}
            pp_row = {"month": month, "metric": metric, "n_years": len(use), "r": pr, "p": pp}
            if bootstrap:
                zl, zh = block_ci(x, y, None, rng)
                pl, ph = block_ci(x, y, climate, rng)
                zr.update(ci95_low=zl, ci95_high=zh)
                pp_row.update(ci95_low=pl, ci95_high=ph)
            zero.append(zr)
            partial.append(pp_row)
    return pd.DataFrame(zero), pd.DataFrame(partial)


def raster_summary() -> dict:
    results = {}
    for metric in ["BS", "PV", "NPV"]:
        base = DATA / "figure7"
        with rasterio.open(base / f"{metric}_r_vs_shrinkage_persistent_Fig5_2001_2024.tif") as rfile, \
             rasterio.open(base / f"{metric}_p_vs_shrinkage_persistent_Fig5_2001_2024.tif") as pfile:
            r, p = rfile.read(1, masked=True), pfile.read(1, masked=True)
        valid = np.isfinite(r) & np.isfinite(p)
        rv, pv = np.asarray(r)[valid], np.asarray(p)[valid]
        results[metric] = {
            "n_valid": int(valid.sum()),
            "mean_r": float(rv.mean()),
            "pct_sig_positive": float(100 * np.mean((rv > 0) & (pv < .05))),
            "pct_sig_negative": float(100 * np.mean((rv < 0) & (pv < .05))),
        }
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap", action="store_true", help="Recompute 3000-replicate Figure 6 confidence intervals")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    checks = {}
    land = read("figure2_net_change.csv").set_index("class_name")
    checks["figure2_bare_change_km2"] = float(land.loc["Bare land", "change_2001_2024_km2"])
    checks["figure2_grass_change_km2"] = float(land.loc["Grassland", "change_2001_2024_km2"])
    f3 = read("figure3_areas.csv")
    checks["figure3_natural_degradation_km2"] = float(f3.query("period == '2001-2024' and region == 'Aral Sea Basin' and category == 'Natural-cover degradation'").area_km2.iloc[0])
    basin = read("figure4_annual_basin.csv")
    slope, intercept, r, p, _ = stats.linregress(basin.year, basin.BS)
    checks["figure4_BS_slope_pp_per_year"] = float(slope)
    checks["figure4_BS_r"] = float(r)
    checks["figure4_BS_p"] = float(p)
    f5 = read("figure5_periods.csv").set_index("period")
    checks["figure5_BS_share_pct"] = float(f5.loc["2001-2024", "BS_dominant_share_pct"])
    zero, partial = correlations(args.bootstrap)
    for name, calculated, archived in [("pearson", zero, read("figure6_pearson.csv")),
                                        ("partial", partial, read("figure6_partial.csv"))]:
        joined = calculated.merge(archived, on=["month", "metric", "n_years"], suffixes=("_new", "_archived"), validate="one_to_one")
        if len(joined) != 36:
            raise AssertionError(f"Figure 6 {name}: incomplete month/metric alignment")
        checks[f"figure6_{name}_max_abs_r_difference"] = float(np.max(np.abs(joined.r_new - joined.r_archived)))
        checks[f"figure6_{name}_max_abs_p_difference"] = float(np.max(np.abs(joined.p_new - joined.p_archived)))
        if args.bootstrap:
            checks[f"figure6_{name}_max_abs_ci_difference"] = float(max(
                np.max(np.abs(joined.ci95_low_new - joined.ci95_low_archived)),
                np.max(np.abs(joined.ci95_high_new - joined.ci95_high_archived)),
            ))
        calculated.to_csv(OUT / f"figure6_{name}_recomputed.csv", index=False)
    checks["figure7"] = raster_summary()
    archived7 = read("figure7_summary.csv").set_index("component")
    for metric, measured in checks["figure7"].items():
        reference = archived7.loc[metric]
        if measured["n_valid"] != int(reference.n_valid_correlation_pixels):
            raise AssertionError(f"Figure 7 {metric}: pixel count differs from archived summary")
        for field in ["mean_r", "pct_sig_positive", "pct_sig_negative"]:
            if not np.isclose(measured[field], reference[field], atol=1e-5):
                raise AssertionError(f"Figure 7 {metric}: {field} differs from archived summary")
    if max(checks["figure6_pearson_max_abs_r_difference"], checks["figure6_partial_max_abs_r_difference"]) > 1e-10:
        raise AssertionError("Figure 6 correlations differ from archived results")
    (OUT / "verification.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
