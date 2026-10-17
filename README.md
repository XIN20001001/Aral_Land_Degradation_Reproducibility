# Lake Shrinkage Amplifies Land Degradation in the Aral Sea Basin

Research code and figure-source data for the 2001–2024 analysis in the manuscript. The archived tables and spatial outputs support the reported figure values and let readers render compact versions of Figures 1–7 without repeating basin-wide remote-sensing processing. This is a figure-source and analysis package, not a complete copy of the public source imagery or every intermediate raster.

## Install

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

Activate the environment before running the commands below. Python 3.11 or newer is recommended. The scripts use paths relative to this directory.

## Reproduce and check the figures

```bash
python scripts/analysis/verify_results.py --bootstrap
python scripts/figures/render_figures.py
```

The first command recomputes the Figure 4 OLS slope, Figure 6 Pearson and climate-adjusted partial correlations and their 3,000-replicate moving-block confidence intervals, and checks the Figure 7 raster statistics. It writes `outputs/analysis/verification.json`. The second command writes `outputs/figures/Figure_1.png` through `Figure_7.png`. Use `--figure 6` to render one figure. The layouts are compact reproductions from the original numerical and spatial figure-source data; journal production styling may differ.

| Figure | Archived inputs | Analysis represented |
|---|---|---|
| 1 | Water-area series; basin and lake-extent vector boundaries | Lake extent and annual area |
| 2 | FROMGLC30 annual areas and transition matrix | Class changes, excluding the Aral water union |
| 3 | First persistent-transition maps, native-grid area table, and Aral water-union mask | Four transition pathways and subbasin areas |
| 4 | Monthly/annual MESMA fraction tables | BS, PV, NPV and annual OLS trends |
| 5 | Persistent-event year and three-component rasters | Ternary response maps and pathway shares |
| 6 | Persistent-mask MESMA, water and ERA5-Land monthly table | Pearson and meteorology-adjusted partial correlations |
| 7 | Six full-resolution Pearson `r` and exact two-sided `p` rasters | Pixelwise coupling inside the Figure 5 mask |

## Full-resolution processing

Place the original downloaded inputs under `data/raw/` as described in [data/README.md](data/README.md). Downstream analysis scripts are retained in:

```bash
python scripts/analysis/waterbody_daily.py --skip-figures
python scripts/analysis/land_cover_persistence.py
python scripts/analysis/mesma_persistence.py
python scripts/analysis/pixelwise_correlation.py
```

These are optional, resource-intensive steps and were **not rerun** for this release. Their outputs go to `outputs/full/`. `land_cover_persistence.py` uses the original two-year source, three-year target and 2020–2024 confirmation rules. `mesma_persistence.py` retains the original component thresholds and first-event logic. `pixelwise_correlation.py` reuses the Figure 5 event mask and computes exact two-sided pixel p values. The supplied figure-source files allow the immediate workflow above without these runs.

The downstream terrestrial analyses exclude only the pixels in the archived `data/figure_source/aral_water_union_30m.tif`, representing the study's 2001–2024 observed-water union. This file is identical to the exclusion mask used for the final results. The compact package supplies the final mask directly; it does not reproduce the upstream delineation of that mask from annual water observations.

## External data and code

- [FROMGLC30 V002](https://doi.org/10.12436/iEarth.0000.20251229.000066.v1), annual land cover, 2001–2024.
- [Global 500 m Daily Water Body Mapping Dataset](https://data-starcloud.pcl.ac.cn/iearthdata/9), daily water observations, 2001–2024.
- [MODIS MCD43A4 Collection 6.1](https://doi.org/10.5067/MODIS/MCD43A4.061), reflectance for MESMA.
- [Sun et al. MESMA code and supplied endmembers](https://doi.org/10.5281/zenodo.10796386). The authors ran this unchanged workflow in Google Earth Engine for the entire 2001–2024 study period, using MODIS MCD43A4 Collection 6.1 reflectance to generate this study's fraction data for every year. Sun et al.'s separately published 2001–2022 fraction product was not used as an input. The third-party code and endmembers are linked rather than redistributed, and the study's annual fraction rasters are not included in this compact package.
- [ERA5-Land monthly aggregated data](https://doi.org/10.24381/cds.68d2bb30), precipitation, 2 m temperature and dew point; VPD and monthly anomalies enter Figure 6.

The archived research-generated inputs in `data/figure_source/` are listed in [data/README.md](data/README.md). No third-party raw imagery, MESMA code, or endmember data are redistributed. The full upstream reflectance-to-fraction and Earth Engine export workflow cannot be rerun from this package alone. The development repository is [XIN20001001/Aral_Land_Degradation_Reproducibility](https://github.com/XIN20001001/Aral_Land_Degradation_Reproducibility); a versioned archival DOI has not yet been assigned.

## Availability and citation

This public repository contains the figure-source data, derived tables, spatial outputs, and analysis scripts described above. The manuscript links directly to [this repository](https://github.com/XIN20001001/Aral_Land_Degradation_Reproducibility) in its Data Availability Statement. Citation metadata is provided in `CITATION.cff`. A versioned archival DOI has not yet been assigned; update the manuscript citation and this file when an archived release receives one.
