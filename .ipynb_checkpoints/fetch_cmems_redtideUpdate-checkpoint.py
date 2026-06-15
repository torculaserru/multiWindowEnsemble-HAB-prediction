"""
fetch_cmems_redtide.py
======================
Fetches daily oceanographic variables from Copernicus Marine Service (CMEMS)
for each of the four Eastern Visayas provinces individually, then merges
them into the red-tide ML dataset as province-specific columns.

Data products used
------------------
1. BGC (Biogeochemistry) — dissolved oxygen & chlorophyll-a
   Dataset : cmems_mod_glo_bgc_my_0.25deg_P1D-m   (≤ 2023)
             cmems_mod_glo_bgc_myint_0.25deg_P1D-m (≥ 2024)
   Variables: o2  (dissolved oxygen,   mmol m⁻³)
              chl (chlorophyll-a,       mg m⁻³)

2. PHY (Physics) — sea surface temperature & salinity
   Dataset : cmems_mod_glo_phy_my_0.083deg_P1D-m   (≤ 2023)
             cmems_mod_glo_phy_myint_0.083deg_P1D-m (≥ 2024)
   Variables: thetao (potential temperature, °C)
              so     (practical salinity,    PSU)

Province bounding boxes (coastal waters only, depth 0–1 m)
----------------------------------------------------------
  Eastern Samar  — eastern coast of Samar Island facing Philippine Sea
                   Lat: 10.80 – 12.50 °N   Lon: 125.00 – 126.20 °E

  Western Samar  — western coast of Samar Island facing San Pedro Bay /
                   Leyte Gulf / Samar Sea
                   Lat: 11.10 – 12.20 °N   Lon: 124.60 – 125.20 °E

  Leyte          — coasts of Leyte Island (Leyte Gulf east, Camotes Sea west,
                   Carigara Bay north)
                   Lat: 10.50 – 11.70 °N   Lon: 124.40 – 125.30 °E

  Biliran        — small island province north of Leyte
                   Lat: 11.45 – 11.85 °N   Lon: 124.35 – 124.65 °E

Strategy
--------
- For each province × product × year: fetch → spatial-mean → checkpoint.
- Outputs 4 province-specific columns per oceanographic variable
  (16 new columns total).
- Checkpoints allow safe resume after any crash.
- Final output: easternVisayas_redtide_with_cmems.csv

Output columns added (per province suffix: _eastern_samar, _western_samar,
                                            _leyte, _biliran)
----------------------------------------------------------------------
  dissolved_oxygen_mmol_m3_<province>
  chlorophyll_a_mg_m3_<province>
  sea_surface_temp_C_<province>
  salinity_PSU_<province>

Requirements
------------
    pip install copernicusmarine pandas xarray numpy tqdm pyarrow
"""

import os
import time
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import copernicusmarine
from tqdm import tqdm

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("fetch_cmems_redtide.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


# ── Credentials ───────────────────────────────────────────────────────────────
# Set as environment variables (recommended) OR edit directly here:
#   export COPERNICUSMARINE_SERVICE_USERNAME="your_username"
#   export COPERNICUSMARINE_SERVICE_PASSWORD="your_password"
CMEMS_USERNAME = os.environ.get("COPERNICUSMARINE_SERVICE_USERNAME", "egtorculas@msi.upd.edu.ph")
CMEMS_PASSWORD = os.environ.get("COPERNICUSMARINE_SERVICE_PASSWORD", "Aero*201950066")


# ── File paths ────────────────────────────────────────────────────────────────
INPUT_CSV      = "easternVisayasRedTide.csv"
OUTPUT_CSV     = "easternVisayas_redtide_with_cmems.csv"
CHECKPOINT_DIR = Path("cmems_checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True)


# ── Province bounding boxes ───────────────────────────────────────────────────
#
#  Each entry: (lat_min, lat_max, lon_min, lon_max)
#
#  Eastern Samar  faces the Philippine Sea (Pacific) to the east.
#  The BFAR red-tide closure data aligns with coastal fishing areas,
#  so the boxes include nearshore + inner-shelf waters only.
#
#  Sources: Wikipedia province pages, biliran.gov.ph, DOH Region VIII,
#           PhilGIS administrative boundaries.
#
PROVINCES = {
    "eastern_samar": {
        "label"   : "Eastern Samar",
        "lat_min" : 10.8598443,   # southern tip near Guiuan
        "lat_max" : 11.2956955,   # border with Northern Samar
        "lon_min" : 125.5049239,  # inland boundary with Western Samar
        "lon_max" : 125.6868417,  # ~40 km offshore into Philippine Sea
        # Capital (Borongan): 11.61°N, 125.43°E
    },
    "western_samar": {
        "label"   : "Western Samar",
        "lat_min" : 11.4707438,   # southern end near San Jorge
        "lat_max" : 11.8250938,   # northern boundary near Calbayog
        "lon_min" : 124.7863609,  # Leyte Gulf / San Pedro Bay
        "lon_max" : 125.0992822,  # central Samar island spine
        # Capital (Catbalogan): 11.78°N, 124.89°E
    },
    "leyte": {
        "label"   : "Leyte",
        "lat_min" : 11.2585972,   # southern tip near Abuyog
        "lat_max" : 11.5244861,   # northern tip / Carigara Bay
        "lon_min" : 124.5245676,  # Camotes Sea (western coast)
        "lon_max" : 124.8541862,  # Leyte Gulf (eastern coast)
        # Capital (Tacloban): 11.24°N, 125.00°E
    },
    "biliran": {
        "label"   : "Biliran",
        "lat_min" : 11.4404487,   # southern shore (Carigara Bay)
        "lat_max" : 11.7495210,   # northern shore (Visayas Sea)
        "lon_min" : 124.3072953,  # Biliran Strait (western side)
        "lon_max" : 124.6651716,  # Samar Sea (eastern side)
        # Capital (Naval): 11.58°N, 124.40°E
    },
}

# Surface layer only
DEPTH_MIN = 0.0
DEPTH_MAX = 1.0


# ── Dataset identifiers ───────────────────────────────────────────────────────
BGC_DATASET     = "cmems_mod_glo_bgc_my_0.25deg_P1D-m"
PHY_DATASET     = "cmems_mod_glo_phy_my_0.083deg_P1D-m"

# Multi-year reanalysis covers up to this year; interim used afterwards
MY_CUTOFF_YEAR  = 2023

BGC_VARIABLES   = ["o2", "chl"]
PHY_VARIABLES   = ["thetao", "so"]

# Base rename: CMEMS var → column prefix (province suffix appended later)
VAR_PREFIX = {
    "o2"    : "dissolved_oxygen_mmol_m3",
    "chl"   : "chlorophyll_a_mg_m3",
    "thetao": "sea_surface_temp_C",
    "so"    : "salinity_PSU",
}

# Retry config
MAX_RETRIES = 3
RETRY_DELAY = 30   # seconds


# ── Helpers ───────────────────────────────────────────────────────────────────

def dataset_id(base: str, year: int) -> str:
    """Switch from 'my' (reanalysis) to 'myint' (interim) for years > cutoff."""
    if year > MY_CUTOFF_YEAR:
        return base.replace("_my_", "_myint_")
    return base


def fetch_province_year(
    ds_id: str,
    variables: list[str],
    year: int,
    prov_key: str,
) -> xr.Dataset:
    """
    Download one year of daily data for a single province's bounding box.
    Returns an xr.Dataset spatially averaged to a (time,) series.
    """
    bbox = PROVINCES[prov_key]
    start, end = f"{year}-01-01", f"{year}-12-31"

    log.info("  [%s] %s  %s → %s", prov_key, ds_id, start, end)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            ds = copernicusmarine.open_dataset(
                dataset_id        = ds_id,
                username          = CMEMS_USERNAME,
                password          = CMEMS_PASSWORD,
                variables         = variables,
                minimum_longitude = bbox["lon_min"],
                maximum_longitude = bbox["lon_max"],
                minimum_latitude  = bbox["lat_min"],
                maximum_latitude  = bbox["lat_max"],
                minimum_depth     = DEPTH_MIN,
                maximum_depth     = DEPTH_MAX,
                start_datetime    = start,
                end_datetime      = end,
            )

            # Surface layer
            if "depth" in ds.dims:
                ds = ds.isel(depth=0)
            elif "elevation" in ds.dims:
                ds = ds.isel(elevation=0)

            # Spatial mean → (time,) only
            spatial_dims = [d for d in ds.dims if d in ("latitude", "longitude", "lat", "lon")]
            if spatial_dims:
                ds = ds.mean(dim=spatial_dims, skipna=True)

            return ds

        except Exception as exc:
            log.warning("  Attempt %d/%d failed: %s", attempt, MAX_RETRIES, exc)
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)
            else:
                log.error("  All retries exhausted — %s | %s | %d", prov_key, ds_id, year)
                raise


def ds_to_df(ds: xr.Dataset, variables: list[str], prov_key: str, year: int) -> pd.DataFrame:
    """Convert spatially-averaged xr.Dataset to a tidy DataFrame with province-suffixed columns."""
    time_dim = "time" if "time" in ds.dims else list(ds.dims)[0]
    dates = pd.to_datetime(ds[time_dim].values).normalize().strftime("%Y-%m-%d")

    rows: dict = {"date": dates}
    for var in variables:
        col_name = f"{VAR_PREFIX[var]}_{prov_key}"
        if var in ds:
            rows[col_name] = ds[var].values.astype(float)
        else:
            log.warning("  Variable '%s' missing for %s / %d", var, prov_key, year)
            rows[col_name] = np.nan

    return pd.DataFrame(rows)


def ckpt_path(product: str, prov_key: str, year: int) -> Path:
    return CHECKPOINT_DIR / f"{product}_{prov_key}_{year}.parquet"


def load_ckpt(product: str, prov_key: str, year: int) -> pd.DataFrame | None:
    p = ckpt_path(product, prov_key, year)
    if p.exists():
        log.info("  Checkpoint hit: %s", p.name)
        return pd.read_parquet(p)
    return None


def save_ckpt(df: pd.DataFrame, product: str, prov_key: str, year: int) -> None:
    p = ckpt_path(product, prov_key, year)
    df.to_parquet(p, index=False)
    log.info("  Saved: %s", p.name)


def fetch_product_all_provinces(
    base_ds: str,
    variables: list[str],
    product_tag: str,
    years: list[int],
    base_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    For a given CMEMS product, fetch all provinces × all years.
    Returns a single wide DataFrame keyed on 'date'.
    """
    # Collect per-province DataFrames, then merge on date
    province_dfs: dict[str, list[pd.DataFrame]] = {k: [] for k in PROVINCES}

    for prov_key in PROVINCES:
        log.info("\n── Province: %s ──", PROVINCES[prov_key]["label"])
        for year in tqdm(years, desc=f"{product_tag} | {prov_key}"):
            cached = load_ckpt(product_tag, prov_key, year)
            if cached is not None:
                province_dfs[prov_key].append(cached)
                continue
            try:
                ds_id = dataset_id(base_ds, year)
                ds    = fetch_province_year(ds_id, variables, year, prov_key)
                df    = ds_to_df(ds, variables, prov_key, year)
                save_ckpt(df, product_tag, prov_key, year)
                province_dfs[prov_key].append(df)
            except Exception as exc:
                log.error("FAILED %s | %s | %d: %s", product_tag, prov_key, year, exc)
                # Fill with NaN so the merge remains intact
                year_dates = base_df.loc[base_df["year"] == year, "date"].tolist()
                null_cols  = {f"{VAR_PREFIX[v]}_{prov_key}": np.nan for v in variables}
                null_cols["date"] = year_dates
                province_dfs[prov_key].append(pd.DataFrame(null_cols))

    # Concatenate years per province then merge all provinces on date
    merged = None
    for prov_key, frames in province_dfs.items():
        prov_df = pd.concat(frames, ignore_index=True)
        merged  = prov_df if merged is None else merged.merge(prov_df, on="date", how="outer")

    return merged


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    log.info("Loading base dataset: %s", INPUT_CSV)
    base_df = pd.read_csv(INPUT_CSV)
    base_df["date"] = pd.to_datetime(base_df["date"]).dt.strftime("%Y-%m-%d")

    years = sorted(base_df["year"].unique().tolist())
    log.info(
        "Date range: %s → %s  |  %d years, %d days",
        base_df["date"].iloc[0], base_df["date"].iloc[-1],
        len(years), len(base_df),
    )

    # Log province bounding boxes for transparency
    log.info("\nProvince bounding boxes:")
    for key, cfg in PROVINCES.items():
        log.info(
            "  %-15s  Lat [%.2f – %.2f]  Lon [%.2f – %.2f]",
            cfg["label"], cfg["lat_min"], cfg["lat_max"],
            cfg["lon_min"], cfg["lon_max"],
        )

    # ── BGC: dissolved oxygen + chlorophyll-a ───────────────────────────────
    log.info("\n══════════ BGC (o2, chl) ══════════")
    bgc_wide = fetch_product_all_provinces(
        BGC_DATASET, BGC_VARIABLES, "bgc", years, base_df
    )

    # ── PHY: temperature + salinity ─────────────────────────────────────────
    log.info("\n══════════ PHY (thetao, so) ══════════")
    phy_wide = fetch_product_all_provinces(
        PHY_DATASET, PHY_VARIABLES, "phy", years, base_df
    )

    # ── Merge into base ──────────────────────────────────────────────────────
    log.info("\n══════════ Merging ══════════")
    result = base_df.merge(bgc_wide, on="date", how="left")
    result = result.merge(phy_wide, on="date", how="left")

    # ── QA report ────────────────────────────────────────────────────────────
    new_cols = [
        f"{prefix}_{prov}"
        for prov in PROVINCES
        for prefix in VAR_PREFIX.values()
    ]
    log.info("\nQA — null counts for new CMEMS columns:")
    for col in new_cols:
        if col in result.columns:
            n   = result[col].isna().sum()
            pct = 100 * n / len(result)
            log.info("  %-45s  %5d nulls  (%.1f%%)", col, n, pct)
        else:
            log.warning("  Column missing from result: %s", col)

    log.info("\nDescriptive stats:")
    present = [c for c in new_cols if c in result.columns]
    log.info("\n%s", result[present].describe().round(4).to_string())

    # ── Save ─────────────────────────────────────────────────────────────────
    result.to_csv(OUTPUT_CSV, index=False)
    log.info(
        "\n✅  Saved: %s  (%d rows × %d cols)",
        OUTPUT_CSV, len(result), len(result.columns),
    )
    log.info("All columns:\n  %s", "\n  ".join(result.columns.tolist()))


if __name__ == "__main__":
    main()
