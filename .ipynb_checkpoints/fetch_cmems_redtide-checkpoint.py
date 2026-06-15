"""
fetch_cmems_redtide.py
======================
Fetches daily oceanographic variables from Copernicus Marine Service (CMEMS)
and merges them into the Eastern Visayas red-tide ML dataset.

Data products used
------------------
1. BGC (Biogeochemistry) — dissolved oxygen & chlorophyll-a
   Dataset : cmems_mod_glo_bgc_my_0.25deg_P1D-m
   Variables: o2   (dissolved oxygen, mmol m-3)
              chl  (chlorophyll-a, mg m-3)

2. PHY (Physics) — sea surface temperature & salinity
   Dataset : cmems_mod_glo_phy_my_0.083deg_P1D-m
   Variables: thetao (potential temperature, °C)
              so     (practical salinity, PSU)

Spatial domain — Eastern Visayas bounding box
   Longitude: 124.0 – 126.5 °E
   Latitude :  10.5 –  12.5 °N
   Depth    :  surface layer only (0 – 1 m)

Strategy
--------
- Fetch each product year-by-year to stay within CMEMS memory limits.
- Average spatially over the bounding box → one value per day.
- Merge on `date` with the existing CSV.
- Save progress after every year so a crash is recoverable.
- Final output: easternVisayas_redtide_with_cmems.csv

Requirements
------------
    pip install copernicusmarine pandas xarray numpy tqdm
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

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("fetch_cmems_redtide.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────

# !! Set your CMEMS credentials here OR export as environment variables:
#    export COPERNICUSMARINE_SERVICE_USERNAME="your_username"
#    export COPERNICUSMARINE_SERVICE_PASSWORD="your_password"
CMEMS_USERNAME = os.environ.get("COPERNICUSMARINE_SERVICE_USERNAME", "YOUR_USERNAME")
CMEMS_PASSWORD = os.environ.get("COPERNICUSMARINE_SERVICE_PASSWORD", "YOUR_PASSWORD")

# Input CSV
INPUT_CSV = "easternVisayasRedTide.csv"

# Output CSV
OUTPUT_CSV = "easternVisayas_redtide_with_cmems.csv"

# Checkpoint directory (saves per-year progress)
CHECKPOINT_DIR = Path("cmems_checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True)

# Eastern Visayas spatial bounding box
LON_MIN, LON_MAX = 124.0, 126.5
LAT_MIN, LAT_MAX = 10.5,  12.5
DEPTH_MIN        = 0.0
DEPTH_MAX        = 1.0      # surface only

# Retry settings
MAX_RETRIES  = 3
RETRY_DELAY  = 30           # seconds between retries

# ── Dataset definitions ───────────────────────────────────────────────────────

BGC_DATASET = "cmems_mod_glo_bgc_my_0.25deg_P1D-m"
PHY_DATASET = "cmems_mod_glo_phy_my_0.083deg_P1D-m"

# NOTE: CMEMS "my" (multi-year reanalysis) products run up to ~2023.
# For 2024-onwards you would switch to the "myint" (interim) variant.
# The script handles this automatically below.

BGC_VARIABLES  = ["o2", "chl"]
PHY_VARIABLES  = ["thetao", "so"]

# Rename map: CMEMS variable name → output column name
RENAME_MAP = {
    "o2":     "dissolved_oxygen_mmol_m3",
    "chl":    "chlorophyll_a_mg_m3",
    "thetao": "sea_surface_temp_C",
    "so":     "salinity_PSU",
}

# CMEMS "my" reanalysis products cover up to this year (inclusive).
# Adjust if CMEMS extends coverage later.
MY_CUTOFF_YEAR = 2023   # multi-year reanalysis
# For years > MY_CUTOFF_YEAR the "myint" (interim) dataset is used.
BGC_DATASET_INT = "cmems_mod_glo_bgc_myint_0.25deg_P1D-m"
PHY_DATASET_INT = "cmems_mod_glo_phy_myint_0.083deg_P1D-m"


# ── Helper functions ──────────────────────────────────────────────────────────

def get_dataset_id(base_dataset: str, year: int) -> str:
    """Return the correct dataset id depending on the year."""
    if year > MY_CUTOFF_YEAR:
        return base_dataset.replace("_my_", "_myint_")
    return base_dataset


def fetch_year(dataset_id: str, variables: list[str], year: int) -> xr.Dataset:
    """
    Download one calendar year of daily data for the study area.
    Returns an xr.Dataset with spatial mean already applied (time only).
    Retries up to MAX_RETRIES on failure.
    """
    start = f"{year}-01-01"
    end   = f"{year}-12-31"
    log.info("  Fetching %s  [%s … %s]", dataset_id, start, end)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            ds = copernicusmarine.open_dataset(
                dataset_id        = dataset_id,
                username          = CMEMS_USERNAME,
                password          = CMEMS_PASSWORD,
                variables         = variables,
                minimum_longitude = LON_MIN,
                maximum_longitude = LON_MAX,
                minimum_latitude  = LAT_MIN,
                maximum_latitude  = LAT_MAX,
                minimum_depth     = DEPTH_MIN,
                maximum_depth     = DEPTH_MAX,
                start_datetime    = start,
                end_datetime      = end,
            )

            # --- Depth: take shallowest available layer -------------------
            if "depth" in ds.dims:
                ds = ds.isel(depth=0)
            elif "elevation" in ds.dims:
                ds = ds.isel(elevation=0)

            # --- Spatial mean over the bounding box -----------------------
            spatial_dims = [d for d in ds.dims if d in ("latitude", "longitude", "lat", "lon")]
            if spatial_dims:
                ds_mean = ds.mean(dim=spatial_dims, skipna=True)
            else:
                ds_mean = ds

            return ds_mean

        except Exception as exc:
            log.warning("  Attempt %d/%d failed: %s", attempt, MAX_RETRIES, exc)
            if attempt < MAX_RETRIES:
                log.info("  Waiting %d s before retry …", RETRY_DELAY)
                time.sleep(RETRY_DELAY)
            else:
                log.error("  All retries exhausted for %s year %d.", dataset_id, year)
                raise


def ds_to_df(ds: xr.Dataset, variables: list[str], year: int) -> pd.DataFrame:
    """Convert an xr.Dataset (time-only, after spatial mean) to a daily DataFrame."""
    time_dim = "time" if "time" in ds.dims else ds.dims[0]
    dates = pd.to_datetime(ds[time_dim].values).normalize()

    rows = {"date": dates}
    for var in variables:
        if var in ds:
            rows[var] = ds[var].values.astype(float)
        else:
            log.warning("  Variable '%s' not found in dataset for year %d.", var, year)
            rows[var] = np.nan

    df = pd.DataFrame(rows)
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    return df


def checkpoint_path(product: str, year: int) -> Path:
    return CHECKPOINT_DIR / f"{product}_{year}.parquet"


def load_checkpoint(product: str, year: int) -> pd.DataFrame | None:
    path = checkpoint_path(product, year)
    if path.exists():
        log.info("  Checkpoint found: %s — skipping download.", path.name)
        return pd.read_parquet(path)
    return None


def save_checkpoint(df: pd.DataFrame, product: str, year: int) -> None:
    path = checkpoint_path(product, year)
    df.to_parquet(path, index=False)
    log.info("  Saved checkpoint: %s", path.name)


# ── Main pipeline ─────────────────────────────────────────────────────────────

def main():
    # 1. Load the base CSV
    log.info("Loading base dataset: %s", INPUT_CSV)
    base_df = pd.read_csv(INPUT_CSV)
    base_df["date"] = pd.to_datetime(base_df["date"]).dt.strftime("%Y-%m-%d")

    years = sorted(base_df["year"].unique().tolist())
    log.info("Date range: %s to %s  (%d years, %d days)",
             base_df["date"].iloc[0], base_df["date"].iloc[-1],
             len(years), len(base_df))

    # 2. Fetch BGC (o2, chl) ─────────────────────────────────────────────────
    log.info("\n=== Fetching BGC variables (o2, chl) ===")
    bgc_frames = []
    for year in tqdm(years, desc="BGC years"):
        cached = load_checkpoint("bgc", year)
        if cached is not None:
            bgc_frames.append(cached)
            continue
        try:
            ds_id = get_dataset_id(BGC_DATASET, year)
            ds    = fetch_year(ds_id, BGC_VARIABLES, year)
            df    = ds_to_df(ds, BGC_VARIABLES, year)
            save_checkpoint(df, "bgc", year)
            bgc_frames.append(df)
        except Exception as exc:
            log.error("BGC year %d failed permanently: %s", year, exc)
            # Fill that year with NaN so the merge still works
            year_dates = base_df.loc[base_df["year"] == year, "date"].tolist()
            fallback   = pd.DataFrame({"date": year_dates, "o2": np.nan, "chl": np.nan})
            bgc_frames.append(fallback)

    bgc_df = pd.concat(bgc_frames, ignore_index=True)
    bgc_df = bgc_df.rename(columns={"o2": RENAME_MAP["o2"], "chl": RENAME_MAP["chl"]})

    # 3. Fetch PHY (thetao, so) ──────────────────────────────────────────────
    log.info("\n=== Fetching PHY variables (thetao, so) ===")
    phy_frames = []
    for year in tqdm(years, desc="PHY years"):
        cached = load_checkpoint("phy", year)
        if cached is not None:
            phy_frames.append(cached)
            continue
        try:
            ds_id = get_dataset_id(PHY_DATASET, year)
            ds    = fetch_year(ds_id, PHY_VARIABLES, year)
            df    = ds_to_df(ds, PHY_VARIABLES, year)
            save_checkpoint(df, "phy", year)
            phy_frames.append(df)
        except Exception as exc:
            log.error("PHY year %d failed permanently: %s", year, exc)
            year_dates = base_df.loc[base_df["year"] == year, "date"].tolist()
            fallback   = pd.DataFrame({"date": year_dates, "thetao": np.nan, "so": np.nan})
            phy_frames.append(fallback)

    phy_df = pd.concat(phy_frames, ignore_index=True)
    phy_df = phy_df.rename(columns={"thetao": RENAME_MAP["thetao"], "so": RENAME_MAP["so"]})

    # 4. Merge everything ────────────────────────────────────────────────────
    log.info("\n=== Merging datasets ===")
    merged = base_df.merge(bgc_df, on="date", how="left")
    merged = merged.merge(phy_df, on="date", how="left")

    # 5. QA report ───────────────────────────────────────────────────────────
    new_cols = list(RENAME_MAP.values())
    log.info("\nQA — null counts for new CMEMS columns:")
    for col in new_cols:
        n_null = merged[col].isna().sum()
        pct    = 100 * n_null / len(merged)
        log.info("  %-35s  %5d nulls  (%.1f%%)", col, n_null, pct)

    log.info("\nDescriptive stats for new columns:")
    log.info("\n%s", merged[new_cols].describe().to_string())

    # 6. Save output ─────────────────────────────────────────────────────────
    merged.to_csv(OUTPUT_CSV, index=False)
    log.info("\n✅  Saved: %s  (%d rows × %d cols)", OUTPUT_CSV, len(merged), len(merged.columns))
    log.info("Columns: %s", merged.columns.tolist())


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()
