"""Instance plugins for the reproducible Malawi Chap example.

Earth Engine is an optional example dependency, not a dependency of GeoService core.
Only an environment variable points to credentials; no key enters a template.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pandas as pd
import rioxarray
import xarray as xr

from open_geo_service.streaming import BaseDatasetPlugin, normalize_period

COLLECTION = "ECMWF/ERA5_LAND/MONTHLY_AGGR"
BANDS = ["temperature_2m", "dewpoint_temperature_2m", "total_precipitation_sum"]
WORLDPOP_LEGACY = (
    "https://data.worldpop.org/GIS/Population/Global_2000_2020_1km_UNadj/"
    "{year}/MWI/mwi_ppp_{year}_1km_Aggregated_UNadj.tif"
)
WORLDPOP_GLOBAL2 = (
    "https://data.worldpop.org/GIS/Population/Global_2015_2030/R2025A/"
    "{year}/MWI/v1/1km_ua/constrained/mwi_pop_{year}_CN_1km_R2025A_UA_v1.tif"
)


def download(url: str, target: Path) -> Path:
    """Download atomically, reusing a previously completed source file."""
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        # Do not log the URL: Earth Engine download URLs contain temporary tokens.
        with httpx.Client(timeout=240, follow_redirects=True) as client:
            response = client.get(url)
            if response.status_code != 200:
                raise RuntimeError(f"Download failed with HTTP {response.status_code}: {target.name}")
        temporary = target.with_suffix(target.suffix + ".part")
        temporary.write_bytes(response.content)
        temporary.replace(target)
    return target


class EarthEngineMonthlyPlugin(BaseDatasetPlugin):
    """Read native-grid ERA5-Land monthly rasters in small, cached annual batches."""

    commit_batch_size = 12

    def __init__(self, variable: str, cache_dir: str) -> None:
        """Select one Chap variable; share the three-band download between variables."""
        if variable not in {"rainfall", "mean_temperature", "mean_relative_humidity"}:
            raise ValueError(f"Unsupported climate variable: {variable}")
        self.variable = variable
        self.cache_dir = Path(cache_dir)

    async def periods(self, start: str, end: str) -> list[str]:
        """Enumerate the fixed benchmark's inclusive monthly coverage."""
        return [str(p) for p in pd.period_range(max(start[:7], "2005-01"), min(end[:7], "2023-12"), freq="M")]

    def annual_path(self, year: int, bbox: list[float]) -> Path:
        """Key source downloads by collection, native grid, year and requested extent."""
        key = hashlib.sha256(json.dumps([COLLECTION, bbox, BANDS]).encode()).hexdigest()[:16]
        return self.cache_dir / f"era5land-{year}-{key}.tif"

    def download_year(self, year: int, bbox: list[float]) -> Path:
        """Export a year in native projection without resampling or district masking."""
        target = self.annual_path(year, bbox)
        if target.exists():
            return target
        import ee

        credential_path = Path(os.environ["GOOGLE_APPLICATION_CREDENTIALS"]).expanduser()
        credentials = json.loads(credential_path.read_text())
        ee.Initialize(
            ee.ServiceAccountCredentials(credentials["client_email"], str(credential_path)),
            project=credentials["project_id"],
        )
        images = (
            ee.ImageCollection(COLLECTION)
            .filterDate(f"{year}-01-01", f"{year + 1}-01-01")
            .sort("system:time_start")
            .select(BANDS)
        )
        image = images.toBands()
        expected = [f"{year}{month:02d}_{band}" for month in range(1, 13) for band in BANDS]
        if image.bandNames().getInfo() != expected:
            raise ValueError(f"Incomplete or unexpected ERA5-Land monthly bands for {year}")
        projection = ee.Image(images.first()).select(BANDS[0]).projection().getInfo()
        # An extra cell preserves pixels intersecting the edge of the GeoService extent.
        west, south, east, north = bbox
        region = ee.Geometry.Rectangle([west - 0.1, south - 0.1, east + 0.1, north + 0.1], geodesic=False)
        url = image.getDownloadURL(
            {"crs": projection["crs"], "crs_transform": projection["transform"], "region": region, "format": "GEO_TIFF"}
        )
        print(f"Downloading ERA5-Land {year}", flush=True)
        return download(url, target)

    def fetch_period(self, period_id: str, bbox: list[float], **params: Any) -> xr.Dataset:
        """Convert units and derive humidity before GeoService performs spatial aggregation."""
        year, month = (int(part) for part in period_id[:7].split("-"))
        path = self.download_year(year, bbox)
        with rioxarray.open_rasterio(path, masked=True) as raster:
            if raster.sizes["band"] != 36:
                raise ValueError(f"Expected 36 ERA5-Land bands in {path.name}")
            offset = 3 * (month - 1)
            temperature = raster.isel(band=offset, drop=True).load() - 273.15
            if self.variable == "mean_temperature":
                values = temperature
                units = "degC"
            elif self.variable == "rainfall":
                # Earth Engine MONTHLY_AGGR is already a monthly sum, unlike CDS monthly means.
                values = raster.isel(band=offset + 2, drop=True).load() * 1000
                units = "mm"
            else:
                dewpoint = raster.isel(band=offset + 1, drop=True).load() - 273.15
                values = 100 * np.exp(
                    17.67 * dewpoint / (dewpoint + 243.5) - 17.67 * temperature / (temperature + 243.5)
                )
                units = "%"
        values.attrs = {"units": units}
        return normalize_period(values.rio.write_crs(4326), variable=self.variable, period=period_id, bbox=bbox)


class MalawiWorldPopPlugin(BaseDatasetPlugin):
    """Read the ticket's two pinned WorldPop products into separate GeoService stores."""

    def __init__(self, product: str, cache_dir: str) -> None:
        """Select legacy UN-adjusted or Global2 R2025A population counts."""
        if product not in {"legacy", "global2"}:
            raise ValueError(f"Unknown WorldPop product: {product}")
        self.product = product
        self.cache_dir = Path(cache_dir)

    async def periods(self, start: str, end: str) -> list[str]:
        """Keep different source grids in different stores."""
        first, last = (2005, 2020) if self.product == "legacy" else (2021, 2023)
        return [str(year) for year in range(max(int(start[:4]), first), min(int(end[:4]), last) + 1)]

    def fetch_period(self, period_id: str, bbox: list[float], **params: Any) -> xr.Dataset:
        """Read people per pixel, masking source nodata before the district sum."""
        template = WORLDPOP_LEGACY if self.product == "legacy" else WORLDPOP_GLOBAL2
        path = download(
            template.format(year=period_id[:4]), self.cache_dir / f"worldpop-{self.product}-{period_id[:4]}.tif"
        )
        with rioxarray.open_rasterio(path, masked=True) as raster:
            values = raster.load()
        values.attrs["units"] = "people"
        # Keep the full national raster: a boundary may extend slightly outside its valid mask.
        return normalize_period(values, variable="population", period=period_id[:4], nodata=-99999)
