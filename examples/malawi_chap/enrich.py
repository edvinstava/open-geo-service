"""Extend a verified Malawi bundle with historical environmental covariates.

Run with python -m examples.malawi_chap.enrich --bundle /path/to/output
--work-dir /path/to/a/new/work-directory. See README for source definitions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import geopandas as gpd
import httpx
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from . import build

CATALOG = "https://developers.google.com/earth-engine/datasets/catalog/"
COLLECTIONS = {
    "weather": "ECMWF/ERA5_LAND/MONTHLY_AGGR",
    "vegetation": "MODIS/061/MOD13A3",
    "landcover": "MODIS/061/MCD12Q1",
    "water": "JRC/GSW1_4/YearlyHistory",
    "terrain": "USGS/SRTMGL1_003",
}
# Output column: (upstream band, multiplier, offset, unit).
WEATHER = {
    "monthly_min_temperature": ("temperature_2m_min", 1, -273.15, "degC"),
    "monthly_max_temperature": ("temperature_2m_max", 1, -273.15, "degC"),
    "mean_dewpoint_temperature": ("dewpoint_temperature_2m", 1, -273.15, "degC"),
    "mean_soil_temperature_0_7cm": ("soil_temperature_level_1", 1, -273.15, "degC"),
    "soil_moisture_0_7cm": ("volumetric_soil_water_layer_1", 1, 0, "m3/m3"),
    "soil_moisture_7_28cm": ("volumetric_soil_water_layer_2", 1, 0, "m3/m3"),
    "soil_moisture_28_100cm": ("volumetric_soil_water_layer_3", 1, 0, "m3/m3"),
    "surface_runoff_mm": ("surface_runoff_sum", 1000, 0, "mm/month"),
    "evaporation_mm": ("total_evaporation_sum", -1000, 0, "mm/month"),
    "solar_radiation_mj_m2": ("surface_solar_radiation_downwards_sum", 1e-6, 0, "MJ/m2/month"),
    "eastward_wind_10m": ("u_component_of_wind_10m", 1, 0, "m/s"),
    "northward_wind_10m": ("v_component_of_wind_10m", 1, 0, "m/s"),
    "leaf_area_index_low_vegetation": ("leaf_area_index_low_vegetation", 1, 0, "m2/m2"),
    "leaf_area_index_high_vegetation": ("leaf_area_index_high_vegetation", 1, 0, "m2/m2"),
}
LAND_CLASSES = {
    "forest_fraction": [1, 2, 3, 4, 5],
    "shrubland_fraction": [6, 7],
    "savanna_fraction": [8, 9],
    "grassland_fraction": [10],
    "wetland_fraction": [11],
    "cropland_or_mosaic_fraction": [12, 14],
    "urban_fraction": [13],
    "bare_land_fraction": [16],
    "landcover_water_fraction": [17],
}
GROUP_COLUMNS = {
    "weather": list(WEATHER),
    "vegetation": ["ndvi", "evi", "vegetation_good_quality_fraction"],
    "landcover": list(LAND_CLASSES),
    "water": ["seasonal_water_detected_fraction", "permanent_water_detected_fraction", "water_classified_fraction"],
    "terrain": ["mean_elevation_m", "mean_slope_degrees"],
}
NINO_URL = "https://www.cpc.ncep.noaa.gov/data/indices/ersst5.nino.mth.91-20.ascii"
WATER_UPDATES = [
    "projects/global-surface-water/assets/GSW1_5/YearlyHistory_2016_2021",
    "projects/global-surface-water/assets/GSW1_5/YearlyHistory_2022_2024",
]


def earth_engine(credentials: Path) -> Any:
    """Authenticate using the existing external key without persisting its contents."""
    import ee

    account = json.loads(credentials.read_text())
    ee.Initialize(
        ee.ServiceAccountCredentials(account["client_email"], str(credentials)), project=account["project_id"]
    )
    ee.data.setDeadline(120000)
    return ee


def source_collection(ee: Any, group: str) -> Any:
    """Use JRC's corrected 2016-2021 history and its 2022-2024 extension."""
    collection = ee.ImageCollection(COLLECTIONS[group])
    if group == "water":
        collection = collection.filterDate("1984-01-01", "2016-01-01")
        for asset in WATER_UPDATES:
            collection = collection.merge(ee.ImageCollection(asset))
    return collection


def image_for(ee: Any, group: str, period: str) -> tuple[Any, Any]:
    """Build numeric bands before reducing; categorical codes are never averaged."""
    if group == "terrain":
        source = ee.Image(COLLECTIONS[group]).select("elevation")
        image = source.rename("mean_elevation_m").addBands(ee.Terrain.slope(source).rename("mean_slope_degrees"))
    else:
        start = pd.Timestamp(period + ("-01" if len(period) == 7 else "-01-01"))
        end = start + (pd.offsets.MonthBegin(1) if len(period) == 7 else pd.offsets.YearBegin(1))
        source = source_collection(ee, group).filterDate(str(start.date()), str(end.date())).first()
        if group == "weather":
            image = ee.Image.cat(
                [
                    source.select(band).multiply(factor).add(offset).rename(name)
                    for name, (band, factor, offset, _) in WEATHER.items()
                ]
            )
        elif group == "vegetation":
            good = source.select("SummaryQA").eq(0)
            indices = source.select(["NDVI", "EVI"]).multiply(0.0001)
            indices = indices.updateMask(good).updateMask(indices.gte(-0.2).And(indices.lte(1)))
            image = indices.rename(["ndvi", "evi"]).addBands(good.rename("vegetation_good_quality_fraction"))
        elif group == "landcover":
            classes = source.select("LC_Type1")
            image = ee.Image.cat(
                [classes.remap(codes, [1] * len(codes), 0).rename(name) for name, codes in LAND_CLASSES.items()]
            ).updateMask(classes.gte(1).And(classes.lte(17)))
        else:
            classes = source.select("waterClass")
            image = (
                classes.eq(2)
                .rename("seasonal_water_detected_fraction")
                .addBands(classes.eq(3).rename("permanent_water_detected_fraction"))
                .unmask(0)
            )
            # Count detections over the whole district, retaining coverage to distinguish unclassified from dry.
            image = image.addBands(classes.neq(0).unmask(0).rename("water_classified_fraction"))
    return image, source.select(0).projection()


def fetch_group(ee: Any, group: str, year: int | None, collection: dict[str, Any], cache: Path) -> pd.DataFrame:
    """Request one year of district statistics at the source grid, caching only public results."""
    monthly = group in {"weather", "vegetation"}
    periods = [f"{year}-{m:02d}" for m in range(1, 13)] if monthly else [str(year) if year else "static"]
    bands = []
    projection = None
    for period in periods:
        image, projection = image_for(ee, group, period)
        prefix = period.replace("-", "")
        bands.append(image.rename([prefix + "__" + name for name in GROUP_COLUMNS[group]]))
    features = ee.FeatureCollection(
        [
            ee.Feature(ee.Geometry(f["geometry"], proj="EPSG:4326", geodesic=False), {"location": f["id"]})
            for f in collection["features"]
        ]
    )
    reduced = (
        ee.Image.cat(bands)
        .reduceRegions(
            collection=features,
            reducer=ee.Reducer.mean(),
            crs=projection,
            tileScale=4,
            maxPixelsPerRegion=1_000_000_000,
        )
        .map(lambda f: ee.Feature(None, f.toDictionary()))
    )
    query = ee.Dictionary({"features": reduced.toList(100)})
    if year:
        images = source_collection(ee, group).filterDate(f"{year}-01-01", f"{year + 1}-01-01")
        query = query.set("dates", images.aggregate_array("system:time_start"))
    digest = hashlib.sha256(query.serialize().encode()).hexdigest()[:16]
    target = cache / f"{group}-{year or 'static'}-{digest}.json"
    if not target.exists():
        print(f"Earth Engine: {group} {year or 'static'}", flush=True)
        result = query.getInfo()
        build.write_json(target.with_suffix(".part"), result)
        target.with_suffix(".part").replace(target)
    result = json.loads(target.read_text())
    if year:
        dates = pd.to_datetime(result["dates"], unit="ms", utc=True)
        actual = dates.strftime("%Y-%m" if monthly else "%Y").tolist()
        if sorted(actual) != periods:
            raise ValueError(f"Unexpected {group} coverage in {year}: {actual}")
    rows = []
    for feature in result["features"]:
        values = feature["properties"]
        for period in periods:
            row = {"location": values["location"], "time_period": period}
            row.update({name: values.get(period.replace("-", "") + "__" + name) for name in GROUP_COLUMNS[group]})
            rows.append(row)
    frame = pd.DataFrame(rows)
    expected = {(f["id"], period) for f in collection["features"] for period in periods}
    if frame.duplicated(build.KEY).any() or set(map(tuple, frame[build.KEY].to_numpy())) != expected:
        raise ValueError(f"Incomplete district keys from {group}")
    if group == "water":
        frame.loc[frame.water_classified_fraction.eq(0), GROUP_COLUMNS["water"][:2]] = np.nan
    return frame


def nino34(cache: Path) -> pd.DataFrame:
    """Read NOAA's explicitly named ERSSTv5 index with its 1991-2020 reference period."""
    target = cache / "ersst5.nino.mth.91-20.ascii"
    if not target.exists():
        response = httpx.get(NINO_URL, timeout=60)
        response.raise_for_status()
        target.write_bytes(response.content)
    frame = pd.read_csv(target, sep=r"\s+")
    if frame.shape[1] != 10 or frame.columns[0] != "YR" or frame.columns[-2] != "NINO3.4":
        raise ValueError("Unexpected NOAA index columns")
    periods = pd.to_datetime({"year": frame.iloc[:, 0], "month": frame.iloc[:, 1], "day": 1})
    values = pd.to_numeric(frame.iloc[:, -1]).where(lambda x: x > -90)
    return pd.DataFrame({"time_period": periods.dt.strftime("%Y-%m"), "nino34_anomaly": values})


def field_definitions() -> list[dict[str, Any]]:
    """Describe the additional columns alongside the base bundle's unchanged definitions."""
    fields = []
    for group, columns in GROUP_COLUMNS.items():
        source = {
            "provider": "Copernicus / ECMWF"
            if group == "weather"
            else "EC JRC / Google"
            if group == "water"
            else "NASA / USGS",
            "collection": [COLLECTIONS[group], *WATER_UPDATES] if group == "water" else COLLECTIONS[group],
            "url": "https://global-surface-water.appspot.com/download"
            if group == "water"
            else CATALOG + COLLECTIONS[group].replace("/", "_"),
            "license": "Copernicus terms; attribution required"
            if group in {"weather", "water"}
            else "NASA/USGS open data; attribution required",
            "aggregation": "Earth Engine weighted mean on the original source grid; no bestEffort or coarsening",
        }
        for name in columns:
            description = "District mean on the source grid."
            unit = "1"
            if group == "weather":
                band, factor, offset, unit = WEATHER[name]
                description = f"District mean of {band} * {factor} + ({offset})."
                if name.startswith("monthly_"):
                    description += " Mean of pixel monthly hourly extremes, not mean daily minima/maxima."
                if name == "evaporation_mm":
                    description += " Positive is evaporation; negative is condensation."
            elif group == "vegetation":
                description = "MOD13A3.061 monthly index * 0.0001; SummaryQA=0; valid scaled range -0.2 to 1."
                if name == "vegetation_good_quality_fraction":
                    description = (
                        "Fraction of source pixels with SummaryQA=0; diagnostic, not an environmental measurement."
                    )
            elif group == "landcover":
                description = f"Fraction of classified pixels in IGBP classes {LAND_CLASSES[name]}; repeated monthly."
            elif group == "water":
                description = "Fraction of the entire district grid detected as waterClass=" + (
                    "2" if name.startswith("seasonal") else "3"
                )
                description += (
                    "; unclassified pixels contribute no detections, not evidence of dry land. "
                    "Read with water_classified_fraction; blank if no pixels are classified."
                )
                if name == "water_classified_fraction":
                    description = (
                        "Fraction of the district grid with a nonzero, unmasked waterClass; all other pixels count "
                        "as unclassified. Coverage diagnostic, not water extent."
                    )
                description += (
                    " Annual value repeated monthly. Original history through 2015; corrected v1.5 for 2016-2021; "
                    "v1.5 Landsat Collection 2 for 2022-2023. Changes across versions may include processing artefacts."
                )
            else:
                unit = "m" if name == "mean_elevation_m" else "degree"
                description = "Static district mean from SRTM (February 2000), repeated for all months."
            fields.append({"name": name, "type": "number", "unit": unit, "description": description, "source": source})
    for name, unit, description, source in [
        (
            "rainfall_chirps3",
            "mm/month",
            "GeoService district mean CHIRPS3 rate multiplied by calendar days.",
            {
                "provider": "UCSB CHC",
                "url": "https://www.chc.ucsb.edu/data/chirps3",
                "license": "CC-BY-4.0",
                "data_source_id": "chirps3_precipitation_monthly",
            },
        ),
        (
            "nino34_anomaly",
            "degC",
            "Monthly ERSSTv5 Nino 3.4 anomaly, 1991-2020 reference; shared by all districts.",
            {"provider": "NOAA CPC", "url": NINO_URL, "license": "US government public data"},
        ),
        (
            "district_area_km2",
            "km2",
            "Area of the supplied district polygon in equal-area EPSG:6933, including inland water.",
            {"provider": "Derived from the base bundle's geoBoundaries polygons"},
        ),
        (
            "population_density_km2",
            "people/km2",
            "Base bundle population divided by district_area_km2.",
            {"provider": "Derived from the base bundle's WorldPop population and polygons"},
        ),
    ]:
        fields.append({"name": name, "type": "number", "unit": unit, "description": description, "source": source})
    return fields


def write_bundle(bundle: Path, work: Path, base: pd.DataFrame, additions: list[pd.DataFrame]) -> None:
    """Join without dropping cases or filling unknown values, then record coverage and provenance."""
    result = base.copy()
    for extra in additions:
        keys = build.KEY if "location" in extra else ["time_period"]
        if extra.duplicated(keys).any():
            raise ValueError("Duplicate covariate keys")
        if set(extra.columns) & (set(result.columns) - set(keys)):
            raise ValueError("Refusing to overwrite an existing column")
        result = result.merge(extra, on=keys, how="left", validate="one_to_one" if len(keys) == 2 else "many_to_one")
    pd.testing.assert_frame_equal(result[base.columns], base, check_dtype=False)
    definitions = field_definitions()
    names = [field["name"] for field in definitions]
    if set(result.columns) != set(base.columns) | set(names):
        raise ValueError("Missing or undocumented covariate columns")
    numbers = result[names].astype(float)
    if np.isinf(numbers.to_numpy()).any():
        raise ValueError("Infinite covariate values")
    fractions = [n for n in names if n.endswith("_fraction") or n.startswith("soil_moisture")]
    if ((numbers[fractions] < 0) | (numbers[fractions] > 1)).any().any():
        raise ValueError("Fraction outside 0 to 1")
    output = work / "output"
    output.mkdir(exist_ok=True)
    result.to_csv(output / f"{build.STEM}.csv", index=False)
    for name in [f"{build.STEM}.geojson", "case_quality_flags.csv"]:
        shutil.copyfile(bundle / name, output / name)
    schema = json.loads((bundle / f"{build.STEM}_schema.json").read_text())
    schema["dataset"].update(
        name="Chap Malawi malaria with environmental covariates", created_utc=datetime.now(UTC).isoformat()
    )
    schema["fields"].extend(definitions)
    schema["limitations"].extend(
        [
            "Base columns and their documented humidity method are unchanged. Enrichment does not replace humidity.",
            "Annual land/water cover is repeated monthly, not observed monthly. SRTM terrain is static from 2000.",
            "JRC seasonal/permanent fractions count detections across the entire district grid. "
            "Read water_classified_fraction alongside them; missing or masked pixels are not evidence of dry land.",
            "Missing covariates are blank and flagged; no zero-fill, interpolation or forward-fill is applied.",
            "This is a retrospective benchmark. Same-month weather, annual maps and revised sources may be "
            "unavailable at forecast time. Validate reporting lags before forecasting.",
            "More covariates do not establish predictive value. Select variables and transforms within training folds.",
        ]
    )
    build.write_json(output / f"{build.STEM}_schema.json", schema)
    flags = []
    coverage = {}
    spatial_coverage = {
        "ndvi": "vegetation_good_quality_fraction",
        "evi": "vegetation_good_quality_fraction",
        "seasonal_water_detected_fraction": "water_classified_fraction",
        "permanent_water_detected_fraction": "water_classified_fraction",
    }
    for name in names:
        valid = result[name].notna()
        partial = (
            valid & result[spatial_coverage[name]].lt(1)
            if name in spatial_coverage
            else pd.Series(False, index=result.index)
        )
        coverage[name] = {
            "present": int(valid.sum()),
            "missing": int((~valid).sum()),
            "partial_spatial_coverage": int(partial.sum()),
            "min": float(numbers[name].min()) if valid.any() else None,
            "max": float(numbers[name].max()) if valid.any() else None,
        }
        for row in result.loc[~valid, build.KEY].to_dict("records"):
            flags.append({**row, "variable": name, "reason": "no_valid_source_pixels_or_value"})
        for row in result.loc[partial, build.KEY].to_dict("records"):
            flags.append({**row, "variable": name, "reason": "partial_spatial_coverage"})
    pd.DataFrame(flags, columns=[*build.KEY, "variable", "reason"]).to_csv(
        output / "covariate_quality_flags.csv", index=False
    )
    build.write_json(
        output / "validation_report.json",
        {
            "rows": len(result),
            "districts": result.location.nunique(),
            "months": result.time_period.nunique(),
            "duplicate_keys": int(result.duplicated(build.KEY).sum()),
            "base_values_unchanged": True,
            "covariates": coverage,
        },
    )
    build.write_json(
        output / "provenance.json",
        {
            "generated_utc": datetime.now(UTC).isoformat(),
            "base_provenance": json.loads((bundle / "provenance.json").read_text()),
            "base_provenance_sha256": build.sha256(bundle / "provenance.json"),
            "script_sha256": build.sha256(Path(__file__)),
            "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, cwd=build.REPO).strip(),
            "source_sha256": {
                str(p.relative_to(work)): build.sha256(p)
                for folder in ["cache", "aggregates"]
                for p in sorted((work / folder).glob("*"))
                if p.is_file()
            },
            "output_sha256": {p.name: build.sha256(p) for p in sorted(output.iterdir()) if p.name != "provenance.json"},
        },
    )
    print(f"Expanded bundle: {output}; {len(result)} rows, {len(result.columns)} columns", flush=True)


def main() -> None:
    """Enrich a verified existing bundle using a separate local instance and public sources."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bundle", type=Path, required=True, help="Existing complete output directory with provenance.json"
    )
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--credentials", type=Path)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--port", type=int, default=9003)
    args = parser.parse_args()
    if args.env_file:
        load_dotenv(args.env_file, override=False)
    bundle, work = args.bundle.resolve(), args.work_dir.resolve()
    if bundle == work or bundle == work / "output":
        parser.error("Use a separate work directory to preserve the original bundle")
    manifest = json.loads((bundle / "provenance.json").read_text())
    for name in [f"{build.STEM}.csv", f"{build.STEM}.geojson", f"{build.STEM}_schema.json", "case_quality_flags.csv"]:
        if build.sha256(bundle / name) != manifest["output_sha256"].get(name):
            raise ValueError(f"Base bundle checksum mismatch: {name}")
    base = pd.read_csv(bundle / f"{build.STEM}.csv", dtype={"location": str, "time_period": str})
    collection = json.loads((bundle / f"{build.STEM}.geojson").read_text())
    expected = pd.MultiIndex.from_product(
        [[f["id"] for f in collection["features"]], pd.period_range("2005-01", "2023-12", freq="M").astype(str)]
    )
    if (
        len(base) != 6156
        or base.duplicated(build.KEY).any()
        or set(map(tuple, base[build.KEY].to_numpy())) != set(expected)
    ):
        raise ValueError("Expected the complete 27-district, 2005-2023 base bundle")
    credentials = (
        (
            args.credentials
            or Path(
                os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "~/.config/open-climate-service/dhis-google-auth.json")
            )
        )
        .expanduser()
        .resolve()
    )
    work.mkdir(parents=True, exist_ok=True)
    cache, aggregates = work / "cache", work / "aggregates"
    cache.mkdir(exist_ok=True)
    aggregates.mkdir(exist_ok=True)
    config = build.configure(work, collection)
    with build.local_ocs(config, args.port, credentials) as client:
        job = build.ingest(client, "chirps3_precipitation_monthly", "2005-01", "2023-12")
        build.write_json(aggregates / "chirps3-job.json", {"job_id": job})
        rain = build.aggregate(
            client,
            "chirps3_precipitation_monthly",
            "2005-01-01",
            "2023-12-31",
            collection,
            False,
            aggregates / "chirps3.csv",
        )
    rain["rainfall_chirps3"] = rain.pop("precip") * pd.to_datetime(rain.time_period).dt.days_in_month
    additions = [rain]
    ee = earth_engine(credentials)
    for group in GROUP_COLUMNS:
        years = [None] if group == "terrain" else range(2005, 2024)
        frames = [fetch_group(ee, group, year, collection, cache) for year in years]
        frame = pd.concat(frames, ignore_index=True)
        if group in {"landcover", "water", "terrain"}:
            panel = base[build.KEY].copy()
            panel["year"] = "static" if group == "terrain" else panel.time_period.str[:4]
            frame = panel.merge(
                frame.rename(columns={"time_period": "year"}),
                on=["location", "year"],
                how="left",
                validate="many_to_one",
            ).drop(columns="year")
        additions.append(frame)
    additions.append(nino34(cache))
    polygons = gpd.GeoDataFrame.from_features(collection["features"], crs="EPSG:4326")
    area = dict(zip([f["id"] for f in collection["features"]], polygons.to_crs(6933).area / 1e6, strict=True))
    density = base[build.KEY].copy()
    density["district_area_km2"] = base.location.map(area)
    density["population_density_km2"] = base.population / density.district_area_km2
    additions.append(density)
    write_bundle(bundle, work, base, additions)


if __name__ == "__main__":
    main()
