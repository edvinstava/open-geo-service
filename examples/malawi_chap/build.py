"""Build the Malawi malaria benchmark through an isolated local GeoService instance.

Run from the repository root:
uv run --all-extras --with earthengine-api==1.7.49 python examples/malawi_chap/build.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from collections.abc import Generator
from contextlib import contextmanager
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv

if __package__:
    from .plugins.vectors import case_series
else:  # Running python examples/malawi_chap/build.py directly.
    from plugins.vectors import case_series

EXAMPLE = Path(__file__).resolve().parent
REPO = EXAMPLE.parents[1]
STEM = "chap_MWI_admin2_monthly"
BBOX = [32.67170594657222, -17.12627881421588, 35.91841715712616, -9.363688805793686]
CASES_URL = case_series.CASES_URL
BOUNDARIES_URL = case_series.BOUNDARIES_URL
CASES_SHA256 = case_series.CASES_SHA256
BOUNDARIES_SHA256 = case_series.BOUNDARIES_SHA256
AUTHORS_REVISION = "75774e57362ad4382f806bd1cdca53c6c63984e8"
CLIMATE_URL = "https://developers.google.com/earth-engine/datasets/catalog/ECMWF_ERA5_LAND_MONTHLY_AGGR"
CLIMATE_VARIABLES = {"rainfall": "mm", "mean_temperature": "degC", "mean_relative_humidity": "%"}
HUMIDITY_DATASET = "era5land_relative_humidity_daily"
HUMIDITY_URL = "https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land?tab=analysis_ready_data"
KEY = ["location", "time_period"]
COLUMNS = ["time_period", "location", "disease_cases", "population", "location_name", *CLIMATE_VARIABLES]


def sha256(path: Path) -> str:
    """Hash a source or output without including credential material."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    """Write portable JSON, rejecting non-finite numbers."""
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def checked_source(url: str, path: Path, expected_hash: str) -> None:
    """Download pinned research sources and reject silent upstream replacements."""
    if not path.exists():
        with httpx.Client(timeout=180, follow_redirects=True) as client:
            response = client.get(url)
            response.raise_for_status()
        temporary = path.with_suffix(path.suffix + ".part")
        temporary.write_bytes(response.content)
        if sha256(temporary) != expected_hash:
            raise ValueError(f"Source checksum changed: {path.name}; review before accepting new data")
        temporary.replace(path)
    if sha256(path) != expected_hash:
        raise ValueError(f"Cached source checksum mismatch: {path.name}")


def prepare_health(work: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Join cases by the authors' district identifiers, never by point coordinates."""
    cache = work / "downloads"
    cache.mkdir(parents=True, exist_ok=True)
    checked_source(CASES_URL, cache / "cases.csv", CASES_SHA256)
    checked_source(BOUNDARIES_URL, cache / "boundaries.geojson", BOUNDARIES_SHA256)
    cases = pd.read_csv(cache / "cases.csv")
    crosswalk = pd.read_csv(EXAMPLE / "district_crosswalk.csv", dtype={"shapeID": str})
    return case_series.district_panel(cases, crosswalk, json.loads((cache / "boundaries.geojson").read_text()))


def configure(work: Path, collection: dict[str, Any]) -> Path:
    """Create a separate Malawi instance without changing the user's running GeoService."""
    config = work / "geoservice.yaml"
    existing_config = yaml.safe_load(config.read_text()) if config.exists() else {}
    if existing_config and existing_config.get("id") != "malawi-chap":
        raise ValueError("Refusing to replace an unrelated instance configuration")
    output = work / "output"
    output.mkdir(exist_ok=True)
    geojson = output / f"{STEM}.geojson"
    write_json(geojson, collection)
    plugins = work / "plugins"
    shutil.copytree(EXAMPLE / "plugins", plugins, dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__"))
    cache = str(work / "downloads")
    templates = []
    for variable, unit in CLIMATE_VARIABLES.items():
        if variable == "mean_relative_humidity":
            continue  # Humidity uses the built-in hourly-derived CDS data source.
        templates.append(
            {
                "id": f"mwi_{variable}",
                "name": f"Malawi ERA5-Land {variable.replace('_', ' ')}",
                "description": (
                    "Copernicus ERA5-Land, monthly aggregates by Google Earth Engine, "
                    "2005-2023, native 0.1-degree grid. "
                    "Rainfall is a monthly total in mm; temperature is Celsius. District values use polygon-overlap "
                    "weighted means. Requires an authorized Google service account."
                ),
                "variable": variable,
                "units": unit,
                "period_type": "monthly",
                "sync": {"kind": "temporal", "execution": "append"},
                "source": "Copernicus Climate Change Service (C3S), monthly aggregation by Google",
                "source_url": CLIMATE_URL,
                "resolution": "0.1 degree",
                "license": {
                    "name": "Licence to Use Copernicus Products",
                    "url": "https://apps.ecmwf.int/datasets/licences/copernicus/",
                },
                "extents": {
                    "spatial": {"bbox": BBOX},
                    "temporal": {"begin": "2005-01", "end": "2023-12", "resolution": "P1M"},
                },
                "ingestion": {
                    "plugin": "rasters.malawi.EarthEngineMonthlyPlugin",
                    "params": {"variable": variable, "cache_dir": cache},
                },
            }
        )
    for product, begin, end in [("legacy", "2005", "2020"), ("global2", "2021", "2023")]:
        templates.append(
            {
                "id": f"mwi_population_{product}",
                "name": f"Malawi WorldPop {product} population (1 km)",
                "description": (
                    "WorldPop, University of Southampton, CC-BY-4.0. Annual people per pixel at 1 km. "
                    + (
                        "2005-2020 Global_2000_2020 UN-adjusted product."
                        if product == "legacy"
                        else "2021-2023 Global2 R2025A UA v1 constrained product."
                    )
                    + " District populations are overlap-weighted pixel sums. "
                    "The product changes in 2021, so changes may include a methodological discontinuity."
                ),
                "variable": "population",
                "units": "people",
                "period_type": "yearly",
                "sync": {"kind": "temporal", "execution": "append"},
                "source": "WorldPop, University of Southampton",
                "source_url": "https://www.worldpop.org/",
                "license": "CC-BY-4.0",
                "resolution": "1 km",
                "extents": {"spatial": {"bbox": BBOX}, "temporal": {"begin": begin, "end": end, "resolution": "P1Y"}},
                "ingestion": {
                    "plugin": "rasters.malawi.MalawiWorldPopPlugin",
                    "params": {"product": product, "cache_dir": cache},
                },
            }
        )
    (plugins / "rasters/malawi.yaml").write_text(yaml.safe_dump(templates, sort_keys=False))
    (plugins / "vectors/malawi.yaml").write_text(
        yaml.safe_dump(
            [
                {
                    "id": "mwi_chap_districts",
                    "name": "Malawi malaria benchmark districts (27)",
                    "description": (
                        "Malawi districts (ADM2), NSO Malawi / OCHA ROSEA via geoBoundaries gbOpen, "
                        "boundary year 2020, release 9469f09, CC BY 3.0 IGO. These are the 27 districts with "
                        "monthly malaria cases in Elling et al. (Zenodo 17161438); Likoma is excluded. "
                        "District-level cases cannot support facility-level estimates without another case source."
                    ),
                    "id_property": "id",
                    "provider": "malawi_chap_boundaries",
                    "params": {"path": str(geojson)},
                    "license": "CC-BY-3.0-IGO",
                    "source_url": BOUNDARIES_URL,
                    "attribution": "National Statistics Office of Malawi / OCHA ROSEA, via geoBoundaries gbOpen",
                }
            ],
            sort_keys=False,
        )
    )
    config.write_text(
        yaml.safe_dump(
            {
                **existing_config,
                "id": "malawi-chap",
                "name": "Malawi Chap preparation",
                "extent": {"name": "Malawi", "bbox": BBOX, "country_code": "MWI"},
                "data_dir": "./ocs-data",
                "plugins_dir": str(plugins),
                "utc_offset_hours": 0,
            },
            sort_keys=False,
        )
    )
    return config


@contextmanager
def local_ocs(config: Path, port: int, credentials: Path) -> Generator[httpx.Client]:
    """Run and clean up a dedicated server, refusing to reuse an occupied port."""
    with socket.socket() as check:
        check.bind(("127.0.0.1", port))
    env = os.environ.copy()
    env.update(
        GEOSERVICE_CONFIG=str(config),
        GOOGLE_APPLICATION_CREDENTIALS=str(credentials),
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        DASK_NUM_WORKERS="2",
        GEOSERVICE_MAX_CONCURRENT_JOBS="1",
    )
    log_path = config.parent / "ocs.log"
    with log_path.open("a") as log:
        process = subprocess.Popen(
            [
                sys.executable,
                "-c",
                # Pre-import xclim once: its circular imports can race with GeoService's
                # background process discovery and the first data-source request.
                "import xclim; import sys; import uvicorn; "
                "uvicorn.run('open_geo_service.main:app', host='127.0.0.1', port=int(sys.argv[1]))",
                str(port),
            ],
            env=env,
            cwd=REPO,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:
            with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=600) as client:
                for _ in range(120):
                    if process.poll() is not None:
                        raise RuntimeError(f"GeoService exited; see {log_path}")
                    try:
                        if client.get("/extent", timeout=2).status_code == 200:
                            break
                    except httpx.TransportError:
                        pass
                    time.sleep(1)
                else:
                    raise RuntimeError(f"GeoService startup timed out; see {log_path}")
                yield client
        finally:
            process.terminate()
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def ingest(client: httpx.Client, dataset: str, start: str, end: str) -> str:
    """Drive the actual GeoService ingestion route and wait for its persistent job result."""
    print(f"Ingesting {dataset}: {start} to {end}", flush=True)
    response = client.post(
        "/ingestions",
        headers={"Prefer": "respond-async"},
        json={"dataset_id": dataset, "start": start, "end": end, "publish": True},
    )
    response.raise_for_status()
    href = response.headers["Location"]
    deadline = time.monotonic() + 7200
    last_message = None
    while time.monotonic() < deadline:
        response = client.get(href)
        response.raise_for_status()
        job = response.json()
        if job["status"] == "successful":
            return str(job["jobID"])
        if job["status"] in {"failed", "canceled", "cancelled", "dismissed"}:
            raise RuntimeError(f"GeoService {dataset} failed: {job.get('error')}")
        message = job.get("progress", {}).get("message")
        if message and message != last_message:
            print(f"  {message}", flush=True)
            last_message = message
        time.sleep(3)
    raise TimeoutError(f"GeoService job still running: {href}")


def aggregate(
    client: httpx.Client,
    dataset: str,
    start: str,
    end: str,
    collection: dict[str, Any],
    annual: bool,
    target: Path,
    *,
    daily: bool = False,
) -> pd.DataFrame:
    """Execute GeoService's Chap export workflow, preserving its raw CSV for inspection."""
    print(f"Aggregating {dataset} through GeoService", flush=True)
    graph = {
        "aggregate": {
            "process_id": "aggregate_to_chap_csv",
            "arguments": {
                "dataset_id": dataset,
                "temporal_extent": [start, end],
                "geometries": collection,
                "method": "sum" if annual else "mean",
                "period_type": "year" if annual else "day" if daily else "month",
            },
            "result": True,
        }
    }
    response = client.post("/result", json={"process": {"process_graph": graph}})
    if response.is_error:
        raise RuntimeError(f"GeoService aggregation failed ({response.status_code}): {response.text[:1500]}")
    target.write_bytes(response.content)
    frame = pd.read_csv(target, dtype={"location": str, "time_period": str})
    if not annual and not daily:
        frame["time_period"] = pd.to_datetime(frame["time_period"], format="%Y%m").dt.strftime("%Y-%m")
    return frame


def monthly_humidity(daily: pd.DataFrame, locations: list[str]) -> pd.DataFrame:
    """Require every daily district value before calculating calendar-month means."""
    dates = pd.date_range("2005-01-01", "2023-12-31", freq="D")
    expected = pd.MultiIndex.from_product([locations, dates.strftime("%Y%m%d")], names=KEY)
    actual = pd.MultiIndex.from_frame(daily[KEY])
    if actual.has_duplicates or set(actual) != set(expected):
        raise ValueError("Humidity must cover every district-day exactly once; partial months are not accepted")
    if not np.isfinite(daily["rh"]).all():
        raise ValueError("Missing or non-finite daily humidity")
    frame = daily.copy()
    frame["time_period"] = pd.to_datetime(frame["time_period"], format="%Y%m%d").dt.strftime("%Y-%m")
    return frame.groupby(KEY, as_index=False)["rh"].mean().rename(columns={"rh": "mean_relative_humidity"})


def assemble(health: pd.DataFrame, climate: list[pd.DataFrame], population: pd.DataFrame) -> pd.DataFrame:
    """Join the full panel, refusing missing, duplicated or implausible results."""
    result = health.copy()
    for frame in climate:
        if set(pd.MultiIndex.from_frame(frame[KEY])) != set(pd.MultiIndex.from_frame(health[KEY])):
            raise ValueError("Climate coverage does not match case coverage")
        result = result.merge(frame, on=KEY, how="left", validate="one_to_one")
    result["year"] = result["time_period"].str[:4]
    annual = population.rename(columns={"time_period": "year"})
    result = result.merge(annual, on=["location", "year"], how="left", validate="many_to_one")
    result = result[COLUMNS].sort_values(KEY).reset_index(drop=True)
    numbers = ["disease_cases", "population", *CLIMATE_VARIABLES]
    if result.isna().any().any() or not np.isfinite(result[numbers].to_numpy()).all():
        raise ValueError("Incomplete or non-finite harmonized output")
    if (result["population"] <= 0).any() or (result["rainfall"] < 0).any():
        raise ValueError("Nonpositive population or negative precipitation; inspect the source, do not silently clip")
    if not result["mean_temperature"].between(-10, 50).all():
        raise ValueError("Implausible temperature: check Kelvin conversion")
    if not result["mean_relative_humidity"].between(0, 100).all():
        raise ValueError("Relative humidity outside 0 to 100 percent")
    if (result["rainfall"] > 3000).any():
        raise ValueError("Implausible monthly rainfall: check accumulation units")
    return result


def deliver(work: Path, result: pd.DataFrame, jobs: list[str]) -> None:
    """Write a data dictionary, provenance manifest and adjustment flags with the CSV."""
    output = work / "output"
    csv_path = output / f"{STEM}.csv"
    result.to_csv(csv_path, index=False)
    case_source = {
        "provider": "Maxwell T. Elling et al.",
        "url": "https://doi.org/10.5281/zenodo.17161438",
        "license": "CC-BY-4.0",
        "download_url": CASES_URL,
    }
    boundary_source = {
        "provider": "NSO Malawi / OCHA ROSEA via geoBoundaries gbOpen",
        "url": BOUNDARIES_URL,
        "license": "CC-BY-3.0-IGO",
        "version": "9469f09",
        "boundary_year": 2020,
    }
    climate_source = {
        "provider": "Copernicus Climate Change Service (C3S); monthly aggregation by Google Earth Engine",
        "url": CLIMATE_URL,
        "collection": "ECMWF/ERA5_LAND/MONTHLY_AGGR",
        "license": "Licence to Use Copernicus Products",
        "license_url": "https://apps.ecmwf.int/datasets/licences/copernicus/",
        "attribution": "Contains modified Copernicus Climate Change Service information (2005-2023).",
    }
    population_source = {
        "provider": "WorldPop, University of Southampton",
        "license": "CC-BY-4.0",
        "url": "https://www.worldpop.org/",
        "products": {"2005-2020": "Global_2000_2020_1km_UNadj", "2021-2023": "Global2 R2025A UA v1, constrained, 1km"},
    }
    humidity_source = {
        "provider": "Copernicus Climate Change Service (C3S), processed by Open Geo Service",
        "url": HUMIDITY_URL,
        "data_source_id": HUMIDITY_DATASET,
        "inputs": ["hourly 2m temperature (t2m)", "hourly 2m dewpoint temperature (d2m)"],
        "license": "CC-BY",
        "license_url": HUMIDITY_URL,
        "attribution": climate_source["attribution"],
    }
    definitions = [
        ("time_period", "string", None, "Calendar month in UTC, YYYY-MM.", {"provider": "Derived"}),
        ("location", "string", None, "Pinned geoBoundaries shapeID, equal to GeoJSON feature.id.", boundary_source),
        (
            "disease_cases",
            "number",
            "cases",
            "Processed malaria case count; fractional source values retained.",
            case_source,
        ),
        (
            "population",
            "number",
            "persons",
            "Overlap-weighted sum of annual people per pixel, repeated each month of that year.",
            population_source,
        ),
        ("location_name", "string", None, "District name, matched using the authors' ID_1 lookup.", boundary_source),
        (
            "rainfall",
            "number",
            "mm",
            "District overlap-weighted mean of monthly total_precipitation_sum multiplied by 1000. "
            "No additional days-in-month multiplier.",
            climate_source,
        ),
        (
            "mean_temperature",
            "number",
            "degC",
            "District overlap-weighted mean of monthly temperature_2m minus 273.15.",
            climate_source,
        ),
        (
            "mean_relative_humidity",
            "number",
            "%",
            "Open Geo Service computes relative humidity for each hour from CDS ERA5-Land temperature and "
            "dewpoint using earthkit-meteo's IFS saturation formula over water, then averages to UTC days. "
            "Daily district overlap-weighted means are averaged over every day of each calendar month; "
            "missing district-days are rejected.",
            humidity_source,
        ),
    ]
    notes = [
        "Likoma has no case series and is excluded; missing cases have not been replaced with zeros.",
        "Source combines surveillance series and averages overlaps. Fractional cases are intentional.",
        "The authors adjusted 2013-2014 for underreporting using relationships with Mozambique. "
        "Treat these years separately in benchmark sensitivity analysis.",
        "A fixed 2020 boundary geography is applied to all years; "
        "historical boundary equivalence has not been independently established.",
        "WorldPop changes product in 2021. The discontinuity may reflect methods as well as population changes.",
        "Population sums cover valid WorldPop cells within the supplied polygons; "
        "coastline and boundary differences may prevent conservation of national totals.",
        "GeoService weights partial grid cells by polygon overlap (exactextract). "
        "Climate means are not population weighted.",
        "TOTALPOP and CASERATE1000 from the health source are excluded: "
        "population uses WorldPop and outcome-derived case rates would leak the target.",
        "This schema is a data dictionary following climate-health-data conventions, not a JSON Schema validator.",
    ]
    schema = {
        "dataset": {
            "name": "Chap harmonized Malawi malaria dataset",
            "description": "Monthly district malaria, WorldPop population and ERA5-Land climate.",
            "country_iso3": "MWI",
            "country_name": "Malawi",
            "admin_level": 2,
            "admin_geojson_file": f"{STEM}.geojson",
            "temporal_resolution": "monthly",
            "time_range": {"start": str(result.time_period.min()), "end": str(result.time_period.max())},
            "primary_key": KEY,
            "row_count": len(result),
            "location_count": int(result.location.nunique()),
            "created_utc": datetime.now(UTC).isoformat(),
            "output_csv": csv_path.name,
        },
        "fields": [{"name": n, "type": t, "unit": u, "description": d, "source": s} for n, t, u, d, s in definitions],
        "limitations": notes,
        "related_paper": "https://www.nature.com/articles/s43856-026-01681-9",
        "district_crosswalk_source": (
            f"https://github.com/Mte2112/climalmal-2025/blob/{AUTHORS_REVISION}/data/shapefiles.tar.gz"
        ),
        "crosswalk_method": "ID_1/NAME_1 from authors' MWI_adm1.shp; exact name join to geoBoundaries. "
        "Authors' geometries are not redistributed.",
    }
    write_json(output / f"{STEM}_schema.json", schema)
    flags = result[KEY].copy()
    flags["source_adjusted_year"] = result.time_period.str[:4].isin(["2013", "2014"])
    flags["fractional_source_cases"] = result.disease_cases % 1 != 0
    flags.to_csv(output / "case_quality_flags.csv", index=False)
    metrics = {
        name: {"min": float(result[name].min()), "max": float(result[name].max())}
        for name in ["disease_cases", "population", *CLIMATE_VARIABLES]
    }
    report = {
        "rows": len(result),
        "districts": int(result.location.nunique()),
        "months": int(result.time_period.nunique()),
        "missing_values": int(result.isna().sum().sum()),
        "duplicate_keys": int(result.duplicated(KEY).sum()),
        "fractional_case_rows": int(flags.fractional_source_cases.sum()),
        "adjusted_year_rows": int(flags.source_adjusted_year.sum()),
        "ranges": metrics,
        "limitations": notes,
    }
    write_json(output / "validation_report.json", report)
    source_hashes = {str(p.relative_to(work)): sha256(p) for p in sorted((work / "downloads").glob("*")) if p.is_file()}
    output_hashes = {p.name: sha256(p) for p in sorted(output.iterdir()) if p.is_file() and p.name != "provenance.json"}
    write_json(
        output / "provenance.json",
        {
            "generated_utc": datetime.now(UTC).isoformat(),
            "geoservice": {
                "version": version("open-geo-service"),
                "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
                "dependency_lock_sha256": sha256(REPO / "uv.lock"),
            },
            "ocs_jobs": jobs,
            "source_sha256": source_hashes,
            "output_sha256": output_hashes,
            "script_sha256": sha256(Path(__file__)),
            "plugin_sha256": {
                str(path.relative_to(EXAMPLE)): sha256(path) for path in sorted((EXAMPLE / "plugins").rglob("*.py"))
            },
            "crosswalk_sha256": sha256(EXAMPLE / "district_crosswalk.csv"),
            "source_urls": {
                "cases": CASES_URL,
                "boundaries": BOUNDARIES_URL,
                "rainfall_and_temperature": CLIMATE_URL,
                "humidity": HUMIDITY_URL,
            },
            "humidity": {
                "data_source_id": HUMIDITY_DATASET,
                "daily_periods": len(pd.date_range("2005-01-01", "2023-12-31")),
                "daily_export_sha256": sha256(work / "aggregates/humidity_daily.csv"),
                "aggregation": "Hourly RH -> daily grid mean -> daily district mean -> calendar-month mean",
            },
            "reproducibility": "Cached Earth Engine and WorldPop rasters are retained. "
            "CDS hourly inputs are read remotely; retain the ingested humidity store and daily export, "
            "because upstream reanalysis can be revised. No credentials are included.",
        },
    )
    print(json.dumps(report, indent=2), flush=True)
    print(f"Outputs: {output}", flush=True)


def main() -> None:
    """Prepare sources, ingest and aggregate through GeoService, then validate and deliver."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, default=REPO / "data/malawi-chap")
    parser.add_argument("--port", type=int, default=9001)
    parser.add_argument("--env-file", type=Path, help="Load CDS_API_KEY and other environment settings from this file")
    parser.add_argument(
        "--credentials",
        type=Path,
        help="Earth Engine service-account JSON; defaults to GOOGLE_APPLICATION_CREDENTIALS or the local key path",
    )
    args = parser.parse_args()
    if args.env_file is not None:
        load_dotenv(args.env_file.expanduser(), override=False)
    if not os.environ.get("CDS_API_KEY", "").strip():
        parser.error("Set CDS_API_KEY in the environment or supply --env-file for the hourly-derived humidity source")
    work = args.work_dir.expanduser().resolve()
    work.mkdir(parents=True, exist_ok=True)
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
    if not credentials.is_file():
        parser.error("Provide an Earth Engine service-account JSON via --credentials or GOOGLE_APPLICATION_CREDENTIALS")
    health, collection = prepare_health(work)
    config = configure(work, collection)
    aggregates = work / "aggregates"
    aggregates.mkdir(exist_ok=True)
    climate = []
    populations = []
    jobs = []
    with local_ocs(config, args.port, credentials) as client:
        response = client.post("/features/mwi_chap_districts/refresh")
        response.raise_for_status()
        for variable in ["rainfall", "mean_temperature"]:
            dataset = f"mwi_{variable}"
            jobs.append(ingest(client, dataset, "2005-01", "2023-12"))
            climate.append(
                aggregate(
                    client, dataset, "2005-01-01", "2023-12-31", collection, False, aggregates / f"{variable}.csv"
                )
            )
        jobs.append(ingest(client, HUMIDITY_DATASET, "2005-01-01", "2023-12-31"))
        daily = aggregate(
            client,
            HUMIDITY_DATASET,
            "2005-01-01",
            "2023-12-31",
            collection,
            False,
            aggregates / "humidity_daily.csv",
            daily=True,
        )
        humidity = monthly_humidity(daily, health["location"].unique().tolist())
        humidity.to_csv(aggregates / "mean_relative_humidity.csv", index=False)
        climate.append(humidity)
        for product, start, end in [("legacy", "2005", "2020"), ("global2", "2021", "2023")]:
            dataset = f"mwi_population_{product}"
            jobs.append(ingest(client, dataset, start, end))
            populations.append(
                aggregate(
                    client,
                    dataset,
                    f"{start}-01-01",
                    f"{end}-12-31",
                    collection,
                    True,
                    aggregates / f"population_{product}.csv",
                )
            )
    result = assemble(health, climate, pd.concat(populations, ignore_index=True))
    deliver(work, result, jobs)


if __name__ == "__main__":
    main()
