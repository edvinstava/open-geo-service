"""Exercise the example's command entry point and raster plugin fetch contracts."""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Generator
from contextlib import contextmanager
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import httpx
import numpy as np
import pandas as pd
import pytest
import rasterio
import xarray as xr
from fastapi.testclient import TestClient
from rasterio.transform import from_origin

from examples.malawi_chap import build
from examples.malawi_chap.plugins.rasters import malawi
from open_geo_service.ingestions.schemas import ArtifactFormat


def _raster(path: Path, values: np.ndarray) -> None:
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        count=len(values),
        height=2,
        width=2,
        dtype="float64",
        crs="EPSG:4326",
        transform=from_origin(33, -13, 1, 1),
        nodata=-99999,
    ) as target:
        target.write(values)


def test_daily_humidity_export_executes_real_workflow(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Run the submitted graph, weighted aggregation and CSV serializer through /result."""
    dates = pd.date_range("2005-01-01", "2023-12-31")
    values = np.broadcast_to(dates.day.to_numpy()[:, None, None], (len(dates), 2, 2)).astype(float).copy()
    values[:, :, 1] += 10
    cube = xr.Dataset(
        {"rh": (("t", "y", "x"), values)},
        coords={"t": dates, "y": [-13.5, -14.5], "x": [33.5, 34.5]},
        attrs={"proj:code": "EPSG:4326"},
    )
    artifact = SimpleNamespace(format=ArtifactFormat.ICECHUNK)
    monkeypatch.setattr("open_geo_service.openeo.execution._get_published_artifact", lambda _id: artifact)
    monkeypatch.setattr("open_geo_service.openeo.execution._open_artifact", lambda _a: cube)
    collection = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "id": "district",
                "properties": {},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[33, -15], [34.5, -15], [34.5, -13], [33, -13], [33, -15]]],
                },
            }
        ],
    }
    daily = build.aggregate(
        client,
        build.HUMIDITY_DATASET,
        "2005-01-01",
        "2023-12-31",
        collection,
        False,
        tmp_path / "daily.csv",
        daily=True,
    )
    assert len(daily) == 6939
    # One whole cell plus half its neighbour: an added 10 contributes 10/3.
    assert daily.loc[daily.time_period == "20050201", "rh"].item() == pytest.approx(1 + 10 / 3)
    monthly = build.monthly_humidity(daily, ["district"])
    assert len(monthly) == 228
    assert monthly.loc[monthly.time_period == "2008-02", "mean_relative_humidity"].item() == pytest.approx(15 + 10 / 3)


@pytest.mark.parametrize(
    "variable,expected",
    [
        ("rainfall", 100.0),
        ("mean_temperature", 26.85),
        ("mean_relative_humidity", 100 * np.exp(17.67 * 16.85 / 260.35 - 17.67 * 26.85 / 270.35)),
    ],
)
def test_monthly_plugin_fetch_units_and_month(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, variable: str, expected: float
) -> None:
    values = np.zeros((36, 2, 2))
    for month in range(12):
        values[month * 3 : month * 3 + 3] = np.array([300, 290, 0.1])[:, None, None]
    values[0] = 310  # January must not accidentally be selected for February.
    values[:, 0, 0] = -99999
    path = tmp_path / "year.tif"
    _raster(path, values)
    plugin = malawi.EarthEngineMonthlyPlugin(variable, str(tmp_path))
    monkeypatch.setattr(plugin, "download_year", lambda *_: path)
    result = plugin.fetch_period("2005-02", [33, -15, 35, -13])
    assert result[variable].dims == ("t", "y", "x")
    assert np.isnan(result[variable].values[0, 0, 0])
    assert result[variable].values[0, 1, 1] == pytest.approx(expected)
    assert str(result.t.values[0]).startswith("2005-02-01")
    assert len(asyncio.run(plugin.periods("2004-01", "2024-12"))) == 228


@pytest.mark.parametrize(
    "product,year,fragment",
    [
        ("legacy", "2005", "Global_2000_2020_1km_UNadj"),
        ("global2", "2021", "Global_2015_2030/R2025A"),
    ],
)
def test_population_fetch_masks_nodata_and_uses_ticket_product(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    product: str,
    year: str,
    fragment: str,
) -> None:
    source = tmp_path / "source.tif"
    _raster(source, np.array([[[1, 2], [3, -99999]]], dtype=float))
    original_client = httpx.Client

    def reply(request: httpx.Request) -> httpx.Response:
        assert fragment in str(request.url)
        assert year in str(request.url)
        return httpx.Response(200, content=source.read_bytes())

    monkeypatch.setattr(malawi.httpx, "Client", lambda **_: original_client(transport=httpx.MockTransport(reply)))
    plugin = malawi.MalawiWorldPopPlugin(product, str(tmp_path))
    result = plugin.fetch_period(year, [33, -15, 35, -13])
    assert float(result.population.sum()) == 6
    assert np.isnan(result.population.values[0, 1, 1])
    assert result.population.dims == ("t", "y", "x")


@pytest.mark.parametrize("missing_climate", [False, True, "missing_humidity_day", "unrelated_instance"])
def test_command_builds_complete_files_or_refuses_missing_climate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    missing_climate: bool | str,
    client: TestClient,
) -> None:
    downloads = tmp_path / "downloads"
    downloads.mkdir()
    crosswalk = pd.read_csv(build.EXAMPLE / "district_crosswalk.csv")
    months = pd.period_range("2005-01", "2023-12", freq="M").astype(str).tolist()
    cases = pd.DataFrame(
        [
            {"DISTRICT": district, "DATETIME": month + "-01", "CASES": 12.5 if month == "2013-01" else 13}
            for district in range(1, 29)
            if district != 10
            for month in months
        ]
    )
    cases.to_csv(downloads / "cases.csv", index=False)
    features = [
        {
            "type": "Feature",
            "properties": {"shapeID": row.shapeID, "shapeName": row.shapeName},
            "geometry": {"type": "Polygon", "coordinates": [[[33, -14], [34, -14], [34, -13], [33, -13], [33, -14]]]},
        }
        for row in crosswalk.itertuples()
    ]
    build.write_json(downloads / "boundaries.geojson", {"type": "FeatureCollection", "features": features})
    monkeypatch.setattr(build, "CASES_SHA256", build.sha256(downloads / "cases.csv"))
    monkeypatch.setattr(build, "BOUNDARIES_SHA256", build.sha256(downloads / "boundaries.geojson"))
    credential = tmp_path / "credential.json"
    credential.write_text("{}")
    monkeypatch.setenv("CDS_API_KEY", "test-key")
    requests = []

    def reply(request: httpx.Request) -> httpx.Response:
        requests.append(request.url.path)
        if request.url.path.endswith("/refresh"):
            return httpx.Response(200, json={})
        if request.url.path == "/ingestions":
            return httpx.Response(202, json={}, headers={"Location": "/ingestions/jobs/test-job"})
        if request.url.path == "/ingestions/jobs/test-job":
            return httpx.Response(200, json={"status": "successful", "jobID": "test-job"})
        assert request.url.path == "/result"
        args = json.loads(request.content)["process"]["process_graph"]["aggregate"]["arguments"]
        locations = [f["id"] for f in args["geometries"]["features"]]
        variable = args["dataset_id"].removeprefix("mwi_")
        annual = args["period_type"] == "year"
        if args["period_type"] == "day":
            assert args["dataset_id"] == build.HUMIDITY_DATASET
            dates = pd.date_range("2005-01-01", "2023-12-31")
            frame = pd.DataFrame(
                [
                    {"location": location, "time_period": date.strftime("%Y%m%d"), "rh": float(date.day)}
                    for location in locations
                    for date in dates
                ]
            )
            if missing_climate == "missing_humidity_day":
                frame = frame.iloc[:-1]
        elif annual:
            assert args["method"] == "sum"
            years = range(int(args["temporal_extent"][0][:4]), int(args["temporal_extent"][1][:4]) + 1)
            frame = pd.DataFrame(
                [
                    {"location": location, "time_period": str(year), "population": year * 100}
                    for location in locations
                    for year in years
                ]
            )
        else:
            assert args["method"] == "mean"
            frame = pd.DataFrame(
                [
                    {"location": location, "time_period": month.replace("-", ""), variable: 25.0}
                    for location in locations
                    for month in months
                ]
            )
            if missing_climate is True:
                frame = frame.iloc[:-1]
        return httpx.Response(200, content=frame.to_csv(index=False).encode(), headers={"Content-Type": "text/csv"})

    @contextmanager
    def server(*_: object) -> Generator[httpx.Client]:
        with httpx.Client(base_url="http://ocs.test", transport=httpx.MockTransport(reply)) as client:
            yield client

    monkeypatch.setattr(build, "local_ocs", server)
    monkeypatch.setattr(build.sys, "argv", ["build.py", "--work-dir", str(tmp_path), "--credentials", str(credential)])
    if missing_climate == "unrelated_instance":
        config = tmp_path / "geoservice.yaml"
        config.write_text("id: another-instance\n")
        with pytest.raises(ValueError, match="unrelated instance"):
            build.main()
        assert config.read_text() == "id: another-instance\n"
        assert not (tmp_path / "output").exists()
        assert not (tmp_path / "plugins").exists()
        return
    if missing_climate:
        match = "district-day" if missing_climate == "missing_humidity_day" else "Climate coverage"
        with pytest.raises(ValueError, match=match):
            build.main()
        assert not (tmp_path / "output" / f"{build.STEM}.csv").exists()
        return
    build.main()
    output = tmp_path / "output"
    result = pd.read_csv(output / f"{build.STEM}.csv")
    schema = json.loads((output / f"{build.STEM}_schema.json").read_text())
    boundaries = json.loads((output / f"{build.STEM}.geojson").read_text())
    assert len(result) == schema["dataset"]["row_count"] == 6156
    assert list(result.columns) == build.COLUMNS
    assert set(result.location) == {f["id"] for f in boundaries["features"]}
    assert set(result.loc[result.time_period == "2013-01", "disease_cases"]) == {12.5}
    assert set(result.loc[result.time_period == "2021-04", "population"]) == {202100}
    assert set(result.loc[result.time_period == "2008-02", "mean_relative_humidity"]) == {15.0}
    assert set(result.loc[result.time_period == "2007-02", "mean_relative_humidity"]) == {14.5}
    assert not result.isna().any().any()
    assert "TOTALPOP" not in result and "CASERATE1000" not in result
    assert requests.count("/result") == 5
    flags = pd.read_csv(output / "case_quality_flags.csv")
    assert int(flags.source_adjusted_year.sum()) == 648
    provenance = json.loads((output / "provenance.json").read_text())
    assert provenance["geoservice"]["dependency_lock_sha256"] == build.sha256(build.REPO / "uv.lock")
    assert len(provenance["geoservice"]["git_revision"]) == 40
    assert provenance["humidity"]["data_source_id"] == build.HUMIDITY_DATASET
    assert provenance["plugin_sha256"]["plugins/rasters/malawi.py"] == build.sha256(
        build.EXAMPLE / "plugins/rasters/malawi.py"
    )

    from examples.malawi_chap.plugins.vectors import case_series
    from open_geo_service import config as api_config
    from open_geo_service.features import providers, templates
    from open_geo_service.ingestions import services

    monkeypatch.setattr(templates, "CONFIGS_DIR", tmp_path / "plugins/vectors")
    templates.reset_feature_template_caches()
    monkeypatch.setattr(
        providers, "load_feature_providers", lambda: {"malawi_malaria_monthly": case_series.monthly_cases}
    )
    monkeypatch.setattr(api_config, "get_features_root", lambda: tmp_path / "vectors")
    monkeypatch.setattr(api_config, "get_data_root", lambda: tmp_path / "data")
    monkeypatch.setattr(services, "ARTIFACTS_DIR", tmp_path / "artifacts")
    monkeypatch.setattr(services, "ARTIFACTS_INDEX_PATH", tmp_path / "artifacts/records.json")
    case_bytes = (downloads / "cases.csv").read_bytes()
    boundary_bytes = (downloads / "boundaries.geojson").read_bytes()
    monkeypatch.setattr(case_series, "CASES_SHA256", hashlib.sha256(case_bytes).hexdigest())
    monkeypatch.setattr(case_series, "BOUNDARIES_SHA256", hashlib.sha256(boundary_bytes).hexdigest())
    failure = ""
    client_class = httpx.Client

    def source(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/versions/latest"):
            return httpx.Response(200, json={"id": 17161439 if failure == "release" else 17161438})
        if str(request.url) == case_series.CASES_URL:
            return httpx.Response(200, content=b"modified" if failure == "checksum" else case_bytes)
        assert str(request.url) == case_series.BOUNDARIES_URL
        return httpx.Response(200, content=boundary_bytes)

    monkeypatch.setattr(case_series.httpx, "Client", lambda **_: client_class(transport=httpx.MockTransport(source)))
    refresh = "/features/mwi_malaria_monthly/refresh"
    assert client.post(refresh).status_code == 200
    published = client.get("/features/mwi_malaria_monthly/data.parquet").content
    mapped = pd.read_parquet(BytesIO(published))
    assert len(mapped) == 28
    assert mapped.loc[mapped.name == "Likoma", "2013-01"].isna().all()
    assert (mapped.loc[mapped.name != "Likoma", "2013-01"] == 12.5).all()
    for failure, message in [("release", "newer malaria release"), ("checksum", "checksum changed")]:
        with pytest.raises(ValueError, match=message):
            client.post(refresh)
        assert client.get("/features/mwi_malaria_monthly/data.parquet").content == published
