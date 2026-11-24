"""Exercise the enrichment command, preserving cases and exposing source gaps."""

from __future__ import annotations

import json
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
import pytest

from examples.malawi_chap import build, enrich


@pytest.mark.parametrize("problem", [None, "duplicate", "modified_base"])
def test_enrichment_command_preserves_base_and_flags_gaps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, problem: str | None
) -> None:
    bundle = tmp_path / "base"
    bundle.mkdir()
    work = tmp_path / "expanded"
    locations = [str(i) for i in range(27)]
    months = pd.period_range("2005-01", "2023-12", freq="M").astype(str)
    base = pd.MultiIndex.from_product([locations, months], names=build.KEY).to_frame(index=False)
    base["population"] = 1000.25
    base["disease_cases"] = 12.5
    base.to_csv(bundle / f"{build.STEM}.csv", index=False)
    features = [
        {
            "type": "Feature",
            "id": location,
            "properties": {"id": location},
            "geometry": {"type": "Polygon", "coordinates": [[[33, -14], [34, -14], [34, -13], [33, -13], [33, -14]]]},
        }
        for location in locations
    ]
    collection = {"type": "FeatureCollection", "features": features}
    build.write_json(bundle / f"{build.STEM}.geojson", collection)
    build.write_json(
        bundle / f"{build.STEM}_schema.json",
        {"dataset": {}, "fields": [{"name": "population"}], "limitations": ["Original humidity method"]},
    )
    base[build.KEY].to_csv(bundle / "case_quality_flags.csv", index=False)
    original_hashes = {p.name: build.sha256(p) for p in bundle.iterdir()}
    build.write_json(bundle / "provenance.json", {"output_sha256": original_hashes})
    if problem == "modified_base":
        (bundle / f"{build.STEM}.csv").write_text("modified")

    @contextmanager
    def server(*_: Any) -> Generator[None]:
        yield None

    monkeypatch.setattr(build, "local_ocs", server)
    monkeypatch.setattr(build, "ingest", lambda *_: "job-id")

    def aggregate(*_: Any) -> pd.DataFrame:
        # GeoService returns a rate; leap-year February must become 29 mm, not 1 mm.
        rain = base[build.KEY].copy()
        rain["precip"] = 1.0
        return rain

    monkeypatch.setattr(build, "aggregate", aggregate)
    monkeypatch.setattr(enrich, "earth_engine", lambda _: None)

    def fetch(_ee: Any, group: str, year: int | None, _collection: Any, _cache: Path) -> pd.DataFrame:
        monthly = group in {"weather", "vegetation"}
        periods = [f"{year}-{m:02d}" for m in range(1, 13)] if monthly else [str(year) if year else "static"]
        frame = pd.MultiIndex.from_product([locations, periods], names=build.KEY).to_frame(index=False)
        for name in enrich.GROUP_COLUMNS[group]:
            frame[name] = 0.5
        if group == "vegetation":
            frame["vegetation_good_quality_fraction"] = 1.0
        if group == "vegetation" and year == 2005:
            frame.loc[0, "ndvi"] = None
            frame.loc[1, "vegetation_good_quality_fraction"] = 0.05
        if group == "water":
            frame["water_classified_fraction"] = 1.0
        if group == "water" and year == 2005:
            frame.loc[0, "seasonal_water_detected_fraction"] = None
            frame.loc[0, "permanent_water_detected_fraction"] = None
            frame.loc[0, "water_classified_fraction"] = 0
            frame.loc[1, "water_classified_fraction"] = 0.05
        if problem == "duplicate" and group == "weather" and year == 2005:
            frame = pd.concat([frame, frame.iloc[:1]], ignore_index=True)
        return frame

    monkeypatch.setattr(enrich, "fetch_group", fetch)
    # Drive the real NOAA download and parser with the published column shape.
    text = "YR MON NINO1+2 ANOM NINO3 ANOM NINO4 ANOM NINO3.4 ANOM\n"
    text += "\n".join(f"{month[:4]} {month[5:]} 20 0 20 0 20 0 20 1.25" for month in months)
    monkeypatch.setattr(
        enrich.httpx,
        "get",
        lambda *_args, **_kwargs: httpx.Response(200, text=text, request=httpx.Request("GET", enrich.NINO_URL)),
    )
    monkeypatch.setattr("sys.argv", ["enrich", "--bundle", str(bundle), "--work-dir", str(work)])
    if problem:
        with pytest.raises(ValueError, match="Duplicate" if problem == "duplicate" else "checksum mismatch"):
            enrich.main()
        assert not (work / "output" / f"{build.STEM}.csv").exists()
        return
    enrich.main()
    output = work / "output"
    result = pd.read_csv(output / f"{build.STEM}.csv", dtype={"location": str})
    pd.testing.assert_frame_equal(result[base.columns], base)
    assert {p.name: build.sha256(p) for p in bundle.iterdir() if p.name != "provenance.json"} == original_hashes
    assert len(result.columns) == len(base.columns) + 35
    assert set(result.loc[result.time_period == "2008-02", "rainfall_chirps3"]) == {29.0}
    assert set(result.loc[result.time_period == "2007-02", "rainfall_chirps3"]) == {28.0}
    assert set(result.nino34_anomaly) == {1.25}
    assert result.loc[result.time_period > "2021-12", "seasonal_water_detected_fraction"].notna().all()
    unclassified = (result.location == "0") & result.time_period.str.startswith("2005")
    assert result.loc[unclassified, "seasonal_water_detected_fraction"].isna().all()
    assert set(result.loc[unclassified, "water_classified_fraction"]) == {0}
    assert (result.population_density_km2 * result.district_area_km2 - result.population).abs().max() < 1e-9
    flags = pd.read_csv(output / "covariate_quality_flags.csv")
    assert len(flags) == 12 * 2 + 1 + 2 + 12 * 2
    assert set(flags.loc[flags.variable == "ndvi", "reason"]) == {
        "no_valid_source_pixels_or_value",
        "partial_spatial_coverage",
    }
    assert len(flags.loc[flags.reason == "partial_spatial_coverage"]) == 26
    schema = json.loads((output / f"{build.STEM}_schema.json").read_text())
    assert "Original humidity method" in schema["limitations"]
    provenance = json.loads((output / "provenance.json").read_text())
    assert all(build.sha256(output / name) == value for name, value in provenance["output_sha256"].items())
