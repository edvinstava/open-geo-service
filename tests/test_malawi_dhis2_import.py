"""Drive the import command through a simulated DHIS2 API, including read-back."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
import pytest

from examples.malawi_chap import import_dhis2 as importer


def _bundle(work: Path) -> None:
    output = work / "output"
    output.mkdir()
    (work / "downloads").mkdir()
    (work / "plugins/vectors").mkdir(parents=True)
    (work / "geoservice.yaml").write_text("id: malawi-chap\ndata_dir: ./ocs-data\n")
    crosswalk = pd.read_csv(Path(importer.__file__).parent / "district_crosswalk.csv")
    features = []
    rows = []
    for district in crosswalk.itertuples():
        features.append(
            {
                "type": "Feature",
                "properties": {"shapeName": district.shapeName, "shapeID": district.shapeID},
                "geometry": {"type": "Polygon", "coordinates": [[[33, -14], [34, -14], [34, -13], [33, -14]]]},
            }
        )
        if district.ID_1 == 10:
            continue
        for month in ["2005-01", "2005-02", "2013-01", "2023-12"]:
            rows.append(
                {
                    "location": district.shapeID,
                    "time_period": month,
                    "location_name": district.shapeName,
                    "disease_cases": 12.5,
                    "rainfall": 123.5,
                    "mean_temperature": 24.0,
                    "mean_relative_humidity": 75.0,
                    "population": int(month[:4]) * 100,
                }
            )
    pd.DataFrame(rows).to_csv(output / f"{importer.STEM}.csv", index=False)
    importer.save(work / "downloads/boundaries.geojson", {"type": "FeatureCollection", "features": features})
    importer.save(
        output / f"{importer.STEM}_schema.json",
        {
            "fields": [
                {
                    "name": variable,
                    "description": variable,
                    "source": {"provider": "Source", "license": "CC-BY-4.0", "url": "https://example.org/source"},
                }
                for variable in importer.FIELDS
            ]
        },
    )
    importer.save(
        output / "provenance.json",
        {"output_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir()}},
    )


@pytest.mark.parametrize("commit,corrupt_readback", [(False, False), (True, False), (True, True)])
def test_command_validates_before_write_and_checks_every_value(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    commit: bool,
    corrupt_readback: bool,
) -> None:
    _bundle(tmp_path)
    writes: list[str] = []
    metadata: dict[str, Any] = {}
    value_sets: dict[str, Any] = {}
    provenance: dict[str, Any] = {}
    validated: set[str] = set()
    metadata_validated = False
    scope = {"id": "Admin000001", "organisationUnits": [{"id": "Lao00000001"}], "dataViewOrganisationUnits": []}

    def respond(request: httpx.Request) -> httpx.Response:
        nonlocal metadata_validated
        path = request.url.path
        if request.method == "GET":
            if path in ["/api/me", "/api/users/Admin000001"]:
                return httpx.Response(200, json=scope)
            if path == "/api/system/id":
                return httpx.Response(200, json={"codes": [f"A{i:010}" for i in range(100)]})
            if path == "/api/dataValueSets":
                payload = value_sets[request.url.params["dataSet"]]
                if corrupt_readback:
                    payload = {**payload, "dataValues": payload["dataValues"][:-1]}
                return httpx.Response(200, json=payload)
            if path == "/api/dataStore/malawi-chap/provenance":
                return httpx.Response(200, json=provenance)
            resource = path.rsplit("/", 1)[-1]
            defaults = [{"id": "Default0001"}] if resource in ["categoryCombos", "categoryOptionCombos"] else []
            return httpx.Response(200, json={resource: defaults})
        if path == "/api/metadata":
            if request.url.params["importMode"] == "VALIDATE":
                metadata_validated = True
            else:
                assert metadata_validated
                writes.append("metadata")
                metadata.update(json.loads(request.content))
            return httpx.Response(200, json={"response": {"status": "OK", "stats": {"ignored": 0}}})
        if path == "/api/users/Admin000001":
            operations = json.loads(request.content)
            assert operations[0]["path"] == "/organisationUnits"
            assert operations[0]["value"][0] == {"id": "Lao00000001"}
            # Preserve the empty data-view list's inherited scope, rather than narrowing it.
            assert len(operations) == 1
            writes.append("scope")
            return httpx.Response(200, json={})
        if path == "/api/maintenance/cacheClear":
            assert "scope" in writes
            writes.append("cache")
            return httpx.Response(204)
        if path == "/api/dataValueSets":
            assert "cache" in writes
            assert request.url.params["strictOrganisationUnits"] == "true"
            payload = json.loads(request.content)
            uid = payload["dataSet"]
            if request.url.params["dryRun"] == "true":
                validated.add(uid)
            else:
                assert uid in validated
                value_sets[uid] = payload
                writes.append("values")
            return httpx.Response(
                200,
                json={
                    "response": {
                        "status": "SUCCESS",
                        "importCount": {
                            "imported": len(payload["dataValues"]),
                            "updated": 0,
                            "ignored": 0,
                            "deleted": 0,
                        },
                    }
                },
            )
        assert path == "/api/dataStore/malawi-chap/provenance"
        provenance.update(json.loads(request.content))
        return httpx.Response(201, json={})

    original = httpx.Client
    monkeypatch.setattr(
        importer.httpx, "Client", lambda **kwargs: original(transport=httpx.MockTransport(respond), **kwargs)
    )
    monkeypatch.setenv("DHIS2_IMPORT_TOKEN", "test-token")
    args = ["import_dhis2", "--work-dir", str(tmp_path), "--url", "http://dhis2.test"]
    monkeypatch.setattr("sys.argv", args + (["--commit"] if commit else []))
    if corrupt_readback:
        with pytest.raises(ValueError, match="read-back coverage"):
            importer.main()
        assert writes.count("values") == 1
        return
    importer.main()
    if not commit:
        assert not writes
        return
    assert (tmp_path / "geoservice.yaml").read_text() == "id: malawi-chap\ndata_dir: ./ocs-data\n"
    assert not list((tmp_path / "plugins/vectors").iterdir())
    assert len(metadata["organisationUnits"]) == 32
    assert len(metadata["dataSets"]) == 3
    assert len(metadata["dataElements"]) == 5
    plan = json.loads((tmp_path / "dhis2/import-plan.json").read_text())
    malaria = value_sets[plan["dataSets"]["MALARIA"]]["dataValues"]
    population = value_sets[plan["dataSets"]["POPULATION"]]["dataValues"]
    assert len(malaria) == 108 and len(population) == 81
    assert all(v["value"] == "12.5" for v in malaria)
    assert all(len(v["period"]) == 4 for v in population)
    assert any("adjusted" in v["comment"] for v in malaria if v["period"] == "201301")
    assert provenance["data_sets"] == plan["dataSets"]
    assert all(len(dataset["organisationUnits"]) == 27 for dataset in metadata["dataSets"])
