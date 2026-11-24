"""Malawi monthly malaria source and district reconciliation shared with the Chap build."""

from __future__ import annotations

import hashlib
import json
from io import BytesIO
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pandas as pd
from shapely.geometry import shape

from open_geo_service.features.providers import feature_provider

CASES_URL = "https://zenodo.org/records/17161438/files/malawi_malaria_incidence_mte2025_2023-2025.csv?download=1"
BOUNDARIES_URL = (
    "https://github.com/wmgeolab/geoBoundaries/raw/9469f09/"
    "releaseData/gbOpen/MWI/ADM2/geoBoundaries-MWI-ADM2_simplified.geojson"
)
CASES_SHA256 = "3b5f45b8883b747709ff8f5d899c7c0b9b00592cc5f82c88f4d97637b9175dca"
BOUNDARIES_SHA256 = "5d1840b4c11dc39947210356279366dce59c1670a1630d4b157f6763902fdd57"


def district_panel(
    cases: pd.DataFrame, crosswalk: pd.DataFrame, source: dict[str, Any]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Match the authors' district codes to the pinned boundaries and require the full case panel."""
    if len(crosswalk) != 28 or not crosswalk.shapeID.is_unique or set(crosswalk.ID_1) != set(range(1, 29)):
        raise ValueError("Invalid district crosswalk")
    if set(cases["DISTRICT"]) != set(range(1, 29)) - {10}:
        raise ValueError("Expected 27 districts, with only Likoma (10) absent")
    joined = cases.merge(crosswalk, left_on="DISTRICT", right_on="ID_1", validate="many_to_one")
    joined["time_period"] = pd.to_datetime(joined["DATETIME"]).dt.strftime("%Y-%m")
    joined = joined.rename(columns={"shapeID": "location", "shapeName": "location_name", "CASES": "disease_cases"})
    expected = pd.MultiIndex.from_product(
        [sorted(joined["location"].unique()), pd.period_range("2005-01", "2023-12", freq="M").astype(str)],
        names=["location", "time_period"],
    )
    actual = pd.MultiIndex.from_frame(joined[["location", "time_period"]])
    if actual.has_duplicates or len(joined) != 6156 or set(actual) != set(expected):
        raise ValueError("Cases must cover every district-month exactly once")
    if not np.isfinite(joined["disease_cases"]).all() or (joined["disease_cases"] < 0).any():
        raise ValueError("Missing, non-finite or negative cases")
    features = {f["properties"]["shapeID"]: f for f in source["features"]}
    selected = []
    for row in crosswalk.itertuples(index=False):
        if row.ID_1 == 10:
            continue
        feature = features[row.shapeID]
        if feature["properties"]["shapeName"] != row.shapeName or not shape(feature["geometry"]).is_valid:
            raise ValueError(f"Invalid boundary or district crosswalk: {row.shapeName}")
        feature["id"] = row.shapeID
        feature["properties"].update(id=row.shapeID, name=row.shapeName, source_district_id=row.ID_1)
        selected.append(feature)
    collection = {"type": "FeatureCollection", "features": selected}
    return joined[["time_period", "location", "location_name", "disease_cases"]], collection


@feature_provider("malawi_malaria_monthly")
def monthly_cases() -> tuple[dict[str, Any], str]:
    """Recheck the latest Zenodo release on every fetch before publishing a new snapshot."""
    with httpx.Client(timeout=90, follow_redirects=True) as client:
        release = client.get("https://zenodo.org/api/records/17161438/versions/latest")
        release.raise_for_status()
        if str(release.json()["id"]) != "17161438":
            raise ValueError("Zenodo has a newer malaria release. Review its coverage and crosswalk before updating.")
        case_file = client.get(CASES_URL)
        case_file.raise_for_status()
        boundary_file = client.get(BOUNDARIES_URL)
        boundary_file.raise_for_status()
    for response, expected in [(case_file, CASES_SHA256), (boundary_file, BOUNDARIES_SHA256)]:
        if hashlib.sha256(response.content).hexdigest() != expected:
            raise ValueError("Source checksum changed. Review the new release before publishing.")
    crosswalk = pd.read_csv(Path(__file__).with_name("district_crosswalk.csv"), dtype={"shapeID": str})
    source = json.loads(boundary_file.content)
    health, collection = district_panel(pd.read_csv(BytesIO(case_file.content)), crosswalk, source)
    values = health.pivot(index="location", columns="time_period", values="disease_cases")
    for feature in collection["features"]:
        feature["properties"].update(values.loc[feature["id"]].to_dict())
    # Likoma is in the boundary release, but has no cases in the surveillance release.
    lake_id = crosswalk.loc[crosswalk.ID_1 == 10, "shapeID"].item()
    likoma = next(f for f in source["features"] if f["properties"]["shapeID"] == lake_id)
    likoma["id"] = lake_id
    likoma["properties"].update(id=lake_id, name="Likoma", source_district_id=10)
    collection["features"].append(likoma)
    return collection, "zenodo-17161438-gb-9469f09"
