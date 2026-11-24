"""Import the prepared Malawi benchmark into DHIS2 without changing Lao metadata.

Run from the checkout with `python -m examples.malawi_chap.import_dhis2 --commit`.
Credentials come from DHIS2_IMPORT_TOKEN; local .env is supported. The default run
only prepares files and validates metadata. Every committed value is read back.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from shapely.geometry import mapping, shape
from shapely.ops import unary_union

ROOT = Path(__file__).resolve().parents[2]
STEM = "chap_MWI_admin2_monthly"
PREFIX = "MWI_CHAP_"
REGIONS = {
    "Northern": ["Chitipa", "Karonga", "Likoma", "Mzimba", "Nkhata Bay", "Rumphi"],
    "Central": ["Dedza", "Dowa", "Kasungu", "Lilongwe", "Mchinji", "Nkhotakota", "Ntcheu", "Ntchisi", "Salima"],
    "Southern": [
        "Balaka",
        "Blantyre",
        "Chikwawa",
        "Chiradzulu",
        "Machinga",
        "Mangochi",
        "Mulanje",
        "Mwanza",
        "Neno",
        "Nsanje",
        "Phalombe",
        "Thyolo",
        "Zomba",
    ],
}
CASE_URL = "https://doi.org/10.5281/zenodo.17161438"
PAPER_URL = "https://www.nature.com/articles/s43856-026-01681-9"
BOUNDARY_URL = "https://www.geoboundaries.org/countryDownloads.html"
CLIMATE_URL = "https://developers.google.com/earth-engine/datasets/catalog/ECMWF_ERA5_LAND_MONTHLY_AGGR"
FIELDS = {
    "disease_cases": ("Malaria cases - Malawi public benchmark (Elling et al.)", "MWI malaria cases (Elling)", "SUM"),
    "rainfall": ("Malawi monthly rainfall (mm) - ERA5-Land", "MWI rainfall mm (ERA5-Land)", "AVERAGE"),
    "mean_temperature": ("Malawi mean temperature (C) - ERA5-Land", "MWI temperature C (ERA5-Land)", "AVERAGE"),
    "mean_relative_humidity": (
        "Malawi derived relative humidity (%) - ERA5-Land",
        "MWI humidity % (ERA5-Land)",
        "AVERAGE",
    ),
    "population": ("Malawi annual population - WorldPop", "MWI population (WorldPop)", "AVERAGE_SUM_ORG_UNIT"),
}
DATASETS = {
    "MALARIA": ("Malawi malaria - public benchmark 2005-2023", "MWI malaria benchmark", "Monthly", ["disease_cases"]),
    "CLIMATE": (
        "Malawi climate - ERA5-Land monthly 2005-2023",
        "MWI climate ERA5-Land",
        "Monthly",
        ["rainfall", "mean_temperature", "mean_relative_humidity"],
    ),
    "POPULATION": (
        "Malawi population - WorldPop annual 2005-2023",
        "MWI population WorldPop",
        "Yearly",
        ["population"],
    ),
}


def save(path: Path, value: Any) -> None:
    """Write reviewable payloads and reports, never credential material."""
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


class Importer:
    """Prepare isolated metadata and drive validated, verified DHIS2 imports."""

    def __init__(self, client: httpx.Client, work: Path) -> None:
        """Read the generated bundle and verify its recorded checksums."""
        self.client = client
        self.work = work
        self.folder = work / "dhis2"
        self.folder.mkdir(exist_ok=True)
        output = work / "output"
        manifest = json.loads((output / "provenance.json").read_text())
        for name, digest in manifest["output_sha256"].items():
            if hashlib.sha256((output / name).read_bytes()).hexdigest() != digest:
                raise ValueError(f"Generated file was modified: {name}")
        self.frame = pd.read_csv(output / f"{STEM}.csv", dtype={"location": str, "time_period": str})
        self.schema = json.loads((output / f"{STEM}_schema.json").read_text())
        self.boundaries = json.loads((work / "downloads/boundaries.geojson").read_text())["features"]
        self.crosswalk = pd.read_csv(Path(__file__).parent / "district_crosswalk.csv")
        self.ids_path = self.folder / "uid-map.json"
        self.ids: dict[str, str] = json.loads(self.ids_path.read_text()) if self.ids_path.exists() else {}
        self.allocated: list[str] = []
        self.orgunits: dict[str, str] = {}
        self.elements: dict[str, str] = {}
        self.datasets: dict[str, str] = {}
        self.metadata: dict[str, Any] = {}
        self.payloads: dict[str, dict[str, Any]] = {}
        self.me = self.get("/api/me", {"fields": "id,organisationUnits[id],dataViewOrganisationUnits[id]"})
        self.default_cc = self.default("categoryCombos")
        self.default_coc = self.default("categoryOptionCombos")

    def get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        """Read the target API with explicit errors."""
        response = self.client.get(path, params=params)
        response.raise_for_status()
        return response.json()

    def default(self, resource: str) -> str:
        """Resolve the actual instance default instead of assuming a category UID."""
        items = self.get(f"/api/{resource}", {"filter": "name:eq:default", "fields": "id", "paging": "false"})[resource]
        if len(items) != 1:
            raise ValueError(f"Expected exactly one default {resource}")
        return str(items[0]["id"])

    def uid(self, resource: str, suffix: str) -> str:
        """Reuse our code or allocate an opaque UID from DHIS2's own generator."""
        code = PREFIX + suffix
        existing = self.get(f"/api/{resource}", {"filter": f"code:eq:{code}", "fields": "id,code", "paging": "false"})[
            resource
        ]
        if len(existing) > 1:
            raise ValueError(f"Ambiguous metadata code: {code}")
        if existing:
            current = str(existing[0]["id"])
            if code in self.ids and self.ids[code] != current:
                raise ValueError(f"UID mapping disagrees with server: {code}")
            self.ids[code] = current
        elif code not in self.ids:
            if not self.allocated:
                self.allocated = list(self.get("/api/system/id", {"limit": 80})["codes"])
            self.ids[code] = self.allocated.pop()
        save(self.ids_path, self.ids)
        return self.ids[code]

    def shared(self) -> dict[str, Any]:
        """Make metadata readable while reserving writes to the importing user."""
        return {"owner": self.me["id"], "public": "r-------", "external": False}

    def prepare(self) -> None:
        """Create country/region/district metadata, source-labelled datasets and a dashboard."""
        by_name = {f["properties"]["shapeName"]: f for f in self.boundaries}
        if set(by_name) != {name for names in REGIONS.values() for name in names}:
            raise ValueError("Expected all 28 named Malawi districts")
        country = self.uid("organisationUnits", "COUNTRY")
        self.orgunits["Malawi"] = country
        units = [
            {
                "id": country,
                "code": PREFIX + "COUNTRY",
                "name": "Malawi",
                "shortName": "Malawi",
                "openingDate": "2005-01-01",
                "url": BOUNDARY_URL,
                "description": "Malawi public research benchmark. Country outline dissolved from geoBoundaries ADM2 "
                "2020 districts, release 9469f09, CC BY 3.0 IGO. Opening date is benchmark coverage, "
                "not administrative establishment. Land outline only.",
                "geometry": mapping(unary_union([shape(f["geometry"]) for f in self.boundaries]).simplify(0.001)),
            }
        ]
        for region, names in REGIONS.items():
            region_id = self.uid("organisationUnits", "REGION_" + region.upper())
            units.append(
                {
                    "id": region_id,
                    "code": PREFIX + "REGION_" + region.upper(),
                    "name": region + " Region",
                    "shortName": region + " Region",
                    "parent": {"id": country},
                    "openingDate": "2005-01-01",
                    "url": BOUNDARY_URL,
                    "description": "Region outline dissolved from constituent geoBoundaries ADM2 districts (2020, "
                    "release 9469f09, CC BY 3.0 IGO). Simplified for overview display only.",
                    "geometry": mapping(
                        unary_union([shape(by_name[name]["geometry"]) for name in names]).simplify(0.001)
                    ),
                }
            )
            for name in names:
                feature = by_name[name]
                location = feature["properties"]["shapeID"]
                number = int(self.crosswalk.loc[self.crosswalk.shapeName == name, "ID_1"].iloc[0])
                uid = self.uid("organisationUnits", f"DISTRICT_{number:02d}")
                self.orgunits[location] = uid
                units.append(
                    {
                        "id": uid,
                        "code": PREFIX + f"DISTRICT_{number:02d}",
                        "name": name,
                        "shortName": name,
                        "parent": {"id": region_id},
                        "openingDate": "2005-01-01",
                        "url": BOUNDARY_URL,
                        "description": f"Malawi district. geoBoundaries shapeID {location}; "
                        f"authors' DISTRICT {number}. "
                        "NSO Malawi / OCHA ROSEA via geoBoundaries gbOpen ADM2, boundary year 2020, "
                        "release 9469f09, CC BY 3.0 IGO. Fixed geometry for all benchmark years. "
                        + (
                            "Likoma has no case series in this source; no zero values were imputed."
                            if number == 10
                            else ""
                        ),
                        "geometry": feature["geometry"],
                    }
                )
        field_schema = {field["name"]: field for field in self.schema["fields"]}
        elements = []
        for variable, (name, short_name, aggregation) in FIELDS.items():
            uid = self.uid("dataElements", "DE_" + variable.upper())
            self.elements[variable] = uid
            field = field_schema[variable]
            source = field["source"]
            description = (
                f"{field['description']} Source: {source['provider']}. {source['url']} License: {source['license']}. "
            )
            if variable == "disease_cases":
                description += "Monthly confirmed outpatient cases, all ages. Authors combine surveillance sources and "
                description += "average overlaps; fractional values retained. 2013-2014 adjusted for underreporting. "
                description += "Research benchmark, not a current operational surveillance feed. " + PAPER_URL
            elif variable == "population":
                description += "2005-2020 UN-adjusted 1 km series; 2021-2023 Global2 R2025A constrained UA v1. "
                description += "Product change in 2021. Stored yearly, not repeated monthly; summed across districts "
                description += "and averaged across years in DHIS2 analytics."
            else:
                description += "Canonical resolution: district/month. DHIS2 regional views are unweighted means of "
                description += "district summaries. Reaggregate rasters in OCS for area-weighted regional estimates. "
                if variable == "rainfall":
                    description += "Coarser temporal DHIS2 views show mean monthly rainfall, not annual totals."
            elements.append(
                {
                    "id": uid,
                    "code": PREFIX + "DE_" + variable.upper(),
                    "name": name,
                    "shortName": short_name,
                    "description": description,
                    "url": source["url"],
                    "valueType": "NUMBER",
                    "domainType": "AGGREGATE",
                    "aggregationType": aggregation,
                    "zeroIsSignificant": True,
                    "categoryCombo": {"id": self.default_cc},
                    "sharing": self.shared(),
                }
            )
        org_refs = [{"id": self.orgunits[location]} for location in sorted(self.frame.location.unique())]
        datasets = []
        for key, (name, short_name, period_type, variables) in DATASETS.items():
            uid = self.uid("dataSets", "DS_" + key)
            self.datasets[key] = uid
            description = "Imported public benchmark, 27 districts, 2005-2023. "
            description += "\n\n".join(
                str(element["description"])
                for element in elements
                if element["id"] in {self.elements[v] for v in variables}
            )
            datasets.append(
                {
                    "id": uid,
                    "code": PREFIX + "DS_" + key,
                    "name": name,
                    "shortName": short_name,
                    "description": description,
                    "periodType": period_type,
                    "mobile": False,
                    "categoryCombo": {"id": self.default_cc},
                    "organisationUnits": org_refs,
                    "dataSetElements": [
                        {"dataSet": {"id": uid}, "dataElement": {"id": self.elements[v]}} for v in variables
                    ],
                    "sharing": self.shared(),
                }
            )
        sources = [
            ("CASES", "Malawi malaria source - Elling et al. (CC BY 4.0)", CASE_URL),
            ("METHODS", "Malawi malaria source methods and adjustments", PAPER_URL),
            ("CLIMATE", "ERA5-Land source, variables and licence", CLIMATE_URL),
            ("POPULATION", "WorldPop population source (CC BY 4.0)", "https://www.worldpop.org/"),
            ("BOUNDARIES", "Malawi district boundaries - geoBoundaries", BOUNDARY_URL),
        ]
        documents = [
            {
                "id": self.uid("documents", "SOURCE_" + code),
                "code": PREFIX + "SOURCE_" + code,
                "name": name,
                "url": url,
                "external": True,
                "attachment": False,
                "sharing": self.shared(),
            }
            for code, name, url in sources
        ]
        dashboard = self.uid("dashboards", "OVERVIEW")
        visual = self.uid("visualizations", "CASES_TREND")
        self.metadata = {
            "organisationUnits": units,
            "dataElements": elements,
            "dataSets": datasets,
            "documents": documents,
            "visualizations": [
                {
                    "id": visual,
                    "code": PREFIX + "CASES_TREND",
                    "name": "Malawi malaria cases - public benchmark",
                    "description": "Sum of 27 districts. Elling et al.; 2013-2014 adjusted. Likoma excluded.",
                    "type": "LINE",
                    "columnDimensions": ["dx"],
                    "rowDimensions": ["pe"],
                    "filterDimensions": ["ou"],
                    "organisationUnits": [{"id": country}],
                    "periods": [{"id": m.replace("-", "")} for m in sorted(self.frame.time_period.unique())],
                    "dataDimensionItems": [
                        {
                            "dataDimensionItemType": "AGGREGATE_DATA_ELEMENT",
                            "dataElement": {"id": self.elements["disease_cases"]},
                        }
                    ],
                    "showData": False,
                    "sharing": self.shared(),
                }
            ],
            "dashboards": [
                {
                    "id": dashboard,
                    "code": PREFIX + "OVERVIEW",
                    "name": "Malawi malaria - data and sources",
                    "description": "Public district-month benchmark, its sources and limitations.",
                    "sharing": self.shared(),
                    "dashboardItems": [
                        {"text": self.dashboard_text(), "x": 0, "y": 0, "width": 60, "height": 28},
                        {"visualization": {"id": visual}, "x": 0, "y": 28, "width": 60, "height": 28},
                    ],
                }
            ],
        }
        self.prepare_values()
        save(self.folder / "metadata.json", self.metadata)
        save(self.folder / "location-map.json", self.orgunits)
        save(
            self.folder / "import-plan.json",
            {
                "dataSets": self.datasets,
                "dataElements": self.elements,
                "country": country,
                "dashboard": dashboard,
                "valueCounts": {key: len(p["dataValues"]) for key, p in self.payloads.items()},
            },
        )

    def dashboard_text(self) -> str:
        """Explain the workflow and sources where the imported data is viewed."""
        return (
            "## Malawi malaria public benchmark\n"
            "**January 2005-December 2023; 27 districts; 6,156 district-months.**\n\n"
            "### Where the data comes from\n"
            f"- **Malaria:** [Elling et al., Zenodo 17161438]({CASE_URL}), CC BY 4.0. Confirmed outpatient cases, "
            "all ages; overlapping surveillance series were averaged. Fractional counts are preserved. "
            f"2013-2014 were adjusted for underreporting: [methods]({PAPER_URL}).\n"
            f"- **Climate:** [Copernicus ERA5-Land monthly aggregates via Google Earth Engine]({CLIMATE_URL}). "
            "Monthly rainfall (mm) and mean temperature (C). Relative humidity follows the calculation "
            "documented in the imported bundle, also shown in its data-element description. "
            "Contains modified Copernicus Climate Change Service information (2005-2023).\n"
            "- **Population:** [WorldPop](https://www.worldpop.org/), CC BY 4.0, annual 1 km population counts. "
            "UN-adjusted 2005-2020; Global2 R2025A constrained 2021-2023. **Product changes in 2021.**\n"
            f"- **Boundaries:** [geoBoundaries gbOpen]({BOUNDARY_URL}), Malawi NSO / OCHA ROSEA, ADM2, "
            "2020 boundaries, release 9469f09, CC BY 3.0 IGO. Historical boundary equivalence is unverified.\n\n"
            "### How to use it\n"
            "Choose **Malawi > region > district**. Cases are in **Malawi malaria - public benchmark 2005-2023**; "
            "climate is in a separate monthly dataset; population is in an annual dataset. "
            "Likoma is shown geographically but has no imported case series. Missing values are not zeros.\n\n"
            "The district is the finest available case resolution. "
            "Finer raster pixels do not make the cases more local. "
            "Use district/month for analysis. Regional climate values in DHIS2 average district summaries; "
            "coarser rainfall periods show mean monthly rainfall, not annual totals. OCS can compute area-weighted "
            "values directly for other boundaries.\n\n"
            "### Workflow\n"
            "Source files -> OCS raster ingestion -> district aggregation -> CHAP table -> DHIS2 import. "
            "New sources can use the same district IDs, months and documented units. "
            "The published malaria file is a historical snapshot, so it is imported once rather than scheduled."
        ).replace("**", "*")

    def prepare_values(self) -> None:
        """Keep population annual and preserve every case and climate value."""
        for key, (_, _, period_type, variables) in DATASETS.items():
            frame = self.frame.copy()
            if period_type == "Yearly":
                frame["time_period"] = frame.time_period.str[:4]
                if frame.groupby(["location", "time_period"]).population.nunique().max() != 1:
                    raise ValueError("Population is not constant within a district-year")
                frame = frame.drop_duplicates(["location", "time_period"])
            values = []
            for row in frame.to_dict("records"):
                for variable in variables:
                    comment = {
                        "MALARIA": "Elling et al., Zenodo 17161438, CC BY 4.0.",
                        "CLIMATE": "Copernicus ERA5-Land; GeoService aggregation. See data-element provenance.",
                        "POPULATION": "WorldPop, CC BY 4.0; annual overlap-weighted population sum.",
                    }[key]
                    if key == "MALARIA" and str(row["time_period"])[:4] in {"2013", "2014"}:
                        comment += " Source adjusted for underreporting."
                    values.append(
                        {
                            "dataElement": self.elements[variable],
                            "period": str(row["time_period"]).replace("-", ""),
                            "orgUnit": self.orgunits[row["location"]],
                            "categoryOptionCombo": self.default_coc,
                            "attributeOptionCombo": self.default_coc,
                            "value": str(row[variable]),
                            "comment": comment,
                        }
                    )
            payload = {"dataSet": self.datasets[key], "dataValues": values}
            self.payloads[key] = payload
            save(self.folder / f"values-{key.lower()}.json", payload)

    def metadata_import(self, commit: bool) -> None:
        """Validate the complete dependency graph before an atomic metadata import."""
        mode = "COMMIT" if commit else "VALIDATE"
        response = self.client.post(
            "/api/metadata",
            params={
                "importMode": mode,
                "importStrategy": "CREATE_AND_UPDATE",
                "atomicMode": "ALL",
                "importReportMode": "FULL",
            },
            json=self.metadata,
        )
        report = response.json()
        save(self.folder / f"metadata-{mode.lower()}-report.json", report)
        response.raise_for_status()
        body = report.get("response", report)
        if body.get("status") not in {"OK", "SUCCESS"}:
            raise RuntimeError(f"Metadata {mode} was not clean; inspect the saved report")
        print(f"Metadata {mode}: {body.get('stats')}", flush=True)

    def allow_country(self) -> None:
        """Add Malawi to the current user's scopes while preserving existing assignments."""
        uid = self.orgunits["Malawi"]
        before = self.get(
            f"/api/users/{self.me['id']}", {"fields": "id,organisationUnits[id],dataViewOrganisationUnits[id]"}
        )
        if not (self.folder / "user-scope-before.json").exists():
            save(self.folder / "user-scope-before.json", before)
        operations = []
        for field in ["organisationUnits", "dataViewOrganisationUnits"]:
            values = before.get(field, [])
            if field == "dataViewOrganisationUnits" and not values:
                continue  # An empty view scope inherits the capture scope, including Lao.
            if uid not in {v["id"] for v in values}:
                operations.append({"op": "add", "path": "/" + field, "value": [*values, {"id": uid}]})
        if operations:
            response = self.client.patch(
                f"/api/users/{self.me['id']}", json=operations, headers={"Content-Type": "application/json-patch+json"}
            )
            response.raise_for_status()
            # In 2.42 the importer's cached UserDetails can retain the old roots
            # after a user patch. Refresh caches; keep strict import checks on.
            self.client.post("/api/maintenance/cacheClear").raise_for_status()

    def import_values(self) -> None:
        """Dry-run each batch, commit it, then read every value back from DHIS2."""
        for key, payload in self.payloads.items():
            for dry_run in [True, False]:
                response = self.client.post(
                    "/api/dataValueSets",
                    params={
                        "dryRun": str(dry_run).lower(),
                        "importStrategy": "CREATE_AND_UPDATE",
                        "strictDataElements": "true",
                        "strictOrganisationUnits": "true",
                    },
                    json=payload,
                )
                report = response.json()
                save(self.folder / f"values-{key.lower()}-{'validate' if dry_run else 'commit'}-report.json", report)
                response.raise_for_status()
                body = report.get("response", report)
                counts = body.get("importCount", {})
                if body.get("status") != "SUCCESS" or counts.get("ignored", 0) or body.get("conflicts"):
                    raise RuntimeError(f"{key} value import was not clean; inspect its saved report")
                print(f"{key} {'dry-run' if dry_run else 'commit'}: {counts}", flush=True)
            actual = self.get(
                "/api/dataValueSets",
                {
                    "dataSet": self.datasets[key],
                    "startDate": "2005-01-01",
                    "endDate": "2023-12-31",
                    "orgUnit": self.orgunits["Malawi"],
                    "children": "true",
                },
            )
            expected_values = {
                (v["dataElement"], v["orgUnit"], v["period"]): float(v["value"]) for v in payload["dataValues"]
            }
            actual_values = {
                (v["dataElement"], v["orgUnit"], v["period"]): float(v["value"]) for v in actual["dataValues"]
            }
            if expected_values.keys() != actual_values.keys():
                raise ValueError(f"{key} read-back coverage differs from the import")
            if not all(
                np.isclose(value, actual_values[k], rtol=1e-12, atol=1e-9) for k, value in expected_values.items()
            ):
                raise ValueError(f"{key} read-back values differ from the import")
            save(self.folder / f"verified-{key.lower()}.json", {"values_verified": len(actual_values), "match": True})
            print(f"{key}: verified all {len(actual_values)} values", flush=True)

    def analytics(self) -> None:
        """Build aggregate analytics across all historical years for charts and maps."""
        response = self.client.post(
            "/api/resourceTables/analytics",
            params={
                "skipResourceTables": "false",
                "skipEvents": "true",
                "skipEnrollment": "true",
                "skipTrackedEntities": "true",
                "skipOrgUnitOwnership": "true",
            },
        )
        response.raise_for_status()
        save(self.folder / "analytics-submission.json", response.json())
        print("Historical analytics rebuild submitted", flush=True)
        params = [
            ("dimension", "dx:" + self.elements["disease_cases"]),
            ("dimension", "pe:200501;201301;202312"),
            ("filter", "ou:" + self.orgunits["Malawi"]),
        ]
        deadline = time.monotonic() + 600
        expected = (
            self.frame[self.frame.time_period.isin(["2005-01", "2013-01", "2023-12"])]
            .groupby("time_period")
            .disease_cases.sum()
        )
        while time.monotonic() < deadline:
            response = self.client.get("/api/analytics", params=params)
            if response.is_success:
                data = response.json()
                headers = [h["name"] for h in data.get("headers", [])]
                if "pe" in headers and "value" in headers:
                    values = {
                        row[headers.index("pe")]: float(row[headers.index("value")]) for row in data.get("rows", [])
                    }
                    if all(
                        np.isclose(values.get(month.replace("-", ""), np.nan), value, rtol=1e-6)
                        for month, value in expected.items()
                    ):
                        save(self.folder / "analytics-verification.json", data)
                        print("Analytics verified against source totals for 200501, 201301 and 202312", flush=True)
                        return
            time.sleep(5)
        raise TimeoutError("Analytics verification is still pending; inspect the scheduler before retrying")

    def store_provenance(self) -> None:
        """Keep the schema, source checksums and UID mapping with the DHIS2 data."""
        manifest = json.loads((self.work / "output/provenance.json").read_text())
        payload = {
            "schema": self.schema,
            "file_provenance": manifest,
            "org_unit_mapping": self.orgunits,
            "data_elements": self.elements,
            "data_sets": self.datasets,
            "import_tool": "examples/malawi_chap/import_dhis2.py",
        }
        response = self.client.put("/api/dataStore/malawi-chap/provenance", json=payload)
        response.raise_for_status()
        if self.get("/api/dataStore/malawi-chap/provenance") != payload:
            raise ValueError("Provenance read-back differs from the saved metadata")


def main() -> None:
    """Prepare by default; commit and refresh analytics only when explicitly selected."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, default=ROOT / "data/malawi-chap")
    parser.add_argument("--url", default="http://localhost:8090")
    parser.add_argument("--commit", action="store_true")
    parser.add_argument("--analytics", action="store_true")
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")
    token = os.environ.get("DHIS2_IMPORT_TOKEN")
    if not token:
        parser.error("Set DHIS2_IMPORT_TOKEN in the environment or local .env")
    with httpx.Client(
        base_url=args.url.rstrip("/"), headers={"Authorization": f"ApiToken {token}"}, timeout=180
    ) as client:
        importer = Importer(client, args.work_dir.expanduser().resolve())
        importer.prepare()
        importer.metadata_import(False)
        if args.commit:
            importer.metadata_import(True)
            importer.allow_country()
            importer.import_values()
            importer.store_provenance()
            if args.analytics:
                importer.analytics()
            print("Dashboard: " + args.url.rstrip("/") + "/dhis-web-dashboard/#/" + importer.ids[PREFIX + "OVERVIEW"])
        else:
            print("Prepared and validated metadata. Use --commit to import, optionally with --analytics.")


if __name__ == "__main__":
    main()
