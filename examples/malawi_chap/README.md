# Malawi malaria benchmark through GeoService

This one-time, rerunnable pipeline produces a Malawi malaria dataset in the same
CSV / GeoJSON / data-dictionary arrangement as the dengue datasets in
[climate-health-data](https://github.com/dhis2/climate-health-data).
The build step produces files; the separate importer below loads them into DHIS2.
Neither command publishes to the data repository.

## Expanded environmental CSV

`enrich.py` adds 35 columns to an existing, checksum-verified bundle, keeping all
6,156 district-month rows and every original value. It writes a separate bundle.
An existing CSV with monthly-derived humidity keeps that method; enrichment does
not claim to have completed the blocked hourly-derived humidity rebuild.

```bash
uv run --all-extras --with earthengine-api==1.7.49 \
  python -m examples.malawi_chap.enrich \
  --bundle /absolute/path/to/existing/output \
  --work-dir /absolute/path/to/new/expanded-work
```

The script starts an isolated GeoService on port 9003 (`--port` overrides it) for
CHIRPS3 ingestion and district aggregation. The other raster sources are reduced
directly in Earth Engine, using the same district polygons and each source's
native grid. These alternatives use the existing Earth Engine service account;
they do not need EDH or CDSE keys. They do not appear as GeoService raster stores.
NOAA supplies a single monthly index table. Requests run sequentially, annual
batches keep Earth Engine calls small, and public responses are cached by the
exact computation. Use a new work directory to request fresh upstream results.

| Additional columns | Source and interpretation |
| --- | --- |
| `rainfall_chirps3` | Built-in CHIRPS3 monthly, 5 km. GeoService stores mm/day; multiplied by calendar days to produce mm/month. Kept alongside ERA5-Land rainfall for comparison. |
| `monthly_min_temperature`, `monthly_max_temperature` | ERA5-Land monthly pixel extrema, then district means in Celsius. These are not averages of daily minima/maxima. |
| `mean_dewpoint_temperature`, `mean_soil_temperature_0_7cm` | ERA5-Land monthly means in Celsius. |
| `soil_moisture_0_7cm`, `soil_moisture_7_28cm`, `soil_moisture_28_100cm` | ERA5-Land volumetric water fractions at three depths. |
| `surface_runoff_mm`, `evaporation_mm`, `solar_radiation_mj_m2` | ERA5-Land monthly accumulations. Evaporation is sign-reversed so positive means water loss; negative condensation is retained. The known swapped evaporation-component bands are not used. |
| `eastward_wind_10m`, `northward_wind_10m` | ERA5-Land monthly mean wind components in m/s, not mean wind speed. |
| `leaf_area_index_low_vegetation`, `leaf_area_index_high_vegetation` | ERA5-Land monthly leaf-area indices. |
| `ndvi`, `evi`, `vegetation_good_quality_fraction` | MOD13A3.061 monthly vegetation at 1 km. Indices use only SummaryQA=0 and valid scaled values; the fraction reports source quality. This is a different product from the built-in CLMS NDVI. |
| `mean_elevation_m`, `mean_slope_degrees` | SRTM 30 m, February 2000, static values repeated across months. A different source from the built-in Copernicus DEM. |
| `forest_fraction`, `shrubland_fraction`, `savanna_fraction`, `grassland_fraction`, `wetland_fraction`, `cropland_or_mosaic_fraction`, `urban_fraction`, `bare_land_fraction`, `landcover_water_fraction` | MODIS MCD12Q1.061 annual IGBP classes at 500 m, repeated monthly. Fractions refer to classified pixels; cropland includes mixed cropland/natural-vegetation mosaics. This is a different classification from ESA CCI land cover. |
| `seasonal_water_detected_fraction`, `permanent_water_detected_fraction`, `water_classified_fraction` | JRC annual water detections as fractions of the entire district grid, 30 m, repeated monthly. Coverage separately reports the classified share, which can be tiny. Unclassified pixels contribute no detections; this is not evidence of dry land. No classified pixels means blank water fractions. Original history through 2015, corrected v1.5 for 2016-2021, and v1.5 Landsat Collection 2 for 2022-2023. Version changes may introduce artefacts. |
| `nino34_anomaly` | NOAA CPC ERSSTv5 monthly Niño 3.4 anomaly against 1991-2020, in Celsius. The same global index for every district. |
| `district_area_km2`, `population_density_km2` | Equal-area polygon area including inland water; original WorldPop population divided by that area. |

The expanded bundle contains the CSV, original boundaries and case flags, updated
field definitions, `covariate_quality_flags.csv`, per-column coverage and ranges,
and provenance including the original manifest and new response hashes. Missing
values stay blank and receive an explicit quality flag. Vegetation and water
values with less than full classified/good-quality coverage receive a separate
`partial_spatial_coverage` flag; their coverage columns show how much was observed.
This flags every partial observation without imposing an arbitrary acceptance threshold.
The optional DHIS2 importer still imports only its
original five variables; enrichment does not add DHIS2 metadata or exports.

Use these as candidate covariates for retrospective research. Annual maps and
same-month weather may not have been available when a forecast would have been
issued. Select variables and lags within training folds. SPI/SPEI, seasonal
forecast hindcasts and transmission-suitability indices are not included: each
needs a separate, validated definition and reference period.

Sources and attribution: [ERA5-Land monthly aggregates](https://developers.google.com/earth-engine/datasets/catalog/ECMWF_ERA5_LAND_MONTHLY_AGGR),
[MOD13A3.061](https://doi.org/10.5067/MODIS/MOD13A3.061),
[MCD12Q1.061](https://doi.org/10.5067/MODIS/MCD12Q1.061),
[SRTM](https://doi.org/10.1029/2005RG000183),
[JRC Global Surface Water and corrected history](https://global-surface-water.appspot.com/download)
(Pekel et al., 2016, *High-resolution mapping of global surface water and its long-term changes*,
[doi:10.1038/nature20584](https://doi.org/10.1038/nature20584)), and
[NOAA CPC ERSSTv5](https://www.cpc.ncep.noaa.gov/data/indices/ersst5.nino.mth.91-20.ascii).
Retain these references and the base bundle's source-specific licences.

## Readiness

The case-map plugin runs without climate credentials and serves the published
2005-2023 research snapshot. A successful refresh checks the source release and
checksums; it does not establish current surveillance coverage or reporting quality.
The build now uses hourly-derived humidity. A complete rebuild requires both an
Earth Engine service account and CDS access with the ERA5-Land licence accepted.
An older CSV made with monthly-derived humidity is a different dataset revision.

Only source code, templates, the district crosswalk and tests belong in Git.
Keep credentials, downloaded rasters, instance configuration and generated bundles
outside the commit. The map is suitable for a local demo; this example does not
configure a public deployment.

## Run

From the GeoService repository root:

```bash
uv run --all-extras --with earthengine-api==1.7.49 python examples/malawi_chap/build.py
```

The service account must belong to a Google project authorized for Earth Engine.
Keep its JSON key outside the checkout, with permissions `600`. The default is
`~/.config/open-climate-service/dhis-google-auth.json`; alternatively set
`GOOGLE_APPLICATION_CREDENTIALS` or pass `--credentials /absolute/path/key.json`.
Set `CDS_API_KEY` for the built-in hourly-derived humidity source. You can load it from
an ignored file with `--env-file /absolute/path/.env`. Accept the CC-BY licence on the
[CDS ERA5-Land download page](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land?tab=download)
using the account that owns the key. Google Earth Engine supplies rainfall and
temperature; Copernicus CDS supplies hourly temperature and dewpoint for humidity.

The script starts a separate GeoService process on `127.0.0.1:9001`, using a Malawi
extent and its own storage, and stops that process on completion. An occupied
port is refused; choose another with `--port`. The existing instance on port
9000 and its configuration are not used.

All downloaded inputs, GeoService stores, intermediate exports and final outputs are
under `data/malawi-chap/` (gitignored). Use `--work-dir` to change that location.
The prepared instance configuration remains there for later inspection.
Existing DHIS2 connection and export mappings are preserved when rebuilding.
The launcher imports xclim before starting the server to avoid a circular-import
race between background process discovery and the first data-source request.

Cached source rasters and completed GeoService ingestions are reused on reruns. To
recompute after changing a plugin or accepting a newer source, use a new work
directory; do not mix a changed conversion with old GeoService stores.

## Processing

1. Download checksum-pinned case data and geoBoundaries geometry. Match the
   authors' numeric `DISTRICT` codes to names using their own `ID_1` lookup,
   then match those names to geoBoundaries. The checked-in crosswalk records
   that mapping. Coordinates in the case CSV are not reliable join keys.
2. Register the 27 case-bearing districts through GeoService's feature provider API.
3. Ingest ERA5-Land through an example instance plugin. Earth Engine exports
   three bands per month in annual batches on the native 0.1-degree grid.
   GeoService stores monthly rainfall in mm and temperature in Celsius. Rainfall is already a monthly accumulation: multiply metres by
   1000, **not by the number of days again**.
4. Ingest the built-in `era5land_relative_humidity_daily` data source from Copernicus CDS.
   GeoService computes humidity from hourly temperature and dewpoint with earthkit-meteo,
   then averages it to UTC days. Export overlap-weighted daily district means and require
   all 6,939 days for each district before averaging them into calendar-month means.
5. Ingest the ticket's annual 1 km WorldPop rasters. The 2005-2020 UN-adjusted
   series and the 2021-2023 Global2 R2025A series have separate stores so their
   grids are never silently resampled together.
6. Execute GeoService's `aggregate_to_chap_csv` workflow for each store. Climate uses
   polygon-overlap-weighted means; population uses overlap-weighted sums.
   Repeat the annual population estimate for each month of that year.
7. Join by district and month, normalize GeoService's `YYYYMM` periods to `YYYY-MM`,
   and refuse incomplete, duplicated, non-finite or implausible outputs.

GeoService handles the raster ingestion, storage, spatial aggregation and initial Chap
exports. The script handles the health table, crosswalk, final join and metadata.
The Malawi-specific providers are instance plugins under `examples/`; humidity uses
the existing built-in data source.

## Outputs

In `data/malawi-chap/output/`:

- `chap_MWI_admin2_monthly.csv`: 6,156 rows, 27 districts, January 2005 to December
  2023. Columns: `time_period`, `location`, `disease_cases`, `population`,
  `location_name`, `rainfall`, `mean_temperature`, `mean_relative_humidity`.
- `chap_MWI_admin2_monthly.geojson`: the same 27 districts; `feature.id` matches
  CSV `location`, using geoBoundaries `shapeID` values from the pinned release.
- `chap_MWI_admin2_monthly_schema.json`: field definitions, units, sources,
  licenses and limitations; a data dictionary, not a JSON Schema validator.
- `case_quality_flags.csv`: source-adjustment and fractional-case flags, kept
  separate from model covariates.
- `validation_report.json` and `provenance.json`: coverage checks, value ranges,
  source/output hashes, GeoService job identifiers, the GeoService Git revision and version,
  dependency lock checksum, and build/plugin checksums.

The `aggregates/humidity_daily.csv` intermediate retains all daily district humidity values;
its checksum and source ID are recorded in provenance.

The three `chap_MWI_*` files are the intended data-repository deliverables.
Keep the quality/provenance files alongside them for review. Source rasters
and GeoService intermediates remain in the work directory, not the data repository.

## Import into DHIS2

The optional importer reads a completed bundle and writes to DHIS2 only with
`--commit`. Dataset generation above does not call it. Its API path was previously
exercised against DHIS2 2.42; validate against your target instance before committing.
It does not configure GeoService connections, pipelines or future exports.

Set `DHIS2_IMPORT_TOKEN` in the environment or the checkout's ignored `.env`.
The importer defaults to `http://localhost:8090`. It prepares reviewable JSON
under `data/malawi-chap/dhis2/` and validates metadata without committing:

```bash
uv run --all-extras python -m examples.malawi_chap.import_dhis2
```

To import and build historical analytics for maps and charts:

```bash
uv run --all-extras python -m examples.malawi_chap.import_dhis2 --commit --analytics
```

The importer creates a separate Malawi country root, three regions and 28
districts. Region and country outlines are dissolved from the district geometry
and simplified for display; district geometry remains unchanged. The fixed
district-to-region lookup was checked against geoBoundaries ADM1 release
`9469f09`. Region geometry from that separate source is not imported.

Only the 27 case-bearing districts are assigned to the benchmark datasets:

| Dataset | Period | Contents |
| --- | --- | --- |
| Malawi malaria - public benchmark 2005-2023 | Monthly | Processed case counts |
| Malawi climate - ERA5-Land monthly 2005-2023 | Monthly | Rainfall, temperature and derived RH |
| Malawi population - WorldPop annual 2005-2023 | Yearly | Population, stored once per district-year |

Metadata uses `MWI_CHAP_` codes and server-generated UIDs, persisted in the work
directory for repeat runs. Existing Lao organisation units and data are retained.
Malawi is added to the importing user's scopes without removing other countries;
DHIS2 2.42's API-token cache is refreshed after a scope change. All data-value
imports retain strict organisation-unit and data-element validation. The importer
dry-runs each dataset, commits it, then reads every value back and compares it.
Each rerun targets the same keys; it does not create duplicate series.

Descriptions, data-element source URLs, external resources and a dashboard explain
the sources and caveats. The full schema, file checksums and UID mapping are also
stored at `/api/dataStore/malawi-chap/provenance` on the target instance.

District/month is the canonical climate resolution. DHIS2 `AVERAGE` aggregation
does not area-weight district summaries. In particular, an annual rainfall view
shows average monthly rainfall, not total annual rainfall. For larger regions or
other periods, aggregate from the original GeoService rasters with the appropriate
spatial and temporal method. Cases use `SUM`; annual population sums across
districts and averages across years (`AVERAGE_SUM_ORG_UNIT`).

## Adding another source

The usual path is source -> GeoService raster store -> district/month aggregation ->
DHIS2 data element or Chap covariate. For an existing GeoService data source, ingest it
for Malawi, aggregate to the same district polygons, then add a documented data
element/export mapping. For an unsupported source, add an instance plugin using
the pattern in `plugins/rasters/malawi.py` and the repository's custom data source
guide. Record the original units, licence, temporal coverage and any transformations.

Choose the statistic to fit the variable: population counts are summed; pixel
temperature is averaged; rainfall rates require a temporal accumulation. New
health outcomes need their own data elements and matching geographic identifiers.
Finer spatial case predictions require finer observed case data; subdividing the
current district totals would add assumptions rather than observations.

## Interpretation and attribution

- Cases: [Elling et al., Zenodo 17161438](https://doi.org/10.5281/zenodo.17161438),
  CC-BY-4.0. This is processed surveillance data, including averages across
  overlapping sources and fractional counts. **2013-2014 were adjusted for
  underreporting**; evaluate those years separately in sensitivity analyses.
  See the [paper](https://www.nature.com/articles/s43856-026-01681-9).
- Crosswalk: `ID_1` and `NAME_1` from the authors'
  [shapefiles archive](https://github.com/Mte2112/climalmal-2025/blob/75774e57362ad4382f806bd1cdca53c6c63984e8/data/shapefiles.tar.gz)
  (`shapefiles/maldists/MWI_adm1.shp`).
  The authors' geometries are not redistributed.
- Boundaries: geoBoundaries gbOpen MWI ADM2, commit `9469f09`, boundary year
  2020, National Statistics Office of Malawi / OCHA ROSEA, CC BY 3.0 IGO.
  Likoma is excluded because no case series is present. Historical boundary
  equivalence has not been independently established.
- Population: WorldPop, University of Southampton, CC-BY-4.0. The 2021 change
  of product introduces a potential methodological discontinuity. Polygon
  sums need not equal national totals because source masks and borders differ.
- Climate: [ERA5-Land monthly aggregates in Earth Engine](https://developers.google.com/earth-engine/datasets/catalog/ECMWF_ERA5_LAND_MONTHLY_AGGR).
  Contains modified Copernicus Climate Change Service information (2005-2023),
  under the [Copernicus licence](https://apps.ecmwf.int/datasets/licences/copernicus/).
  Rainfall and temperature use these monthly aggregates. Humidity uses the built-in
  `era5land_relative_humidity_daily` source from the CDS hourly ARCO archive.
  Relative humidity is computed each hour before daily and monthly averaging;
  it remains a reanalysis-derived estimate, not a direct humidity observation.

Retain source-specific attribution and licenses when redistributing the bundle.
Do not label the fractional case counts or estimated population as integers,
round them silently, or add the source's outcome-derived case rate as a predictor.

## Explore the district case counts

The monthly malaria layer can be fetched independently of the benchmark CSV and climate
credentials. Install the example's `plugins/vectors/case_series.py`, `case_series.yaml`
and `district_crosswalk.csv` together in an instance's vectors plugin directory (copy the
contents of the crosswalk when moving its symlink). Restart the service, open **Data sources →
Malawi monthly malaria cases**, and fetch with publication enabled. The example's instance
setup copies these files. A refresh retrieves the Zenodo latest-version record, the pinned
case CSV and the pinned boundaries. It refuses unexpected source revisions, invalid counts
and incomplete district-time coverage. A failed check leaves the last published layer available.

In this checkout, open `/map?dataset=mwi_malaria_monthly`, choose a month and click a district.
The colour intervals are fixed across months. Likoma is present as a polygon with no values.
The last successful refresh dates the source check; the final observation is December 2023.
Use the source page to recheck before a presentation. The separate Chap build uses the same
district matching and case validation to join cases with population and climate.
