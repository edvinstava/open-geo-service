"""Publish the harmonized district boundaries through the GeoService feature API."""

import json
from pathlib import Path
from typing import Any

from open_geo_service.features.providers import feature_provider


@feature_provider("malawi_chap_boundaries")
def boundaries(path: str) -> tuple[dict[str, Any], str]:
    """Read the prepared geoBoundaries subset, with the pinned release identity."""
    return json.loads(Path(path).read_text()), "9469f09"
