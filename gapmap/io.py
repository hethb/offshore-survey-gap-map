"""GeoJSON in and out, so real BOEM and NOAA exports drop straight in."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .model import Gap, LeaseArea, SurveyLine, SurveyPlan


def _ring(coords: list) -> np.ndarray:
    ring = np.array(coords, dtype=float)
    if len(ring) > 1 and np.allclose(ring[0], ring[-1]):
        ring = ring[:-1]
    return ring


def load_leases(path: str | Path) -> list[LeaseArea]:
    """Read Polygon or MultiPolygon features; the outer ring of each is used.

    Property names follow BOEM's lease shapefiles where they are known
    (`LEASE_NUMBER`, `COMPANY`), with generic fallbacks so any export works.
    """
    data = json.loads(Path(path).read_text())
    out: list[LeaseArea] = []
    for feature in data.get("features", []):
        props: dict[str, Any] = feature.get("properties", {}) or {}
        geom = feature.get("geometry") or {}
        name = str(
            props.get("LEASE_NUMBER") or props.get("lease") or props.get("name") or f"lease-{len(out) + 1}"
        )
        operator = props.get("COMPANY") or props.get("operator")
        if geom.get("type") == "Polygon":
            rings = [geom["coordinates"][0]]
        elif geom.get("type") == "MultiPolygon":
            rings = [poly[0] for poly in geom["coordinates"]]
        else:
            continue
        for i, ring in enumerate(rings):
            label = name if len(rings) == 1 else f"{name} ({i + 1})"
            out.append(LeaseArea(name=label, ring=_ring(ring), operator=operator))
    return out


def load_lines(path: str | Path, default_swath_m: float = 150.0) -> list[SurveyLine]:
    """Read LineString or MultiLineString features as survey tracks."""
    data = json.loads(Path(path).read_text())
    out: list[SurveyLine] = []
    for feature in data.get("features", []):
        props: dict[str, Any] = feature.get("properties", {}) or {}
        geom = feature.get("geometry") or {}
        swath = float(props.get("swath_m") or props.get("swath") or default_swath_m)
        survey = str(props.get("survey") or props.get("SURVEY_ID") or "unnamed")
        year = props.get("year") or props.get("YEAR")
        sensor = str(props.get("sensor") or "bathymetry")
        if geom.get("type") == "LineString":
            tracks = [geom["coordinates"]]
        elif geom.get("type") == "MultiLineString":
            tracks = geom["coordinates"]
        else:
            continue
        for track in tracks:
            out.append(
                SurveyLine(
                    track=np.array(track, dtype=float)[:, :2],
                    swath_m=swath,
                    survey=survey,
                    year=int(year) if year is not None else None,
                    sensor=sensor,
                )
            )
    return out


def leases_to_geojson(leases: list[LeaseArea]) -> dict:
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"lease": lease.name, "operator": lease.operator},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[*lease.ring.tolist(), lease.ring[0].tolist()]],
                },
            }
            for lease in leases
        ],
    }


def lines_to_geojson(lines: list[SurveyLine]) -> dict:
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "survey": line.survey,
                    "year": line.year,
                    "sensor": line.sensor,
                    "swath_m": line.swath_m,
                },
                "geometry": {"type": "LineString", "coordinates": line.track.tolist()},
            }
            for line in lines
        ],
    }


def gaps_to_geojson(gaps: list[Gap], projection) -> dict:
    """Gap bounding boxes as polygons, with their area as a property."""
    features = []
    for gap in gaps:
        xmin, ymin, xmax, ymax = gap.bbox
        xs = np.array([xmin, xmax, xmax, xmin, xmin])
        ys = np.array([ymin, ymin, ymax, ymax, ymin])
        lon, lat = projection.inverse(xs, ys)
        features.append(
            {
                "type": "Feature",
                "properties": {"lease": gap.lease, "area_km2": round(gap.area_km2, 3)},
                "geometry": {"type": "Polygon", "coordinates": [np.column_stack([lon, lat]).tolist()]},
            }
        )
    return {"type": "FeatureCollection", "features": features}


def plans_to_geojson(plans: list[SurveyPlan], projection) -> dict:
    features = []
    for plan in plans:
        for a, b in plan.lines:
            lon, lat = projection.inverse(np.array([a[0], b[0]]), np.array([a[1], b[1]]))
            features.append(
                {
                    "type": "Feature",
                    "properties": {
                        "lease": plan.gap.lease,
                        "heading_deg": round(plan.heading_deg, 1),
                        "spacing_m": plan.line_spacing_m,
                    },
                    "geometry": {"type": "LineString", "coordinates": np.column_stack([lon, lat]).tolist()},
                }
            )
    return {"type": "FeatureCollection", "features": features}


def write_json(path: str | Path, payload: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=1))
