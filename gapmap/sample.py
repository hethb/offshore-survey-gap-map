"""Sample lease areas and survey lines.

The real inputs are public: BOEM publishes wind energy lease area shapefiles,
and NOAA's NCEI holds bathymetric survey coverage. Neither is redistributed
here, and neither is reachable from a sandbox, so this module generates
geometry with the same shape as the real thing: rectangular-ish lease blocks
a few tens of kilometres across off the US east coast, surveyed by sets of
parallel track lines with realistic swath widths, including the two failure
modes that matter in practice - a survey that stopped early, and a survey
whose line spacing was wider than its swath, leaving stripes of nothing
between the lines.

`gapmap.io` reads GeoJSON, so pointing this at real data is a file path.
"""

from __future__ import annotations

import numpy as np

from .model import LeaseArea, SurveyLine


def _rect(
    lon0: float, lat0: float, width_km: float, height_km: float, rotation_deg: float = 0.0
) -> np.ndarray:
    """A rotated rectangle in lon/lat, sized in kilometres."""
    dlat = height_km / 111.32
    dlon = width_km / (111.32 * np.cos(np.radians(lat0)))
    corners = np.array(
        [[-dlon / 2, -dlat / 2], [dlon / 2, -dlat / 2], [dlon / 2, dlat / 2], [-dlon / 2, dlat / 2]]
    )
    t = np.radians(rotation_deg)
    rot = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    return (corners @ rot.T) + np.array([lon0, lat0])


def lease_areas() -> list[LeaseArea]:
    return [
        LeaseArea("OCS-A 0001", _rect(-74.20, 39.30, 22.0, 18.0, 8.0), operator="Example Wind LLC"),
        LeaseArea("OCS-A 0002", _rect(-73.95, 39.52, 16.0, 24.0, -5.0), operator="Example Wind LLC"),
        LeaseArea("OCS-A 0003", _rect(-74.45, 39.58, 19.0, 14.0, 20.0), operator="Northeast Offshore"),
    ]


def _parallel_lines(
    lease: LeaseArea,
    spacing_m: float,
    swath_m: float,
    heading_deg: float,
    survey: str,
    year: int,
    sensor: str = "bathymetry",
    coverage: float = 1.0,
    rng: np.random.Generator | None = None,
) -> list[SurveyLine]:
    """Track lines across a lease, optionally stopping after `coverage` of it."""
    rng = rng or np.random.default_rng(0)
    lat0 = float(lease.lats.mean())
    m_per_deg_lat = 111_320.0
    m_per_deg_lon = 111_320.0 * np.cos(np.radians(lat0))
    center = np.array([lease.lons.mean(), lease.lats.mean()])

    t = np.radians(heading_deg)
    along = np.array([np.sin(t) / m_per_deg_lon, np.cos(t) / m_per_deg_lat])
    across = np.array([np.cos(t) / m_per_deg_lon, -np.sin(t) / m_per_deg_lat])

    # Tracks run a little past the lease edge, as real ones do for turn room,
    # rather than across the whole map.
    half_width_m = float(np.ptp(lease.lons)) * m_per_deg_lon / 2 + 600.0
    half_height_m = float(np.ptp(lease.lats)) * m_per_deg_lat / 2 + 600.0
    extent_m = float(np.hypot(half_width_m, half_height_m))
    offsets = np.arange(-extent_m, extent_m, spacing_m)
    keep = offsets[: max(1, int(len(offsets) * coverage))]
    out: list[SurveyLine] = []
    for off in keep:
        jitter = rng.normal(0, spacing_m * 0.03)
        start = center + across * (off + jitter) - along * extent_m
        end = center + across * (off + jitter) + along * extent_m
        # A few waypoints so the track is a polyline, as real navigation data is.
        track = np.array([start + (end - start) * f for f in np.linspace(0, 1, 6)])
        out.append(SurveyLine(track=track, swath_m=swath_m, survey=survey, year=year, sensor=sensor))
    return out


def survey_lines() -> list[SurveyLine]:
    """Three surveys, each with a different problem."""
    leases = lease_areas()
    rng = np.random.default_rng(7)
    lines: list[SurveyLine] = []

    # Full modern coverage of the first lease.
    lines += _parallel_lines(
        leases[0], spacing_m=120.0, swath_m=150.0, heading_deg=8.0, survey="2023 MBES", year=2023, rng=rng
    )

    # The second lease was abandoned two thirds of the way through.
    lines += _parallel_lines(
        leases[1],
        spacing_m=120.0,
        swath_m=150.0,
        heading_deg=-5.0,
        survey="2022 MBES",
        year=2022,
        coverage=0.62,
        rng=rng,
    )

    # The third was run at 400 m spacing with a 150 m swath: stripes of nothing.
    lines += _parallel_lines(
        leases[2],
        spacing_m=400.0,
        swath_m=150.0,
        heading_deg=20.0,
        survey="2016 reconnaissance",
        year=2016,
        rng=rng,
    )

    # A magnetometer survey that only covers part of the first lease.
    lines += _parallel_lines(
        leases[0],
        spacing_m=200.0,
        swath_m=120.0,
        heading_deg=8.0,
        survey="2021 magnetometer",
        year=2021,
        sensor="magnetometer",
        coverage=0.45,
        rng=rng,
    )
    return lines
