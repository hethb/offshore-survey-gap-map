"""Geometry on a sphere, without a geospatial stack.

Offshore lease areas are a few tens of kilometres across, so the accurate and
simple thing to do is project to a local plane once, in metres, and do every
subsequent computation there.

The projection is sinusoidal about the study's own centre: the east-west
scale uses each point's own latitude rather than the reference latitude. That
makes it equal-area, which matters because area is what this tool reports.

Distortion is real and measured rather than assumed. Against haversine
distance at 39.5 degrees north, the error is under 0.07% for point pairs
within about 30 km of the reference (a single lease) and under 0.15% out to
70 km (a group of leases). On a 20 km lease edge that is under 14 m, well
below the 100 m grid everything is computed on, and the test
`test_projection_distance_matches_haversine` pins both bounds so a change to
the projection cannot quietly make them worse.

Everything downstream (coverage, gaps, line planning, cost) is plane geometry
in metres, which is both easier to reason about and easier to test.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

#: WGS-84 semi-major axis, metres.
EARTH_R = 6_378_137.0
#: Metres per nautical mile.
NM = 1852.0


@dataclass(frozen=True)
class Projection:
    """Sinusoidal projection about a reference point, in metres."""

    lon0: float
    lat0: float

    @classmethod
    def centered_on(cls, lons: np.ndarray, lats: np.ndarray) -> Projection:
        return cls(float(np.mean(lons)), float(np.mean(lats)))

    def forward(self, lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        lat = np.asarray(lat, dtype=float)
        x = np.radians(np.asarray(lon, dtype=float) - self.lon0) * EARTH_R * np.cos(np.radians(lat))
        y = np.radians(lat - self.lat0) * EARTH_R
        return x, y

    def inverse(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # Latitude comes from y alone, so the inverse is exact rather than iterative.
        lat = np.degrees(np.asarray(y, dtype=float) / EARTH_R) + self.lat0
        lon = np.degrees(np.asarray(x, dtype=float) / (EARTH_R * np.cos(np.radians(lat)))) + self.lon0
        return lon, lat


def haversine_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Great-circle distance in metres, used to check the projection."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_R * math.asin(math.sqrt(a))


def polygon_area_m2(xy: np.ndarray) -> float:
    """Shoelace area of a closed or open ring, always positive."""
    x, y = xy[:, 0], xy[:, 1]
    return abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))) / 2.0


def points_in_polygon(points: np.ndarray, ring: np.ndarray) -> np.ndarray:
    """Vectorized even-odd ray casting. points: (n, 2), ring: (m, 2)."""
    x, y = points[:, 0], points[:, 1]
    xi, yi = ring[:, 0], ring[:, 1]
    xj, yj = np.roll(xi, -1), np.roll(yi, -1)
    inside = np.zeros(len(points), dtype=bool)
    for a in range(len(ring)):
        x1, y1, x2, y2 = xi[a], yi[a], xj[a], yj[a]
        if y1 == y2:
            continue
        crosses = (y1 > y) != (y2 > y)
        with np.errstate(divide="ignore", invalid="ignore"):
            x_at = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
        inside ^= crosses & (x < x_at)
    return inside


def segment_distance(points: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Distance from each point to the segment ab."""
    ab = b - a
    denom = float(ab @ ab)
    if denom == 0.0:
        return np.linalg.norm(points - a, axis=1)
    t = np.clip(((points - a) @ ab) / denom, 0.0, 1.0)
    closest = a + t[:, None] * ab
    return np.linalg.norm(points - closest, axis=1)


def densify(ring: np.ndarray, max_step_m: float) -> np.ndarray:
    """Insert vertices so no edge is longer than max_step_m, for smooth outlines."""
    out = []
    n = len(ring)
    for i in range(n):
        a, b = ring[i], ring[(i + 1) % n]
        out.append(a)
        d = float(np.linalg.norm(b - a))
        for k in range(1, int(d // max_step_m)):
            out.append(a + (b - a) * (k * max_step_m / d))
    return np.array(out)
