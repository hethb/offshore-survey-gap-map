"""Domain types: lease areas, survey lines, coverage grids and plans."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .geo import NM, Projection, polygon_area_m2


@dataclass
class LeaseArea:
    """One offshore lease area, as a polygon in longitude/latitude degrees."""

    name: str
    #: (n, 2) array of [lon, lat], not closed.
    ring: np.ndarray
    operator: str | None = None

    @property
    def lons(self) -> np.ndarray:
        return self.ring[:, 0]

    @property
    def lats(self) -> np.ndarray:
        return self.ring[:, 1]

    def projected(self, projection: Projection) -> np.ndarray:
        x, y = projection.forward(self.lons, self.lats)
        return np.column_stack([x, y])

    def area_km2(self, projection: Projection) -> float:
        return polygon_area_m2(self.projected(projection)) / 1e6


@dataclass
class SurveyLine:
    """A track the vessel or AUV ran, with the swath it imaged.

    `swath_m` is the total width covered, so coverage extends swath_m / 2
    either side of the track.
    """

    #: (n, 2) array of [lon, lat] waypoints.
    track: np.ndarray
    swath_m: float
    survey: str = "unnamed"
    year: int | None = None
    #: What the line measured: bathymetry, magnetometer, sub-bottom profiler.
    sensor: str = "bathymetry"

    def projected(self, projection: Projection) -> np.ndarray:
        x, y = projection.forward(self.track[:, 0], self.track[:, 1])
        return np.column_stack([x, y])

    def length_m(self, projection: Projection) -> float:
        xy = self.projected(projection)
        return float(np.sum(np.linalg.norm(np.diff(xy, axis=0), axis=1)))


@dataclass
class CoverageGrid:
    """A raster of an area in local metres: which cells are inside, and which are covered."""

    projection: Projection
    #: Cell size in metres.
    cell_m: float
    #: Lower-left corner in projected metres.
    origin: tuple[float, float]
    #: (rows, cols) boolean arrays.
    inside: np.ndarray
    covered: np.ndarray

    @property
    def shape(self) -> tuple[int, int]:
        return self.inside.shape

    @property
    def cell_area_km2(self) -> float:
        return self.cell_m**2 / 1e6

    @property
    def area_km2(self) -> float:
        return float(self.inside.sum()) * self.cell_area_km2

    @property
    def covered_km2(self) -> float:
        return float((self.inside & self.covered).sum()) * self.cell_area_km2

    @property
    def gap_km2(self) -> float:
        return self.area_km2 - self.covered_km2

    @property
    def coverage_fraction(self) -> float:
        return self.covered_km2 / self.area_km2 if self.area_km2 else 0.0

    def cell_centers(self) -> tuple[np.ndarray, np.ndarray]:
        rows, cols = self.shape
        xs = self.origin[0] + (np.arange(cols) + 0.5) * self.cell_m
        ys = self.origin[1] + (np.arange(rows) + 0.5) * self.cell_m
        return xs, ys


@dataclass
class Gap:
    """A connected unsurveyed region inside a lease area."""

    lease: str
    area_km2: float
    #: Bounding box in projected metres: (xmin, ymin, xmax, ymax).
    bbox: tuple[float, float, float, float]
    #: Centroid in lon/lat.
    centroid: tuple[float, float]
    #: Cells belonging to this gap, as (row, col) index arrays.
    cells: tuple[np.ndarray, np.ndarray] = field(
        repr=False, default_factory=lambda: (np.array([]), np.array([]))
    )

    @property
    def width_km(self) -> float:
        return (self.bbox[2] - self.bbox[0]) / 1000.0

    @property
    def height_km(self) -> float:
        return (self.bbox[3] - self.bbox[1]) / 1000.0


@dataclass
class SurveyPlan:
    """Planned lawnmower lines over one gap, with the time they take."""

    gap: Gap
    #: Each line is a pair of endpoints in projected metres.
    lines: list[tuple[np.ndarray, np.ndarray]]
    line_spacing_m: float
    heading_deg: float
    line_km: float
    turn_km: float
    hours: float

    @property
    def total_km(self) -> float:
        return self.line_km + self.turn_km


@dataclass
class Vessel:
    """Survey platform parameters, in the units operators quote them in."""

    name: str = "AUV"
    speed_kn: float = 4.0
    swath_m: float = 150.0
    #: Fraction of swath overlapped with the neighbouring line.
    overlap: float = 0.2
    #: Minutes lost in each turn at the end of a line.
    turn_minutes: float = 6.0
    #: Hours of survey available per day, after weather and launch/recovery.
    hours_per_day: float = 16.0
    day_rate_usd: float = 55_000.0

    @property
    def speed_ms(self) -> float:
        return self.speed_kn * NM / 3600.0

    @property
    def line_spacing_m(self) -> float:
        return self.swath_m * (1.0 - self.overlap)
