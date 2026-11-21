"""Rasterize lease areas and survey swaths, then find the gaps.

The whole analysis is a raster problem once the geometry is projected:

1. Lay a grid over the lease area at the resolution the answer is needed at.
2. Mark cells inside the lease polygon.
3. Mark cells within half a swath width of any survey track.
4. The gap is what is left, split into connected components.

Raster rather than exact polygon clipping because the question being asked is
"how many square kilometres are missing and where", not "what is the exact
boundary of the uncovered region". A 100 m grid over a 300 square kilometre
lease is 30,000 cells, which is instant, and the area error is bounded by the
cell size along the gap's perimeter.
"""

from __future__ import annotations

import numpy as np

from .geo import Projection, points_in_polygon, segment_distance
from .model import CoverageGrid, Gap, LeaseArea, SurveyLine


def build_grid(lease: LeaseArea, cell_m: float, projection: Projection | None = None) -> CoverageGrid:
    """Grid covering the lease's bounding box, with `inside` marked."""
    proj = projection or Projection.centered_on(lease.lons, lease.lats)
    ring = lease.projected(proj)
    xmin, ymin = ring.min(axis=0)
    xmax, ymax = ring.max(axis=0)
    cols = max(1, int(np.ceil((xmax - xmin) / cell_m)))
    rows = max(1, int(np.ceil((ymax - ymin) / cell_m)))
    xs = xmin + (np.arange(cols) + 0.5) * cell_m
    ys = ymin + (np.arange(rows) + 0.5) * cell_m
    gx, gy = np.meshgrid(xs, ys)
    pts = np.column_stack([gx.ravel(), gy.ravel()])
    inside = points_in_polygon(pts, ring).reshape(rows, cols)
    return CoverageGrid(
        projection=proj,
        cell_m=cell_m,
        origin=(float(xmin), float(ymin)),
        inside=inside,
        covered=np.zeros((rows, cols), dtype=bool),
    )


def apply_coverage(
    grid: CoverageGrid,
    lines: list[SurveyLine],
    sensors: set[str] | None = None,
    min_year: int | None = None,
) -> CoverageGrid:
    """Mark cells within half a swath of any qualifying survey line.

    `sensors` and `min_year` are how a real question gets asked: "which parts
    of the lease have no magnetometer coverage since 2020" is a different map
    from "which parts have never been surveyed at all", and the difference is
    what decides whether a vessel sails.
    """
    xs, ys = grid.cell_centers()
    gx, gy = np.meshgrid(xs, ys)
    pts = np.column_stack([gx.ravel(), gy.ravel()])
    covered = grid.covered.ravel().copy()

    for line in lines:
        if sensors is not None and line.sensor not in sensors:
            continue
        if min_year is not None and (line.year is None or line.year < min_year):
            continue
        xy = line.projected(grid.projection)
        half = line.swath_m / 2.0
        for a, b in zip(xy[:-1], xy[1:], strict=True):
            # Only test cells near this segment's bounding box.
            lo = np.minimum(a, b) - half
            hi = np.maximum(a, b) + half
            near = (pts[:, 0] >= lo[0]) & (pts[:, 0] <= hi[0]) & (pts[:, 1] >= lo[1]) & (pts[:, 1] <= hi[1])
            if not near.any():
                continue
            idx = np.flatnonzero(near)
            covered[idx] |= segment_distance(pts[idx], a, b) <= half

    grid.covered = covered.reshape(grid.shape)
    return grid


def connected_components(mask: np.ndarray) -> list[np.ndarray]:
    """Label 4-connected components of a boolean mask; returns index arrays.

    Written out rather than pulled from scipy to keep the dependency list to
    numpy: an iterative flood fill over a grid this size costs milliseconds.
    """
    rows, cols = mask.shape
    labels = np.zeros((rows, cols), dtype=np.int32)
    out: list[np.ndarray] = []
    current = 0
    for start in np.argwhere(mask):
        r0, c0 = int(start[0]), int(start[1])
        if labels[r0, c0]:
            continue
        current += 1
        stack = [(r0, c0)]
        labels[r0, c0] = current
        members = []
        while stack:
            r, c = stack.pop()
            members.append((r, c))
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < rows and 0 <= cc < cols and mask[rr, cc] and not labels[rr, cc]:
                    labels[rr, cc] = current
                    stack.append((rr, cc))
        out.append(np.array(members))
    return out


def find_gaps(grid: CoverageGrid, lease_name: str, min_area_km2: float = 0.1) -> list[Gap]:
    """Connected uncovered regions inside the lease, largest first.

    `min_area_km2` drops slivers along the lease boundary, which are an
    artifact of rasterizing a polygon edge rather than a real hole in the
    data. Without it the output is dominated by one-cell gaps nobody will
    send a vessel to.
    """
    mask = grid.inside & ~grid.covered
    xs, ys = grid.cell_centers()
    gaps: list[Gap] = []
    for members in connected_components(mask):
        rr, cc = members[:, 0], members[:, 1]
        area = len(members) * grid.cell_area_km2
        if area < min_area_km2:
            continue
        x = xs[cc]
        y = ys[rr]
        half = grid.cell_m / 2.0
        lon, lat = grid.projection.inverse(np.array([x.mean()]), np.array([y.mean()]))
        gaps.append(
            Gap(
                lease=lease_name,
                area_km2=area,
                bbox=(
                    float(x.min() - half),
                    float(y.min() - half),
                    float(x.max() + half),
                    float(y.max() + half),
                ),
                centroid=(float(lon[0]), float(lat[0])),
                cells=(rr, cc),
            )
        )
    return sorted(gaps, key=lambda g: -g.area_km2)


def analyze(
    lease: LeaseArea,
    lines: list[SurveyLine],
    cell_m: float = 100.0,
    sensors: set[str] | None = None,
    min_year: int | None = None,
    min_gap_km2: float = 0.1,
    projection: Projection | None = None,
) -> tuple[CoverageGrid, list[Gap]]:
    """Analyze one lease.

    Pass the same `projection` for every lease in a study: grids carry
    coordinates in projected metres, so two grids built on different local
    projections cannot be drawn on one map or compared to each other.
    """
    grid = build_grid(lease, cell_m, projection)
    apply_coverage(grid, lines, sensors, min_year)
    return grid, find_gaps(grid, lease.name, min_gap_km2)


def shared_projection(leases: list[LeaseArea]) -> Projection:
    """One projection centred on every lease in the study."""
    lons = np.concatenate([lease.lons for lease in leases])
    lats = np.concatenate([lease.lats for lease in leases])
    return Projection.centered_on(lons, lats)
