"""Turn a gap into survey lines, and survey lines into hours and dollars.

A gap is only actionable with a number attached: how long it takes to close
and what that costs. The planner lays parallel lines (a lawnmower pattern)
across the gap at the spacing the sensor's swath allows, clips each line to
the cells that are actually missing, and adds the turn cost between lines.

The heading is chosen rather than assumed. Running lines along a gap's long
axis means fewer turns, and turns are where survey time is lost: a six-minute
turn on a 2 km line is a fifth of the line's own duration. The planner tries
headings and keeps the cheapest, which on a long thin gap reproduces the
obvious answer and on an awkward one does better than a fixed north-south
grid.
"""

from __future__ import annotations

import math

import numpy as np

from .model import CoverageGrid, Gap, SurveyPlan, Vessel


def _gap_mask(grid: CoverageGrid, gap: Gap) -> np.ndarray:
    mask = np.zeros(grid.shape, dtype=bool)
    mask[gap.cells[0], gap.cells[1]] = True
    return mask


def _cell_lookup(grid: CoverageGrid, mask: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """Is each projected point inside a masked cell?"""
    cols = np.floor((xy[:, 0] - grid.origin[0]) / grid.cell_m).astype(int)
    rows = np.floor((xy[:, 1] - grid.origin[1]) / grid.cell_m).astype(int)
    ok = (rows >= 0) & (rows < grid.shape[0]) & (cols >= 0) & (cols < grid.shape[1])
    out = np.zeros(len(xy), dtype=bool)
    out[ok] = mask[rows[ok], cols[ok]]
    return out


def plan_lines(
    grid: CoverageGrid, gap: Gap, vessel: Vessel, heading_deg: float, sample_m: float | None = None
) -> SurveyPlan:
    """Lawnmower lines at a fixed heading, clipped to the gap's own cells."""
    mask = _gap_mask(grid, gap)
    spacing = vessel.line_spacing_m
    step = sample_m or max(grid.cell_m / 2.0, 25.0)
    theta = math.radians(heading_deg)
    along = np.array([math.sin(theta), math.cos(theta)])  # compass heading
    across = np.array([along[1], -along[0]])

    xs, ys = grid.cell_centers()
    pts = np.column_stack([xs[gap.cells[1]], ys[gap.cells[0]]])
    center = pts.mean(axis=0)
    u = (pts - center) @ along
    v = (pts - center) @ across

    lines: list[tuple[np.ndarray, np.ndarray]] = []
    line_m = 0.0
    offsets = np.arange(v.min(), v.max() + spacing, spacing)
    for off in offsets:
        start = center + across * off + along * (u.min() - step)
        n = int((u.max() - u.min() + 2 * step) / step) + 1
        samples = start + np.outer(np.arange(n) * step, along)
        hit = _cell_lookup(grid, mask, samples)
        if not hit.any():
            continue
        # Split into runs so a line crossing a covered strip becomes two lines.
        # Runs of consecutive hits become separate lines, so a line that
        # crosses an already-covered strip is not planned across it.
        edges = np.flatnonzero(np.diff(hit.astype(np.int8)))
        bounds = np.concatenate(
            [
                np.array([0] if hit[0] else [], dtype=int),
                edges + 1,
                np.array([len(hit)] if hit[-1] else [], dtype=int),
            ]
        ).astype(int)
        for a, b in zip(bounds[::2], bounds[1::2], strict=False):
            p0, p1 = samples[int(a)], samples[min(int(b), len(samples) - 1)]
            d = float(np.linalg.norm(p1 - p0))
            if d < grid.cell_m:
                continue
            lines.append((p0, p1))
            line_m += d

    turns = max(0, len(lines) - 1)
    turn_m = turns * vessel.turn_minutes * 60.0 * vessel.speed_ms
    hours = (line_m + turn_m) / vessel.speed_ms / 3600.0 if vessel.speed_ms else 0.0
    return SurveyPlan(
        gap=gap,
        lines=lines,
        line_spacing_m=spacing,
        heading_deg=heading_deg % 180.0,
        line_km=line_m / 1000.0,
        turn_km=turn_m / 1000.0,
        hours=hours,
    )


def best_plan(grid: CoverageGrid, gap: Gap, vessel: Vessel, headings: int = 12) -> SurveyPlan:
    """Try evenly spaced headings and keep the fastest plan.

    Headings are only tried over 180 degrees because a lawnmower pattern is
    symmetric: running the lines north-to-south costs the same as
    south-to-north.
    """
    candidates = [plan_lines(grid, gap, vessel, 180.0 * i / headings) for i in range(headings)]
    return min((p for p in candidates if p.lines), key=lambda p: p.hours, default=candidates[0])


def plan_all(grid: CoverageGrid, gaps: list[Gap], vessel: Vessel, headings: int = 12) -> list[SurveyPlan]:
    return [best_plan(grid, gap, vessel, headings) for gap in gaps]


def campaign_cost(plans: list[SurveyPlan], vessel: Vessel) -> dict[str, float]:
    """Hours, days and dollars for a set of plans, at the vessel's day rate."""
    hours = sum(p.hours for p in plans)
    days = hours / vessel.hours_per_day if vessel.hours_per_day else 0.0
    return {
        "survey_hours": hours,
        "survey_days": days,
        "cost_usd": days * vessel.day_rate_usd,
        "line_km": sum(p.line_km for p in plans),
        "turn_km": sum(p.turn_km for p in plans),
        "gap_km2": sum(p.gap.area_km2 for p in plans),
    }


def prioritize(plans: list[SurveyPlan]) -> list[SurveyPlan]:
    """Order by square kilometres closed per survey hour, best first.

    This is the number that decides what a vessel does first. A large gap that
    happens to be long and thin can be worse value than a smaller compact one,
    because the thin gap is mostly turns.
    """
    return sorted(plans, key=lambda p: -(p.gap.area_km2 / p.hours if p.hours else 0.0))
