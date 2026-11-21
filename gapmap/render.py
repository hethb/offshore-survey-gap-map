"""A self-contained HTML map: inline SVG, no tiles, no network.

Survey planning gets reviewed by people who will not install a GIS, and often
on a vessel with no connectivity. The output is therefore one file with the
geometry drawn as SVG paths, which opens anywhere and prints.
"""

from __future__ import annotations

import html

import numpy as np

from .model import CoverageGrid, Gap, LeaseArea, SurveyLine, SurveyPlan, Vessel

PALETTE = {
    "sea": "#0d1b2a",
    "lease": "#5c7a99",
    "lease_fill": "#16293d",
    "covered": "#2d6a4f",
    "gap": "#e07a5f",
    "track": "#8aa6c1",
    "plan": "#f2cc8f",
    "ink": "#e6edf3",
    "muted": "#8d9aa8",
}


def _svg_path(points: np.ndarray, close: bool = False) -> str:
    cmds = [f"M{points[0, 0]:.1f},{points[0, 1]:.1f}"]
    cmds += [f"L{x:.1f},{y:.1f}" for x, y in points[1:]]
    if close:
        cmds.append("Z")
    return "".join(cmds)


class _View:
    """Maps projected metres to SVG pixels, y flipped so north is up."""

    def __init__(self, bounds: tuple[float, float, float, float], width: int, height: int, pad: int = 28):
        xmin, ymin, xmax, ymax = bounds
        self.scale = min(
            (width - 2 * pad) / max(xmax - xmin, 1.0), (height - 2 * pad) / max(ymax - ymin, 1.0)
        )
        self.xmin, self.ymax = xmin, ymax
        self.pad = pad
        self.width, self.height = width, height
        self.dx = (width - (xmax - xmin) * self.scale) / 2
        self.dy = (height - (ymax - ymin) * self.scale) / 2

    def __call__(self, xy: np.ndarray) -> np.ndarray:
        xy = np.atleast_2d(xy)
        px = (xy[:, 0] - self.xmin) * self.scale + self.dx
        py = (self.ymax - xy[:, 1]) * self.scale + self.dy
        return np.column_stack([px, py])


def render_map(
    leases: list[LeaseArea],
    lines: list[SurveyLine],
    grids: dict[str, CoverageGrid],
    gaps: list[Gap],
    plans: list[SurveyPlan],
    vessel: Vessel,
    totals: dict[str, float],
    width: int = 980,
    height: int = 760,
    title: str = "Survey coverage and gaps",
) -> str:
    projection = next(iter(grids.values())).projection
    all_xy = np.vstack([lease.projected(projection) for lease in leases])
    bounds = (all_xy[:, 0].min(), all_xy[:, 1].min(), all_xy[:, 0].max(), all_xy[:, 1].max())
    view = _View(bounds, width, height)

    parts: list[str] = [f'<rect width="{width}" height="{height}" fill="{PALETTE["sea"]}"/>']

    # Covered cells, drawn as one path per grid so the file stays small.
    for name, grid in grids.items():
        del name
        xs, ys = grid.cell_centers()
        rr, cc = np.nonzero(grid.inside & grid.covered)
        if not len(rr):
            continue
        half = grid.cell_m / 2
        d = []
        for r, c in zip(rr, cc, strict=True):
            p0 = view(np.array([xs[c] - half, ys[r] + half]))[0]
            p1 = view(np.array([xs[c] + half, ys[r] - half]))[0]
            d.append(f"M{p0[0]:.1f},{p0[1]:.1f}h{p1[0] - p0[0]:.1f}v{p1[1] - p0[1]:.1f}h{p0[0] - p1[0]:.1f}Z")
        parts.append(f'<path d="{"".join(d)}" fill="{PALETTE["covered"]}" opacity="0.75"/>')

    # Gaps.
    for grid in grids.values():
        xs, ys = grid.cell_centers()
        mask = grid.inside & ~grid.covered
        rr, cc = np.nonzero(mask)
        if not len(rr):
            continue
        half = grid.cell_m / 2
        d = []
        for r, c in zip(rr, cc, strict=True):
            p0 = view(np.array([xs[c] - half, ys[r] + half]))[0]
            p1 = view(np.array([xs[c] + half, ys[r] - half]))[0]
            d.append(f"M{p0[0]:.1f},{p0[1]:.1f}h{p1[0] - p0[0]:.1f}v{p1[1] - p0[1]:.1f}h{p0[0] - p1[0]:.1f}Z")
        parts.append(f'<path d="{"".join(d)}" fill="{PALETTE["gap"]}" opacity="0.85"/>')

    # Survey tracks.
    for line in lines:
        pts = view(line.projected(projection))
        parts.append(
            f'<path d="{_svg_path(pts)}" fill="none" stroke="{PALETTE["track"]}" stroke-width="0.4" opacity="0.5"/>'
        )

    # Planned lines.
    for plan in plans:
        for a, b in plan.lines:
            pts = view(np.vstack([a, b]))
            parts.append(
                f'<path d="{_svg_path(pts)}" fill="none" stroke="{PALETTE["plan"]}" stroke-width="0.9" opacity="0.95"/>'
            )

    # Lease outlines and labels.
    for lease in leases:
        pts = view(lease.projected(projection))
        parts.append(
            f'<path d="{_svg_path(pts, close=True)}" fill="none" stroke="{PALETTE["lease"]}" stroke-width="1.6"/>'
        )
        c = pts.mean(axis=0)
        top = pts[:, 1].min()
        parts.append(
            f'<text x="{c[0]:.0f}" y="{top - 8:.0f}" fill="{PALETTE["ink"]}" font-size="12" text-anchor="middle">{html.escape(lease.name)}</text>'
        )

    # Scale bar.
    grid0 = next(iter(grids.values()))
    del grid0
    bar_m = 5000.0
    bar_px = bar_m * view.scale
    x0, y0 = 24, height - 26
    parts.append(
        f'<line x1="{x0}" y1="{y0}" x2="{x0 + bar_px:.1f}" y2="{y0}" stroke="{PALETTE["ink"]}" stroke-width="2"/>'
    )
    parts.append(f'<text x="{x0}" y="{y0 - 6}" fill="{PALETTE["muted"]}" font-size="11">5 km</text>')

    svg = f'<svg viewBox="0 0 {width} {height}" width="100%" role="img" aria-label="{html.escape(title)}">{"".join(parts)}</svg>'

    rows = "".join(
        f"<tr><td>{html.escape(p.gap.lease)}</td><td>{p.gap.area_km2:.1f}</td><td>{p.heading_deg:.0f}&deg;</td>"
        f"<td>{len(p.lines)}</td><td>{p.line_km:.0f}</td><td>{p.turn_km:.0f}</td><td>{p.hours:.1f}</td></tr>"
        for p in plans[:12]
    )
    coverage_rows = "".join(
        f"<tr><td>{html.escape(name)}</td><td>{g.area_km2:.0f}</td><td>{g.coverage_fraction * 100:.1f}%</td><td>{g.gap_km2:.1f}</td></tr>"
        for name, g in grids.items()
    )

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>
body{{margin:0;background:#0b1118;color:{PALETTE["ink"]};font:14px/1.5 -apple-system,"Segoe UI",Inter,sans-serif}}
.wrap{{max-width:1040px;margin:0 auto;padding:24px}}
h1{{font-size:20px;margin:0 0 4px}}
h2{{font-size:12px;letter-spacing:.07em;text-transform:uppercase;color:{PALETTE["muted"]};margin:22px 0 8px}}
.muted{{color:{PALETTE["muted"]}}}
.kpis{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:16px 0}}
.kpi{{background:#121c27;border:1px solid #1e2b38;border-radius:10px;padding:12px 14px}}
.kpi b{{display:block;font-size:22px}}.kpi span{{color:{PALETTE["muted"]};font-size:12px}}
.map{{background:{PALETTE["sea"]};border:1px solid #1e2b38;border-radius:10px;overflow:hidden}}
.legend{{display:flex;gap:18px;flex-wrap:wrap;color:{PALETTE["muted"]};font-size:12px;margin-top:10px}}
.sw{{display:inline-block;width:11px;height:11px;border-radius:2px;margin-right:6px;vertical-align:-1px}}
table{{width:100%;border-collapse:collapse;font-size:13px}}
td,th{{text-align:left;padding:6px 8px;border-top:1px solid #1e2b38}}
th{{color:{PALETTE["muted"]};font-size:12px;border-top:none;font-weight:600}}
</style></head><body><div class="wrap">
<h1>{html.escape(title)}</h1>
<p class="muted">{len(leases)} lease areas, {len(lines)} survey lines, {len(gaps)} gaps above the reporting threshold.
Planned with a {vessel.name} at {vessel.speed_kn:g} kn, {vessel.swath_m:g} m swath, {vessel.overlap * 100:.0f}% overlap.</p>
<div class="kpis">
<div class="kpi"><b>{totals["gap_km2"]:.0f} km&sup2;</b><span>unsurveyed inside the leases</span></div>
<div class="kpi"><b>{totals["line_km"]:.0f} km</b><span>of survey line to close it</span></div>
<div class="kpi"><b>{totals["survey_days"]:.1f}</b><span>survey days at {vessel.hours_per_day:g} h/day</span></div>
<div class="kpi"><b>${totals["cost_usd"] / 1e6:.2f}M</b><span>at ${vessel.day_rate_usd / 1000:.0f}k/day</span></div>
</div>
<div class="map">{svg}</div>
<div class="legend">
<span><span class="sw" style="background:{PALETTE["covered"]}"></span>Surveyed</span>
<span><span class="sw" style="background:{PALETTE["gap"]}"></span>Gap</span>
<span><span class="sw" style="background:{PALETTE["track"]}"></span>Existing track lines</span>
<span><span class="sw" style="background:{PALETTE["plan"]}"></span>Planned lines</span>
<span><span class="sw" style="background:{PALETTE["lease"]}"></span>Lease boundary</span>
</div>
<h2>Coverage by lease</h2>
<table><tr><th>Lease</th><th>Area (km&sup2;)</th><th>Covered</th><th>Gap (km&sup2;)</th></tr>{coverage_rows}</table>
<h2>Survey plans, best value first</h2>
<table><tr><th>Lease</th><th>Gap (km&sup2;)</th><th>Heading</th><th>Lines</th><th>Line km</th><th>Turn km</th><th>Hours</th></tr>{rows}</table>
</div></body></html>"""
