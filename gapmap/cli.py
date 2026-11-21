"""Command line interface.

gapmap report  [--leases leases.geojson --lines tracks.geojson]
               [--cell 100] [--sensor bathymetry] [--since 2018]
               [--html out/map.html] [--geojson out/]
gapmap sample  --out data/          write the sample inputs as GeoJSON
"""

from __future__ import annotations

import argparse

from .coverage import analyze, shared_projection
from .io import (
    gaps_to_geojson,
    leases_to_geojson,
    lines_to_geojson,
    load_leases,
    load_lines,
    plans_to_geojson,
    write_json,
)
from .model import Vessel
from .plan import campaign_cost, plan_all, prioritize
from .render import render_map
from .sample import lease_areas, survey_lines


def cmd_report(a) -> None:
    leases = load_leases(a.leases) if a.leases else lease_areas()
    lines = load_lines(a.lines) if a.lines else survey_lines()
    vessel = Vessel(speed_kn=a.speed, swath_m=a.swath, overlap=a.overlap, day_rate_usd=a.day_rate)
    sensors = {a.sensor} if a.sensor else None

    # One projection for every lease, so grids share a coordinate system and
    # can be drawn on the same map.
    projection = shared_projection(leases)
    grids = {}
    gaps = []
    for lease in leases:
        grid, lease_gaps = analyze(
            lease,
            lines,
            cell_m=a.cell,
            sensors=sensors,
            min_year=a.since,
            min_gap_km2=a.min_gap,
            projection=projection,
        )
        grids[lease.name] = grid
        gaps.extend(lease_gaps)

    plans = prioritize(
        [plan for gap in gaps for plan in plan_all(grids[gap.lease], [gap], vessel, a.headings)]
    )
    totals = campaign_cost(plans, vessel)

    filt = []
    if a.sensor:
        filt.append(f"sensor={a.sensor}")
    if a.since:
        filt.append(f"since {a.since}")
    print(
        f"{len(leases)} lease(s), {len(lines)} survey line(s){', ' + ', '.join(filt) if filt else ''}, {a.cell:g} m grid\n"
    )
    print(f"{'lease':<14}{'area km2':>10}{'covered':>10}{'gap km2':>10}{'gaps':>6}")
    for lease in leases:
        g = grids[lease.name]
        n = sum(1 for gap in gaps if gap.lease == lease.name)
        print(
            f"{lease.name:<14}{g.area_km2:>10.0f}{g.coverage_fraction * 100:>9.1f}%{g.gap_km2:>10.1f}{n:>6}"
        )

    if plans:
        print(
            f"\n{'lease':<14}{'gap km2':>9}{'heading':>9}{'lines':>7}{'line km':>9}{'turn km':>9}{'hours':>7}{'km2/h':>8}"
        )
        for plan in plans[: a.top]:
            rate = plan.gap.area_km2 / plan.hours if plan.hours else 0.0
            print(
                f"{plan.gap.lease:<14}{plan.gap.area_km2:>9.1f}{plan.heading_deg:>8.0f}°{len(plan.lines):>7}"
                f"{plan.line_km:>9.0f}{plan.turn_km:>9.0f}{plan.hours:>7.1f}{rate:>8.2f}"
            )
    print(
        f"\ntotal: {totals['gap_km2']:.0f} km2 of gap, {totals['line_km']:.0f} km of line, "
        f"{totals['survey_hours']:.0f} survey hours = {totals['survey_days']:.1f} days = ${totals['cost_usd'] / 1e6:.2f}M"
    )

    if a.html:
        from pathlib import Path

        out = Path(a.html)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(render_map(leases, lines, grids, gaps, plans, vessel, totals))
        print(f"wrote {out}")
    if a.geojson:
        projection = next(iter(grids.values())).projection
        write_json(f"{a.geojson}/gaps.geojson", gaps_to_geojson(gaps, projection))
        write_json(f"{a.geojson}/planned-lines.geojson", plans_to_geojson(plans, projection))
        print(f"wrote {a.geojson}/gaps.geojson and {a.geojson}/planned-lines.geojson")


def cmd_sample(a) -> None:
    write_json(f"{a.out}/leases.geojson", leases_to_geojson(lease_areas()))
    write_json(f"{a.out}/surveys.geojson", lines_to_geojson(survey_lines()))
    print(f"wrote {a.out}/leases.geojson and {a.out}/surveys.geojson")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="gapmap", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("report")
    r.add_argument("--leases")
    r.add_argument("--lines")
    r.add_argument("--cell", type=float, default=100.0, help="grid cell size in metres")
    r.add_argument("--sensor", help="only count coverage from this sensor")
    r.add_argument("--since", type=int, help="only count coverage from this year onward")
    r.add_argument("--min-gap", type=float, default=0.25, help="ignore gaps smaller than this, km2")
    r.add_argument("--headings", type=int, default=12)
    r.add_argument("--top", type=int, default=10)
    r.add_argument("--speed", type=float, default=4.0, help="survey speed, knots")
    r.add_argument("--swath", type=float, default=150.0)
    r.add_argument("--overlap", type=float, default=0.2)
    r.add_argument("--day-rate", type=float, default=55_000.0)
    r.add_argument("--html")
    r.add_argument("--geojson")

    s = sub.add_parser("sample")
    s.add_argument("--out", default="data")

    a = ap.parse_args(argv)
    {"report": cmd_report, "sample": cmd_sample}[a.cmd](a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
