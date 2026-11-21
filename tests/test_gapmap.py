import json
import math

import numpy as np
import pytest

from gapmap.cli import main
from gapmap.coverage import (
    analyze,
    apply_coverage,
    build_grid,
    connected_components,
    find_gaps,
    shared_projection,
)
from gapmap.geo import (
    NM,
    Projection,
    densify,
    haversine_m,
    points_in_polygon,
    polygon_area_m2,
    segment_distance,
)
from gapmap.io import gaps_to_geojson, leases_to_geojson, lines_to_geojson, load_leases, load_lines
from gapmap.model import LeaseArea, SurveyLine, Vessel
from gapmap.plan import best_plan, campaign_cost, plan_lines, prioritize
from gapmap.render import render_map
from gapmap.sample import lease_areas, survey_lines


def square_lease(name="L", lon=-74.0, lat=39.5, km=10.0) -> LeaseArea:
    dlat = km / 111.32
    dlon = km / (111.32 * math.cos(math.radians(lat)))
    ring = np.array(
        [
            [lon - dlon / 2, lat - dlat / 2],
            [lon + dlon / 2, lat - dlat / 2],
            [lon + dlon / 2, lat + dlat / 2],
            [lon - dlon / 2, lat + dlat / 2],
        ]
    )
    return LeaseArea(name, ring)


# --- geometry -------------------------------------------------------------------


def test_projection_round_trips():
    proj = Projection(-74.0, 39.5)
    lon = np.array([-74.1, -73.9])
    lat = np.array([39.4, 39.6])
    x, y = proj.forward(lon, lat)
    back_lon, back_lat = proj.inverse(x, y)
    assert np.allclose(back_lon, lon)
    assert np.allclose(back_lat, lat)


def test_projection_distance_matches_haversine():
    """Pins the distortion bounds the geo module documents.

    Within a lease (about 30 km) the error is under 0.07%; across a group of
    leases (70 km) it is under 0.15%. Both are far below the grid resolution
    the analysis runs at, but they are measured rather than assumed.
    """
    proj = Projection(-74.0, 39.5)
    worst_near = worst_far = 0.0
    for dlon in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5):
        for dlat in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5):
            if dlon == dlat == 0:
                continue
            x, y = proj.forward(np.array([-74.0, -74.0 + dlon]), np.array([39.5, 39.5 + dlat]))
            planar = math.hypot(x[1] - x[0], y[1] - y[0])
            exact = haversine_m(-74.0, 39.5, -74.0 + dlon, 39.5 + dlat)
            err = abs(planar - exact) / exact
            if exact <= 30_000:
                worst_near = max(worst_near, err)
            worst_far = max(worst_far, err)
    assert worst_near < 7e-4
    assert worst_far < 1.5e-3


def test_projection_preserves_area():
    """Sinusoidal is equal-area, which is what makes the km2 numbers trustworthy."""
    proj = Projection(-74.0, 39.5)
    lease = square_lease(km=20.0)
    # A 20 km square should measure 400 km2 regardless of where it sits.
    for lat in (39.0, 39.5, 40.0):
        moved = LeaseArea("x", lease.ring + np.array([0.0, lat - 39.5]))
        assert moved.area_km2(proj) == pytest.approx(400.0, rel=0.01)


def test_polygon_area_of_a_known_square():
    square = np.array([[0.0, 0.0], [1000.0, 0.0], [1000.0, 1000.0], [0.0, 1000.0]])
    assert polygon_area_m2(square) == pytest.approx(1e6)
    assert polygon_area_m2(square[::-1]) == pytest.approx(1e6)  # winding does not matter


def test_point_in_polygon_handles_edges_and_concavity():
    #  an L shape
    ring = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [1.0, 1.0], [1.0, 2.0], [0.0, 2.0]])
    pts = np.array([[0.5, 0.5], [1.5, 0.5], [1.5, 1.5], [0.5, 1.5], [3.0, 3.0]])
    assert points_in_polygon(pts, ring).tolist() == [True, True, False, True, False]


def test_segment_distance_clamps_to_the_endpoints():
    a, b = np.array([0.0, 0.0]), np.array([10.0, 0.0])
    pts = np.array([[5.0, 3.0], [-4.0, 0.0], [14.0, 0.0]])
    assert segment_distance(pts, a, b).tolist() == [3.0, 4.0, 4.0]
    # A degenerate segment is a point.
    assert segment_distance(np.array([[3.0, 4.0]]), a, a).tolist() == [5.0]


def test_densify_inserts_vertices_without_moving_the_ring():
    ring = np.array([[0.0, 0.0], [1000.0, 0.0], [1000.0, 1000.0], [0.0, 1000.0]])
    dense = densify(ring, 250.0)
    assert len(dense) > len(ring)
    assert polygon_area_m2(dense) == pytest.approx(polygon_area_m2(ring), rel=1e-9)


# --- coverage ---------------------------------------------------------------------


def test_grid_area_matches_the_polygon_area():
    lease = square_lease(km=10.0)
    proj = Projection.centered_on(lease.lons, lease.lats)
    grid = build_grid(lease, 50.0, proj)
    assert grid.area_km2 == pytest.approx(lease.area_km2(proj), rel=0.02)
    assert grid.coverage_fraction == 0.0


def test_a_single_line_covers_a_swath_wide_strip():
    lease = square_lease(km=10.0)
    proj = Projection.centered_on(lease.lons, lease.lats)
    grid = build_grid(lease, 50.0, proj)
    line = SurveyLine(track=np.array([[lease.lons.min(), 39.5], [lease.lons.max(), 39.5]]), swath_m=500.0)
    apply_coverage(grid, [line])
    # Strip area is about swath width times lease width.
    expected = 0.5 * 10.0
    assert grid.covered_km2 == pytest.approx(expected, rel=0.15)


def test_coverage_can_be_filtered_by_sensor_and_year():
    lease = square_lease(km=6.0)
    proj = Projection.centered_on(lease.lons, lease.lats)
    track = np.array([[lease.lons.min(), 39.5], [lease.lons.max(), 39.5]])
    old = SurveyLine(track=track, swath_m=400.0, year=2010, sensor="bathymetry")
    new = SurveyLine(track=track, swath_m=400.0, year=2023, sensor="magnetometer")

    both = build_grid(lease, 50.0, proj)
    apply_coverage(both, [old, new])
    assert both.covered_km2 > 0

    mag_only = build_grid(lease, 50.0, proj)
    apply_coverage(mag_only, [old, new], sensors={"magnetometer"})
    assert mag_only.covered_km2 == pytest.approx(both.covered_km2, rel=0.05)

    recent = build_grid(lease, 50.0, proj)
    apply_coverage(recent, [old], min_year=2018)
    assert recent.covered_km2 == 0.0


def test_connected_components_separates_islands():
    mask = np.zeros((5, 5), dtype=bool)
    mask[0, 0] = mask[0, 1] = True
    mask[4, 4] = True
    comps = connected_components(mask)
    assert sorted(len(c) for c in comps) == [1, 2]


def test_gaps_are_found_ordered_and_filtered_by_size():
    lease = square_lease(km=10.0)
    proj = Projection.centered_on(lease.lons, lease.lats)
    grid = build_grid(lease, 100.0, proj)
    # Cover everything except two strips of different widths.
    grid.covered = grid.inside.copy()
    grid.covered[20:30, :] = False  # wide strip
    grid.covered[60:61, :] = False  # narrow strip
    gaps = find_gaps(grid, lease.name, min_area_km2=0.0)
    assert len(gaps) == 2
    assert gaps[0].area_km2 > gaps[1].area_km2
    big = find_gaps(grid, lease.name, min_area_km2=gaps[1].area_km2 * 1.5)
    assert len(big) == 1


def test_a_fully_covered_lease_has_no_gaps():
    lease = square_lease(km=5.0)
    proj = Projection.centered_on(lease.lons, lease.lats)
    grid = build_grid(lease, 100.0, proj)
    grid.covered = np.ones_like(grid.inside)
    assert find_gaps(grid, lease.name) == []
    assert grid.coverage_fraction == 1.0


def test_shared_projection_puts_every_lease_in_one_frame():
    leases = lease_areas()
    proj = shared_projection(leases)
    grids = [build_grid(lease, 400.0, proj) for lease in leases]
    # Two leases that are far apart must not land on top of each other.
    assert abs(grids[0].origin[0] - grids[2].origin[0]) > 5_000


# --- planning -----------------------------------------------------------------------


@pytest.fixture(scope="module")
def strip_gap():
    """A 10 x 1 km lease with everything covered except a 2 km wide strip."""
    lease = square_lease(km=10.0)
    proj = Projection.centered_on(lease.lons, lease.lats)
    grid = build_grid(lease, 100.0, proj)
    grid.covered = grid.inside.copy()
    grid.covered[:, 20:40] = False
    gaps = find_gaps(grid, lease.name, min_area_km2=0.0)
    return grid, gaps[0]


def test_plan_covers_the_gap_at_the_vessel_spacing(strip_gap):
    grid, gap = strip_gap
    vessel = Vessel(swath_m=200.0, overlap=0.25)
    plan = plan_lines(grid, gap, vessel, heading_deg=0.0)
    assert plan.line_spacing_m == pytest.approx(150.0)
    # Enough lines to cross a 2 km wide gap at 150 m spacing.
    assert len(plan.lines) >= int(2000 / 150) - 1
    assert plan.line_km > 0


def test_best_heading_runs_along_the_long_axis(strip_gap):
    grid, gap = strip_gap
    vessel = Vessel(swath_m=200.0, overlap=0.25, turn_minutes=10.0)
    plan = best_plan(grid, gap, vessel, headings=12)
    # The strip is tall and narrow, so the lines should run roughly north-south.
    assert min(plan.heading_deg, 180 - plan.heading_deg) < 35
    across = plan_lines(grid, gap, vessel, heading_deg=90.0)
    assert plan.hours < across.hours


def test_turn_cost_grows_with_the_number_of_lines(strip_gap):
    grid, gap = strip_gap
    patient = plan_lines(grid, gap, Vessel(turn_minutes=1.0), heading_deg=0.0)
    impatient = plan_lines(grid, gap, Vessel(turn_minutes=15.0), heading_deg=0.0)
    assert impatient.turn_km > patient.turn_km
    assert impatient.hours > patient.hours


def test_lines_are_split_rather_than_run_across_covered_water():
    lease = square_lease(km=10.0)
    proj = Projection.centered_on(lease.lons, lease.lats)
    grid = build_grid(lease, 100.0, proj)
    grid.covered = grid.inside.copy()
    grid.covered[10:20, :] = False
    grid.covered[70:80, :] = False
    gaps = find_gaps(grid, lease.name, min_area_km2=0.0)
    assert len(gaps) == 2
    vessel = Vessel(swath_m=400.0, overlap=0.0)
    # Running north-south across both strips must produce two runs per line.
    merged = plan_lines(grid, gaps[0], vessel, heading_deg=0.0)
    assert all(np.linalg.norm(b - a) < 2_000 for a, b in merged.lines)


def test_campaign_cost_scales_with_the_day_rate(strip_gap):
    grid, gap = strip_gap
    cheap = Vessel(day_rate_usd=10_000.0)
    dear = Vessel(day_rate_usd=100_000.0)
    plan = best_plan(grid, gap, cheap, headings=6)
    a = campaign_cost([plan], cheap)
    b = campaign_cost([plan], dear)
    assert b["cost_usd"] == pytest.approx(a["cost_usd"] * 10)
    assert a["survey_days"] == pytest.approx(plan.hours / cheap.hours_per_day)


def test_prioritize_puts_the_best_value_first(strip_gap):
    grid, gap = strip_gap
    vessel = Vessel()
    good = best_plan(grid, gap, vessel, headings=6)
    bad = plan_lines(grid, gap, vessel, heading_deg=90.0)
    assert prioritize([bad, good])[0] is good


def test_speed_conversion_uses_nautical_miles():
    assert Vessel(speed_kn=1.0).speed_ms == pytest.approx(NM / 3600.0)


# --- sample data and io --------------------------------------------------------------


def test_sample_data_reproduces_its_three_intended_problems():
    leases = lease_areas()
    lines = survey_lines()
    proj = shared_projection(leases)
    results = {}
    for lease in leases:
        # 100 m cells: the stripes left by the 2016 survey are 250 m wide, and
        # a grid coarser than the feature merges them into one blob.
        grid, gaps = analyze(lease, lines, cell_m=100.0, projection=proj, min_gap_km2=0.2)
        results[lease.name] = (grid.coverage_fraction, len(gaps))

    assert results["OCS-A 0001"][0] > 0.99  # fully surveyed
    assert 0.6 < results["OCS-A 0002"][0] < 0.9  # survey abandoned part way
    assert results["OCS-A 0003"][0] < 0.6  # line spacing wider than the swath
    assert results["OCS-A 0003"][1] > 5  # many parallel stripes


def test_geojson_round_trips(tmp_path):
    leases = lease_areas()
    lines = survey_lines()[:5]
    (tmp_path / "l.geojson").write_text(json.dumps(leases_to_geojson(leases)))
    (tmp_path / "s.geojson").write_text(json.dumps(lines_to_geojson(lines)))

    back_leases = load_leases(tmp_path / "l.geojson")
    back_lines = load_lines(tmp_path / "s.geojson")
    assert [x.name for x in back_leases] == [x.name for x in leases]
    assert np.allclose(back_leases[0].ring, leases[0].ring)
    assert back_lines[0].swath_m == lines[0].swath_m
    assert back_lines[0].year == lines[0].year


def test_loader_accepts_multipolygon_and_boem_property_names(tmp_path):
    payload = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"LEASE_NUMBER": "OCS-A 9999", "COMPANY": "Example"},
                "geometry": {
                    "type": "MultiPolygon",
                    "coordinates": [
                        [[[-74.0, 39.0], [-73.9, 39.0], [-73.9, 39.1], [-74.0, 39.1], [-74.0, 39.0]]],
                        [[[-73.5, 39.0], [-73.4, 39.0], [-73.4, 39.1], [-73.5, 39.1], [-73.5, 39.0]]],
                    ],
                },
            }
        ],
    }
    path = tmp_path / "m.geojson"
    path.write_text(json.dumps(payload))
    leases = load_leases(path)
    assert [x.name for x in leases] == ["OCS-A 9999 (1)", "OCS-A 9999 (2)"]
    assert leases[0].operator == "Example"
    assert len(leases[0].ring) == 4  # the repeated closing vertex is dropped


def test_gap_area_is_stable_across_resolutions_but_segmentation_is_not():
    """Total missing area converges quickly; how it splits into gaps does not.

    The 2016 survey leaves 250 m stripes. Every grid from 400 m down agrees on
    the total gap area to within a percent, because the cells that straddle a
    stripe edge average out. The number of distinct gaps does not converge
    until the cell is comfortably smaller than the stripe: at 400 m the
    stripes merge into blobs, and at 200 m they touch diagonally and merge
    into two. From 150 m down the answer is stable.
    """
    leases = lease_areas()
    proj = shared_projection(leases)
    areas = {}
    counts = {}
    for cell in (400.0, 200.0, 150.0, 100.0):
        grid, gaps = analyze(leases[2], survey_lines(), cell_m=cell, projection=proj, min_gap_km2=0.2)
        areas[cell] = grid.gap_km2
        counts[cell] = len(gaps)

    assert max(areas.values()) - min(areas.values()) < 0.01 * min(areas.values())
    assert counts[150.0] == counts[100.0]
    assert counts[400.0] < counts[100.0] / 2


def test_gap_geojson_is_valid_and_carries_area():
    leases = lease_areas()
    proj = shared_projection(leases)
    grid, gaps = analyze(leases[2], survey_lines(), cell_m=100.0, projection=proj, min_gap_km2=0.2)
    fc = gaps_to_geojson(gaps, proj)
    assert fc["type"] == "FeatureCollection"
    first = fc["features"][0]
    assert first["geometry"]["type"] == "Polygon"
    assert len(first["geometry"]["coordinates"][0]) == 5
    assert first["properties"]["area_km2"] > 0


# --- rendering and CLI -------------------------------------------------------------------


def test_render_produces_a_self_contained_page():
    leases = lease_areas()
    lines = survey_lines()
    proj = shared_projection(leases)
    grids = {}
    gaps = []
    for lease in leases:
        g, gg = analyze(lease, lines, cell_m=400.0, projection=proj, min_gap_km2=0.5)
        grids[lease.name] = g
        gaps.extend(gg)
    vessel = Vessel()
    plans = [best_plan(grids[gap.lease], gap, vessel, headings=4) for gap in gaps[:3]]
    html = render_map(leases, lines, grids, gaps, plans, vessel, campaign_cost(plans, vessel))
    assert "<svg" in html
    assert "http://" not in html and "https://" not in html  # no network dependencies
    assert "undefined" not in html
    for lease in leases:
        assert lease.name in html


def test_cli_report_and_sample(tmp_path, capsys):
    assert (
        main(
            [
                "report",
                "--cell",
                "400",
                "--min-gap",
                "1.0",
                "--headings",
                "4",
                "--html",
                str(tmp_path / "m.html"),
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "survey hours" in out
    assert (tmp_path / "m.html").exists()

    assert main(["sample", "--out", str(tmp_path)]) == 0
    assert (tmp_path / "leases.geojson").exists()

    assert (
        main(
            [
                "report",
                "--leases",
                str(tmp_path / "leases.geojson"),
                "--lines",
                str(tmp_path / "surveys.geojson"),
                "--cell",
                "400",
                "--headings",
                "4",
            ]
        )
        == 0
    )
    assert "lease" in capsys.readouterr().out
